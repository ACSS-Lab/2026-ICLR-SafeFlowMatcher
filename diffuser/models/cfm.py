import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import torch
from torch import nn
import torchdiffeq
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdyn.core import NeuralODE
from torch.distributions.normal import Normal
import diffuser.utils as utils
import pdb
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

# For NLL
class SimpleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # expects model(x, cond, t)

    def forward(self, x, t):
        t_batch = torch.full((x.shape[0],), t, device=x.device)
        return self.model(x, None, t_batch)

class CFM(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        # CFM setting
        sigma = 0.0
        self.FM = ConditionalFlowMatcher(sigma=sigma)
        self.node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

        # Get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        # One-shot initialization
        self.one_shot_enabled = False

        # Safety
        self.safety_enabled = False
        self.cbf = None
        self.norm_mins = 0
        self.norm_maxs = 0
        self.safe1 = 0
        self.safe2 = 0

        # Settings for compatibility with diffusion models (Not important for CFM)
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        # manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#
    @torch.no_grad()
    def conditioned_ode_func_record(self, t, x, cond, trajectory_list):
        """
        ODE vector field function with conditioning applied at each step.
        
        t (float or tensor): Current time step in the ODE solver.
        x (tensor): Current state.
        cond (dict): Conditioning dictionary used to fix specific time steps or values.
        trajectory_list (list): List to accumulate the trajectory.
        
        vt (tensor): Vector field output from the model.
        """
        trajectory_list.append(x)  # Append the current state to the trajectory list

        # 1. Apply conditioning to the current state
        x_cond = apply_conditioning(x, cond, self.action_dim)
        
        # 2. Compute the vector field from the conditioned state
        t_batch = torch.full((x.shape[0],), t, device=x.device)
        vt = self.model(x_cond, None, t_batch)

        return vt

    @torch.no_grad()
    def conditioned_ode_func(self, t, x, cond):
        """
        Computes the ODE vector field with conditioning at each step
        """
        # 1. Apply condition to current state
        x_cond = apply_conditioning(x, cond, self.action_dim)
        
        # 2. Compute vector field on the conditioned state
        t_batch = torch.full((x.shape[0],), t, device=x.device)
        vt = self.model(x_cond, None, t_batch)
        
        return vt

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, record_traj=False):
        """
        Generate samples by solving the conditional ODE
        """
        # Initial noise
        x0 = torch.randn(shape).to(self.device)
        
        # Apply condition to initial state
        x0 = apply_conditioning(x0, cond, self.action_dim)

        # Wrapper function for torchdiffeq.odeint (must accept only t and x as arguments)
        if record_traj:
            trajectory_list = []
            ode_fn = lambda t, x: self.conditioned_ode_func_record(t, x, cond, trajectory_list)
        else:
            ode_fn = lambda t, x: self.conditioned_ode_func(t, x, cond)
        
        iter_start = time.time()
        # Solve ODE using wrapper
        traj = torchdiffeq.odeint(
            ode_fn,
            x0,
            torch.linspace(0, 1, self.n_timesteps + 1).to(self.device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        iter_end = time.time()
        iter_time = iter_end - iter_start

        x1 = traj[-1]
        # Apply condition again at the end (for safety)
        x1 = apply_conditioning(x1, cond, self.action_dim)
        
        # pdb.set_trace()
        if record_traj:
            trajectory_list.append(x1) # append last step x
            return x1, torch.stack(trajectory_list, dim=1), [iter_time/self.n_timesteps]
        return x1
    
    @torch.no_grad()
    def p_sample_loop_ode_planning(self, shape, cond, verbose=True, record_traj=False):
        """
        Solve ODE planning with explicit control-corrected RHS (e.g., CBF applied)
        """
        OSI_start = time.time()
        n_timesteps = self.n_timesteps
        # ================ one-shot initialization ================
        if self.one_shot_enabled:
            batch_size = len(cond[0])
            x0_1st_phase = torch.randn(shape).to(self.device)
            x0_1st_phase = apply_conditioning(x0_1st_phase, cond, self.action_dim)
            
            # Obtain velocity field for one-shot
            t_batch = torch.zeros((batch_size,), device=self.device) # same with torch.full((x.shape[0],), t=0, device=x.device)
            v0 = self.model(x0_1st_phase, None, t_batch) 

            # Obtain one-shot prediction (1-step Euler)
            x1_pred = x0_1st_phase.clone()
            x1_pred = x0_1st_phase + v0
            
            x0_2nd_phase = x1_pred
            vs = True  
        # ================ Multi-step Planning ================
        else:
            x0_2nd_phase = torch.randn(shape).to(self.device)
            vs = False
        OSI_end = time.time()
        OSI_time = OSI_end - OSI_start

        x0_2nd_phase = apply_conditioning(x0_2nd_phase, cond, self.action_dim)

        T = n_timesteps + 1

        # Adaptive scheduling
        # time_list = self.adaptive_scheduling(T, device=self.device)
        # Uniform scheduling
        time_list = torch.linspace(0, 1, T).to(self.device)  # [0, 1] for uniform scheduling
        # print(f"Adaptive scheduling: {time_list}")
        
        traj = [x0_2nd_phase]
        
        iter_time = 0
        for i in range(1, T):
            iter_start = time.time()
            # print(f"{i}-th iter / {T} (time: {t_act1 - t_start:.2f}s)", end="\r")
            t_now = time_list[i-1]
            # define dt based on scheduling
            if vs:
                lmbd = 1.0  # bigger give more power to previous step (0,1)
                dt = ((2*lmbd)/(n_timesteps *(n_timesteps + 1))) * (n_timesteps-(i-1))+ (1-lmbd)/(n_timesteps)
            else:
                dt = time_list[i] - t_now
            x_now = traj[-1]

            B = x_now.shape[0]
            t_batch = torch.full((B,), t_now, device=x_now.device)
            # Step forward via some base policy (e.g., learned dynamics model)
            u_raw = self.model(x_now, None, t_batch)  # [B, H, D] - same shape as dx/dt

            # CBF correction
            if self.safety_enabled and self.cbf is not None:
                x_next_naive = x_now + u_raw * dt
                x_corr, _ = self.cbf.apply(x_now, x_next_naive, t=t_now)
                dx = x_corr - x_now
            else:
                dx = u_raw * dt

            x_next = x_now + dx
            x_next = apply_conditioning(x_next, cond, self.action_dim)

            traj.append(x_next)
            iter_end = time.time()
            iter_time += (iter_end - iter_start)

        traj_tensor = torch.stack(traj, dim=1)  # [T, B, H, D]

        if record_traj:
            return traj_tensor[:,T-1,:,:], traj_tensor, [iter_time/n_timesteps]  # sample, diffusion_paths, avg_iter_time
        else:
            return traj_tensor[:,T-1,:,:]               # just sample

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, record_traj=True, **kwargs):
        '''
        conditions : [ (time, state), ... ]
        '''
        # device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        
        if self.safety_enabled: # Planning
            return self.p_sample_loop_ode_planning(shape, cond, record_traj=record_traj, *args, **kwargs)
        else: # Training or Planning without CBF #TODO: separate training and planning
            # return self.p_sample_loop_ode_planning(shape, cond, record_traj=record_traj, *args, **kwargs) # Planning without CBF
            return self.p_sample_loop(shape, cond, record_traj=record_traj, *args, **kwargs) # Training

    @property
    def device(self):
        """
        Get the device where the model's parameters are allocated
        """
        # Assumes the model's parameters are all on the same device.
        return next(self.parameters()).device
    
    #------------------------------------------ training ------------------------------------------#
    
    def loss(self, x, cond):
        x = x.to(self.device)
        batch_size = len(x)

        t = torch.rand(batch_size, device=x.device)
        
        x1 = x.to(self.device)
        x0 = torch.randn_like(x1)

        # Generate xt and flow field ut at time t
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)

        # Apply condition
        xt = apply_conditioning(xt, cond, self.action_dim)

        # Compute vector field
        vt = self.model(xt, None, t) # if there are cond, modify None -> cond

        # Compute loss
        loss, info = self.loss_fn(vt, ut)
        
        return loss, info

    # wthout segementing
    def forward(self, cond, *args, **kwargs): 

        x1, traj, iter_per_time =  self.conditional_sample(cond=cond, *args, **kwargs)

        safe_l, cbf_warn = self.cbf.cbf_nv(x1)
        
        return x1, traj, iter_per_time, safe_l, cbf_warn

    # ==================== with segmenting ==================== #
    # ==================== with segmenting ==================== #
    # ==================== with segmenting ==================== #
    # ==================== with segmenting ==================== #
    @torch.no_grad()
    def conditional_sample1(self, cond, *args, horizon=None, record_traj=True, **kwargs):
        '''
        conditions : [ (time, state), ... ]
        '''
        # device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        
        return self.p_sample_loop_ode_planning1(shape, cond, record_traj=record_traj, *args, **kwargs)

    @torch.no_grad()
    def p_sample_loop_ode_planning1(self, shape, cond, verbose=True, record_traj=False):
        """
        Solve ODE planning with explicit control-corrected RHS (e.g., CBF applied)
        """
        OSI_start = time.time()
        n_timesteps = 0
        # ================ one-shot initialization ================
        if self.one_shot_enabled:
            batch_size = len(cond[0])
            x0_1st_phase = torch.randn(shape).to(self.device)
            x0_1st_phase = apply_conditioning(x0_1st_phase, cond, self.action_dim)
            
            # Obtain velocity field for one-shot
            t_batch = torch.zeros((batch_size,), device=self.device) # same with torch.full((x.shape[0],), t=0, device=x.device)
            v0 = self.model(x0_1st_phase, None, t_batch) 

            # Obtain one-shot prediction (1-step Euler)
            x1_pred = x0_1st_phase.clone()
            x1_pred = x0_1st_phase + v0
            
            x0_2nd_phase = x1_pred
        # ================ Multi-step Planning ================
        else:
            x0_2nd_phase = torch.randn(shape).to(self.device)
        OSI_end = time.time()
        OSI_time = OSI_end - OSI_start

        x0_2nd_phase = apply_conditioning(x0_2nd_phase, cond, self.action_dim)

        T = n_timesteps + 1

        # Adaptive scheduling
        # time_list = self.adaptive_scheduling(T, device=self.device)
        # Uniform scheduling
        time_list = torch.linspace(0, 1, T).to(self.device)  # [0, 1] for uniform scheduling
        # print(f"Adaptive scheduling: {time_list}")
        
        traj = [x0_2nd_phase]
        
        iter_time = 0
        for i in range(1, T):
            iter_start = time.time()
            # print(f"{i}-th iter / {T} (time: {t_act1 - t_start:.2f}s)", end="\r")
            t_now = time_list[i-1]
            # define dt based on scheduling
            vs = True
            if vs:
                lmbd = 1.0  # bigger give more power to previous step (0,1)
                dt = ((2*lmbd)/(n_timesteps *(n_timesteps + 1))) * (n_timesteps-(i-1))+ (1-lmbd)/(n_timesteps)
            else:
                dt = time_list[i] - t_now
            x_now = traj[-1]

            B = x_now.shape[0]
            t_batch = torch.full((B,), t_now, device=x_now.device)
            # Step forward via some base policy (e.g., learned dynamics model)
            u_raw = self.model(x_now, None, t_batch)  # [B, H, D] - same shape as dx/dt

            # CBF correction
            if self.safety_enabled and self.cbf is not None:
                x_next_naive = x_now + u_raw * dt
                x_corr, safe_val = self.cbf.apply(x_now, x_next_naive, t=t_now)
                dx = x_corr - x_now
                self.safe1 = safe_val[0]
                self.safe2 = safe_val[1]
            else:
                dx = u_raw * dt

            x_next = x_now + dx
            x_next = apply_conditioning(x_next, cond, self.action_dim)

            traj.append(x_next)
            iter_end = time.time()
            iter_time += (iter_end - iter_start)

        traj_tensor = torch.stack(traj, dim=1)  # [T, B, H, D]

        if record_traj:
            return traj_tensor[:,T-1,:,:], traj_tensor, [iter_time]  # sample, diffusion_paths, avg_iter_time
        else:
            return traj_tensor[:,T-1,:,:]               # just sample

    @torch.no_grad()
    def conditional_sample2(self, cond, *args, horizon=None, record_traj=True, x1=None, **kwargs):
        '''
        conditions : [ (time, state), ... ]
        '''
        # device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        
        return self.p_sample_loop_ode_planning2(shape, cond, record_traj=record_traj, x1=x1, *args, **kwargs)
    
    @torch.no_grad()
    def p_sample_loop_ode_planning2(self, shape, cond, verbose=True, record_traj=False, x1=None):
        """
        Solve ODE planning with explicit control-corrected RHS (e.g., CBF applied)
        """
        OSI_start = time.time()
        n_timesteps = self.n_timesteps
        
        # x0_2nd_phase = torch.randn(shape).to(self.device)
        x0_2nd_phase = x1
        OSI_end = time.time()
        OSI_time = OSI_end - OSI_start

        x0_2nd_phase = apply_conditioning(x0_2nd_phase, cond, self.action_dim)

        T = n_timesteps + 1

        # Adaptive scheduling
        # time_list = self.adaptive_scheduling(T, device=self.device)
        # Uniform scheduling
        time_list = torch.linspace(0, 1, T).to(self.device)  # [0, 1] for uniform scheduling
        # print(f"Adaptive scheduling: {time_list}")
        
        traj = [x0_2nd_phase]
        
        iter_time = 0
        for i in range(1, T):
            iter_start = time.time()
            # print(f"{i}-th iter / {T} (time: {t_act1 - t_start:.2f}s)", end="\r")
            t_now = time_list[i-1]
            # define dt based on scheduling
            vs = True
            if vs:
                lmbd = 1.0  # bigger give more power to previous step (0,1)
                dt = ((2*lmbd)/(n_timesteps *(n_timesteps + 1))) * (n_timesteps-(i-1))+ (1-lmbd)/(n_timesteps)
            else:
                dt = time_list[i] - t_now
            x_now = traj[-1]

            B = x_now.shape[0]
            t_batch = torch.full((B,), t_now, device=x_now.device)
            # Step forward via some base policy (e.g., learned dynamics model)
            u_raw = self.model(x_now, None, t_batch)  # [B, H, D] - same shape as dx/dt

            # CBF correction
            if self.safety_enabled and self.cbf is not None:
                x_next_naive = x_now + u_raw * dt
                x_corr, safe_val = self.cbf.apply(x_now, x_next_naive, t=t_now)
                dx = x_corr - x_now
                self.safe1 = safe_val[0]
                self.safe2 = safe_val[1]
            else:
                dx = u_raw * dt

            x_next = x_now + dx
            x_next = apply_conditioning(x_next, cond, self.action_dim)

            traj.append(x_next)
            iter_end = time.time()
            iter_time += (iter_end - iter_start)

        traj_tensor = torch.stack(traj, dim=1)  # [T, B, H, D]

        if record_traj:
            return traj_tensor[:,T-1,:,:], traj_tensor, [iter_time/n_timesteps]  # sample, diffusion_paths, avg_iter_time
        else:
            return traj_tensor[:,T-1,:,:]               # just sample
    # ==================== with segmenting ==================== #
    # ==================== with segmenting ==================== #
    # ==================== with segmenting ==================== #
    # ==================== with segmenting ==================== #

    def forward23(self, cond, *args, **kwargs):
        batch_size = len(cond[0])
        horizon = self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        
        # 1. Initialize noisy trajectory with boundary condition
        x0 = torch.randn(shape).to(self.device)
        x0 = apply_conditioning(x0, cond, self.action_dim)
        
        # 2. Predict velocity
        t_batch = torch.zeros((batch_size,), device=self.device)
        v0 = self.model(x0, None, t_batch) 

        # 3. Predict trajectory via 1-step Euler
        x1_pred = x0.clone()
        x1_pred = x0 + v0
        
        # 4. Forecast violation
        t_list, sub_goal_list = self.cbf.forecast_violation(x0, x1_pred)
        
        # 5. Add start & end goals
        sub_goal_pairs = [[0, cond[0]]]
        for t, g in zip(t_list, sub_goal_list):
            if t != 0:
                sub_goal_pairs.append([t, g])
        sub_goal_pairs.append([horizon, cond[horizon - 1]])
        sub_goal_pairs = sorted(sub_goal_pairs, key=lambda x: x[0])

        # 6. Build condition sets per segment
        cond_list = []
        step_list = []
        for i in range(len(sub_goal_pairs) - 1):
            t0, g0 = sub_goal_pairs[i]
            t1, g1 = sub_goal_pairs[i + 1]
            steps = t1 - t0
            step_list.append(steps)
            cond_list.append({0: g0, steps - 1: g1})
        
        # 7. Plan each segment
        x1_list = []
        traj_list = []
        iter_per_time_list = []
        print(f"step: {step_list}")
        for i in range(len(cond_list)):
            #print(f"task {i}/ step: {step_list[i]}, cond: {cond_list[i]}")
            x1_temp, traj_temp, iter_per_time_temp = self.conditional_sample1(cond=cond_list[i], *args, horizon=step_list[i], **kwargs)#@@@@@@@@@@
            x1_list.append(x1_temp)
            traj_list.append(traj_temp)
            iter_per_time_list.append(iter_per_time_temp[0])

        visualize_trajectory(x1_list, self.action_dim,
                            title="CBF-based trajectory planning",
                            save_path="logs/trajectory_segments.png")

        x1 = torch.cat(x1_list, dim=1)
        traj = torch.cat(traj_list, dim=2)
        iter_per_time = [sum(iter_per_time_list)]

        total_cond = {}
        for pair in sub_goal_pairs:
            t, g = pair
            total_cond[t] = g
        total_cond[383] = total_cond.pop(384)

        x1, traj, iter_per_time = self.conditional_sample2(cond=total_cond, *args, x1=x1, **kwargs) 

        safe_l, cbf_warn = self.cbf.cbf_nv(x1)
        
        return x1, traj, iter_per_time, safe_l, cbf_warn

    # ------------------------------------------ NLL calculation ------------------------------------------#
    def compute_nll(self, x1, num_steps=200, exact_div=False, verbose=False):
        """
        Compute negative log-likelihood for the given samples using ODE integration.
        
        Parameters:
        -----------
        x1 : torch.Tensor
            Data samples for which to compute NLL, shape: (batch_size, horizon, transition_dim)
        num_steps : int, optional
            Number of integration steps, default: 200
        exact_div : bool, optional
            Whether to use exact divergence calculation (True) or Hutchinson estimator (False)
        verbose : bool, optional
            Whether to show progress bar
        
        Returns:
        --------
        x0 : torch.Tensor
            Reconstructed initial noise
        nll : torch.Tensor
            Negative log-likelihood value
        """
        device = self.device
        x1 = x1.to(device)
        batch_size = x1.shape[0]
        
        # 시간 스텝 설정
        ts = torch.linspace(0, 1, num_steps+1, device=device)
        dt =  ts[1] - ts[0] # 1 / num_steps
        # 역방향 ODE 적분 (x1 -> x0)
        xt = x1
        sum_log_probs = torch.zeros(batch_size, device=device)
        
        reversed_ts = ts[:-1].flip(0)  # 역순으로 시간 스텝을 뒤집음
        for t in reversed_ts:  # 역순으로 진행
            t_batch = torch.full((batch_size,), t, device=device)
            
            # 모델에서 벡터 필드 예측
            with torch.no_grad():
                vt = self.model(xt, None, t_batch)
            
            # 발산 계산 (∇·v)
            if exact_div:
                div = self._compute_exact_divergence(xt, t_batch)
            else:
                # 근사적 발산 계산 (Hutchinson's trace estimator)
                div = self._estimate_divergence_hutchinson(xt, t_batch)
                print(f"{t}-time div : {div.item()}")
                # print(f"Input data x1 mean: {xt.mean().item()}, std: {xt.std().item()}")
                # print(f"xt_mean: {xt.mean().item()}, xt_std: {xt.std().item()}")
            # 로그 확률 업데이트 (ODE 적분)
            sum_log_probs = sum_log_probs + div * dt
            
            # 상태 업데이트 (Euler 방법)
            xt = xt - vt * dt
        
        # after reverse
        x0 = xt
        # 최종 로그 확률 계산
        log_p0 = -0.5 * (x0 ** 2).reshape(batch_size, -1).sum(1) \
                - 0.5 * np.log(2 * np.pi) * np.prod(x1.shape[1:])
        print(f"log2pi: {-0.5 * (x0 ** 2).reshape(batch_size, -1).sum(1)} /x^2: {- 0.5 * np.log(2 * np.pi) * np.prod(x1.shape[1:])}")
        log_px1 = log_p0 + sum_log_probs
        print(f"log_p0: {log_p0.item()}, log_probs: {sum_log_probs.item()}")
        print(f"x0: {x0.std()}/ {x0.mean()}, x1: {x1.std()}")
        # NLL 계산
        nll = -log_px1.mean()
        nll = nll / (np.prod(x1.shape[1:]) * np.log(2.0))  # Normalize by dimension and log(2)
        print(f"NLL: {nll.item()}")
        
        return x0, nll

    def _compute_exact_divergence(self, x, t):
        """
        벡터 필드의 정확한 발산을 계산합니다.
        
        Parameters:
        -----------
        x : torch.Tensor
            입력 데이터
        t : torch.Tensor
            시간 스텝
            
        Returns:
        --------
        div : torch.Tensor
            계산된 발산 값
        """
        import numpy as np
        batch_size = x.shape[0]
        x_dims = np.prod(x.shape[1:])
        
        # 자동 미분을 위해 requires_grad 설정
        x_with_grad = x.detach().requires_grad_(True)
        
        # 벡터 필드 계산
        v = self.model(x_with_grad, None, t)
        
        # 각 차원에 대한 자코비안 대각 요소 계산
        div = torch.zeros(batch_size, device=x.device)
        
        for i in range(x_dims):
            # 평탄화된 인덱스를 원래 차원으로 변환
            idx = np.unravel_index(i, x.shape[1:])
            idx_tuple = tuple(slice(None) if j == 0 else idx[j-1] for j in range(len(x.shape)))
            
            # i번째 출력에 대한 그래디언트 계산
            v_i = v[idx_tuple]
            grad_v_i = torch.autograd.grad(v_i.sum(), x_with_grad, create_graph=False, retain_graph=True)[0]
            
            # 대각 요소 추출 및 합산
            div_i = grad_v_i[idx_tuple]
            div = div + div_i
        
        return div

    def _estimate_divergence_hutchinson(self, x, t, num_samples=4):
        """
        Hutchinson's trace estimator를 사용하여 벡터 필드의 발산을 추정합니다.
        
        Parameters:
        -----------
        x : torch.Tensor
            입력 데이터
        t : torch.Tensor
            시간 스텝
        num_samples : int, optional
            추정에 사용할 샘플 수, 기본값: 1
            
        Returns:
        --------
        div : torch.Tensor
            추정된 발산 값
        """
        batch_size = x.shape[0]
        div_samples = []
        
        for _ in range(num_samples):
            # 자동 미분을 위해 requires_grad 설정
            x_with_grad = x.detach().requires_grad_(True)
            
            # 랜덤 벡터 생성 (Rademacher 분포)
            epsilon = (torch.randint(0, 2, size=x.shape, device=x.device) * 2 - 1).float()
            
            # 벡터 필드 계산
            v = self.model(x_with_grad, None, t)
            
            # Hutchinson's trace estimator
            # v_eps = (v * epsilon).sum()
            v_eps = (v * epsilon).reshape(batch_size, -1).sum(dim=1)
            # grad_v_eps = torch.autograd.grad(v_eps, x_with_grad, create_graph=False)[0]
            grad_v_eps = torch.autograd.grad(
                    v_eps, x_with_grad, grad_outputs=torch.ones_like(v_eps), create_graph=False
                    )[0]
            div_sample = (grad_v_eps * epsilon).reshape(batch_size, -1)
            # print(div_sample.max().item(), div_sample.min().item(), div_sample.mean().item())
            div_sample = div_sample.sum(1)
            
            div_samples.append(div_sample)
            
        
        # 여러 샘플의 평균
        div = torch.stack(div_samples).mean(0)
        
        # print("epsilon mean/std:", epsilon.mean().item(), epsilon.std().item())
        # print("v mean/std:", v.mean().item(), v.std().item())
        # print("v_eps mean/std:", v_eps.mean().item(), v_eps.std().item())
        # print("grad_v_eps mean/std:", grad_v_eps.mean().item(), grad_v_eps.std().item())
        # print("div mean/std:", div.mean().item(), div.std().item())

        return div

    # def compute_nll(self, x1, num_steps=200):
    #     device = self.device
    #     x1 = x1.to(device)
    #     B, H, D = x1.shape

    #     def log_p0(x):
    #         return Normal(0.0, 1.0).log_prob(x).sum(dim=(1, 2))  # [B]

    #     z = (torch.randint_like(x1, low=0, high=2) * 2 - 1).to(device)

    #     def dynamics(t, states):
    #         x_t, log_det = states
    #         x_t.requires_grad_()

    #         t_batch = torch.full((B,), t.item(), device=device)
    #         ut = self.model(x_t, None, t_batch)

    #         dot = (ut * z).sum()
    #         grad = torch.autograd.grad(dot, x_t, create_graph=False, retain_graph=False)[0]
    #         div = (grad * z).flatten(1).sum(dim=1)

    #         return ut, div

    #     t_grid = torch.linspace(0, 1, num_steps).to(device)
    #     t_grid = torch.flip(t_grid, dims=[0])

    #     y0 = (x1, torch.zeros(B, device=device))

    #     sol_x, sol_log = torchdiffeq.odeint(
    #         dynamics, y0, t_grid, method='dopri5', atol=1e-5, rtol=1e-5
    #     )

    #     x0 = sol_x[-1]
    #     log_det = sol_log[-1]
    #     log_px1 = log_p0(x0) + log_det
    #     nll = -log_px1.mean()
    #     print(f"log_p0: {log_p0(x0)}")
    #     print(f"log_det: {log_det}")
              
    #     return x0, nll
    
    def adaptive_scheduling(self, num_steps, device=None):
        # O((1-t)^3)
        N = num_steps
        weights = [(1 - (i - 1) / N) ** 3 for i in range(1, N)]
        total = sum(weights)
        steps = [w / total for w in weights]  # Normalize so they sum to 1

        t_list = [0.0]
        t = 0.0
        for dt in steps:
            t += dt
            t_list.append(t)

        return torch.tensor(t_list, device=device)


# =========== under is func for visualization ============
def visualize_trajectory(x1_list, action_dim, title="trajectory Visualization", save_path="trajectory_visualization.png"):
    """
    Function to visualize trajectories using position coordinates
    
    Parameters:
    - x1_list: List of trajectory segment tensors
    - action_dim: Index where position dimensions start
    - title: Plot title
    - save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 8))

    num_x1 = len(x1_list)
    if num_x1 > 0:
        x1_1 = x1_list[0]
        pos_y_1 = x1_1[0, :, action_dim].detach().cpu().numpy()
        pos_x_1 = x1_1[0, :, action_dim+1].detach().cpu().numpy()
        plt.plot(pos_x_1, pos_y_1, 'b-', linewidth=2, label='1st segment')
    if num_x1 > 1:
        x1_2 = x1_list[1]
        pos_y_2 = x1_2[0, :, action_dim].detach().cpu().numpy()
        pos_x_2 = x1_2[0, :, action_dim+1].detach().cpu().numpy()
        plt.plot(pos_x_2, pos_y_2, 'g-', linewidth=2, label='2nd segment')
    if num_x1 > 2:
        x1_3 = x1_list[2]
        pos_y_3 = x1_3[0, :, action_dim].detach().cpu().numpy()
        pos_x_3 = x1_3[0, :, action_dim+1].detach().cpu().numpy()
        plt.plot(pos_x_3, pos_y_3, 'r-', linewidth=2, label='3rd segment')
    
    # Mark start and end points
    plt.scatter(pos_x_1[0], pos_y_1[0], color='blue', s=100, marker='o', label='start point')
    if num_x1 == 3:
        plt.scatter(pos_x_3[-1], pos_y_3[-1], color='red', s=100, marker='o', label='end point')
    elif num_x1 == 2:
        plt.scatter(pos_x_2[-1], pos_y_2[-1], color='red', s=100, marker='o', label='end point')
    elif num_x1 == 1:
        plt.scatter(pos_x_1[-1], pos_y_1[-1], color='red', s=100, marker='o', label='end point')
    
    # Mark transition points
    if num_x1 > 1:
        plt.scatter(pos_x_1[-1], pos_y_1[-1], color='purple', s=150, marker='*', label='seg_1')
    if num_x1 > 2:
        plt.scatter(pos_x_2[-1], pos_y_2[-1], color='purple', s=150, marker='*', label='seg_2')
    
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    
    plt.savefig(save_path)
    plt.close()
    
    print(f"Trajectory visualization saved at {save_path}")
    return save_path
