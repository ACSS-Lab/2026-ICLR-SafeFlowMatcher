import torch
from qpth.qp import QPFunction, QPSolvers

class CBF:
    def __init__(
        self, norm_mins, norm_maxs, obstacles,
        cbf_solver='qp', cbf_method='normal', robust_term=0.01, relax_threshold=0.9
    ):
        device = norm_mins.device  # make sure norms are already on device
        self.norm_mins = norm_mins
        self.norm_maxs = norm_maxs
        self.obstacles = obstacles
        self.cbf_solver = cbf_solver
        self.robust_term = robust_term
        self.cbf_method = cbf_method
        self.relax_threshold = relax_threshold
        self.device = device

        # Precompute normalization factors
        self.xr = 2 / (self.norm_maxs[1] - self.norm_mins[1])
        self.yr = 2 / (self.norm_maxs[0] - self.norm_mins[0])
    
    @torch.no_grad()
    def compute_cbf_constraint_robust(self, order, dx, dy):
        """
        Robust CBF constraint for n-th order CBFs.
        """
        if order < 1:
            raise ValueError("Order must be at least 1")
        
        b = dy**order + dx**order - 1 - self.robust_term
        L1 = order * dy**(order-1) / self.yr
        L2 = order * dx**(order-1) / self.xr

        G = torch.cat([-L1, -L2], dim=1).unsqueeze(1)
        h = b
        safe = torch.min(b + self.robust_term)
        
        return G, h, safe
    
    @torch.no_grad()
    def compute_cbf_constriant_relax(self, order, dx, dy, t, sign):
        """
        Robust CBF constraint for n-th order CBFs.
        """
        if order < 1:
            raise ValueError("Order must be at least 1")
        
        b = dy**order + dx**order - 1 - self.robust_term
        L1 = order * dy**(order-1) / self.yr
        L2 = order * dx**(order-1) / self.xr

        rx0 = torch.zeros_like(L1)
        rx1 = sign * torch.ones_like(L1)
        G = torch.cat([-L1, -L2, rx1, rx0], dim=1).unsqueeze(1)
        h = b
        safe = torch.min(b + self.robust_term)

        return G, h, safe

    @torch.no_grad()
    def compute_cbf_constraint_time_varying(self, order, dx, dy, t, t_bias, a):
        """
        Time-varying CBF constraint for n-th order CBFs.
        """
        if order < 1:
            raise ValueError("Order must be at least 1")
        
        s = torch.sigmoid(a*(t - t_bias))  # scalar
        Lfb = a * s * (1 - s)              # time-varying Lie derivative approx.

        b  = dy**order + dx**order - s - self.robust_term
        L1 = order * dy**(order-1) / self.yr
        L2 = order * dx**(order-1) / self.xr

        G = torch.cat([-L1, -L2], dim=1).unsqueeze(1)
        h = Lfb + b
        safe = torch.min(b + self.robust_term)
        return G, h, safe
    
    @torch.no_grad()
    def solve_qp(self, u_ref, G, h):
        """
        GP (total dim=2)
        min ‖u - u_ref‖²
        s.t. G u ≤ h
        """
        q = -u_ref[:, 2:4].to(self.device)      # [B, 2]: position increment vector -(Δx, Δy)
        
        Q = torch.eye(2, device=self.device).unsqueeze(0).expand(u_ref.size(0), -1, -1) # Weight matrix, I
        e = torch.empty(0, device=self.device)  # no equality constraints

        out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)
        return out

    def solve_qp_relax(self, u_ref, G, h):
        """
        QP with relaxation variables r1, r2 (total dim=4)
        min ‖u - u_ref‖²
        s.t. G [u; r] ≤ h
        """
        q_u = -u_ref[:, 2:4]                    # [B, 2]: position increment vector -(Δx, Δy)
        q_r = torch.zeros_like(q_u)             # [B, 2]
        q = torch.cat([q_u, q_r], dim=1)        # [B, 4]

        Q = torch.eye(4, device=self.device).unsqueeze(0).expand(u_ref.size(0), 4, 4) # Weight matrix, Is
        e = torch.empty(0, device=self.device)  # no equality constraints

        out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)
        return out
    
    def solve_closed_form(self, ref, G, h):
        """
        Closed-form solution for the CBF QP.
        """
        pass

    def solve_closed_from_relax(self, ref, G, h):
        """
        Closed-form solution for the CBF QP with relaxation.
        """
        pass

    @torch.no_grad()
    def apply(self, x, xp1, t=None):
        # remove the leading batch‐of‐1 dim
        x   = x.squeeze(0)    # [B, state_dim]
        xp1 = xp1.squeeze(0)  # [B, state_dim]

        # desired increment
        ref = xp1 - x         # [B, state_dim]

        G_list, h_list, safe_vals = [], [], []

        for obs in self.obstacles:
            # normalize obstacle center once
            cx, cy = obs['center']
            off_x = 2*(cx - 0.5 - self.norm_mins[1])/(self.norm_maxs[1]-self.norm_mins[1]) - 1
            off_y = 2*(cy - 0.5 - self.norm_mins[0])/(self.norm_maxs[0]-self.norm_mins[0]) - 1

            # extract the normalized y,x coords (same as apply_test)
            dy = (x[:,2:3] - off_y)/self.yr   # [B,1]
            dx = (x[:,3:4] - off_x)/self.xr   # [B,1]

            # Parameters for CBF
            # Relax
            a = 1
            sign = 100.0 if (t is not None and t <= self.relax_threshold) else 0.0
            # Time-varying
            t_bias = self.relax_threshold
            t_bias = 0.90

            if self.cbf_method == 'robust':
                G_i, h_i, safe_i = self.compute_cbf_constraint_robust(obs['order'], dx, dy)
            elif self.cbf_method == 'relax':
                G_i, h_i, safe_i = self.compute_cbf_constriant_relax(obs['order'], dx, dy, t, sign)
            elif self.cbf_method == 'time':
                G_i, h_i, safe_i = self.compute_cbf_constraint_time_varying(obs['order'], dx, dy, t, t_bias, a)
            else:
                raise ValueError(f"Unknown CBF method '{self.cbf_method}'")

            G_list.append(G_i)
            h_list.append(h_i)
            safe_vals.append(safe_i)

        # if you have no obstacles, just apply the reference control
        if not G_list:
            out = ref[:,2:4]
        else:
            G = torch.cat(G_list, dim=1)  # [B, num_obs, dim]
            h = torch.cat(h_list, dim=1)  # [B, num_obs]

            if self.cbf_solver == 'qp':
                if self.cbf_method == 'robust':
                    out = self.solve_qp(ref, G, h)
                elif self.cbf_method == 'relax':
                    out = self.solve_qp_relax(ref, G, h)
                elif self.cbf_method == 'time':
                    out = self.solve_qp(ref, G, h)
                else:
                    raise ValueError(f"Unknown CBF method '{self.cbf_method}'")
            # elif self.method == 'closed_form':
            #     TODO: out = self.solve_closed_form(ref, G, h)
            else:
                raise ValueError(f"Unknown CBF solver {self.cbf_solver}")
            
        # rebuild the next‐state
        rt = xp1.clone()
        rt[:,2:4] = x[:,2:4] + out[:, :2]
        return rt.unsqueeze(0), safe_vals