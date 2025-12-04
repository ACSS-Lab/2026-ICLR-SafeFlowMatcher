from collections import namedtuple
# import numpy as np
import torch
import einops
import pdb
from diffuser.models.cbf import CBF

import diffuser.utils as utils
# from diffusion.datasets.preprocessing import get_policy_preprocess_fn

Trajectories = namedtuple('Trajectories', 'actions observations')
# GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations value')

class Policy:

    def __init__(self, diffusion_model, normalizer, args):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

        # Enable control barrier function
        device = next(diffusion_model.parameters()).device
        norm_mins = torch.tensor(normalizer.normalizers['observations'].mins, device=device)
        norm_maxs = torch.tensor(normalizer.normalizers['observations'].maxs, device=device)

        self.diffusion_model.one_shot_enabled = args.one_shot_enabled

        self.diffusion_model.safety_enabled = args.safety_enabled
        self.diffusion_model.cbf = CBF(norm_mins, norm_maxs, args)
        self.n_diffusion_steps = args.n_diffusion_steps
        self.cbf_method = args.cbf_method

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device=self.device)
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1):
        conditions = self._format_conditions(conditions, batch_size)

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        self.diffusion_model.norm_mins = self.normalizer.normalizers['observations'].mins
        self.diffusion_model.norm_maxs = self.normalizer.normalizers['observations'].maxs
        sample, diffusion, iter_time = self.diffusion_model(conditions, self.n_diffusion_steps)
        safe_l = self.diffusion_model.cbf.cbf_nv(sample)
        c_smooth, s_smooth = self.diffusion_model.cbf.calc_smooth(sample)

        ########################################################## calculate number of traps
        num_trap = utils.local_trap(diffusion, self.diffusion_model.cbf, batch_idx=0, n_timesteps=self.n_diffusion_steps-1)

        sum_elbo = 0
        # end#########################################################################

        sample = utils.to_np(sample)
        diffusion = utils.to_np(diffusion)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        normed_diffusion = diffusion[:,:,:,self.action_dim:]
        diffusions = self.normalizer.unnormalize(normed_diffusion, 'observations')

        trajectories = Trajectories(actions, observations)
        #return action, trajectories, diffusions, self.diffusion_model.safe1, self.diffusion_model.safe2, sum_elbo, num_trap, iter_time, cbf_warn
        return action, trajectories, diffusions, safe_l[0], safe_l[1], sum_elbo, num_trap, iter_time, c_smooth, s_smooth
