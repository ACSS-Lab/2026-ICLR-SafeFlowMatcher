import json
import numpy as np
from os.path import join, dirname
import pdb
import csv
import time
import os

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
import torch

# fix seed
# import random
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
# fix seed

# diffusion model:
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --logbase logs --method base
# cfm model:
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 --logbase logs --method cfm

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'
    method: str = 'cfm'


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer, args)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def smooth(diffusion):
    steps, horizon = diffusion.shape[0], diffusion.shape[1]
    diffusion_copy = diffusion.copy()
    for i in range(steps - 20, steps, 1):
        for j in range(5, horizon, 1):
            diffusion_copy[i,j,0:2] = np.mean(diffusion[i, j-5:j, 0:2], axis=0)
    
    return diffusion_copy

# --------------------------------- csv header ----------------------------------#
# save results to csv file
csv_path = join(args.savepath, 'results.csv')
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['iter', 'safe1', 'safe2', 'trap1', 'trap2', 'score', 'itertime', 'success'])

#---------------------------------- main loop ----------------------------------#
safe1_batch, safe2_batch = [], []
score_batch = []
elbo_batch = []
is_trap1, is_trap2 = 0, 0
num_trap1, num_trap2 = 0, 0
c_smooth_batch, s_smooth_batch = [], []
num_success = 0
iter_time_batch = []

TOTAL_TEST_ITER = 1
for iter in range(1, TOTAL_TEST_ITER+1):   # num of testing runs
    print("Total test iteration: ", iter, f"/{TOTAL_TEST_ITER}")

    observation = env.reset()    #array([ 0.94875744,  8.93648809, -0.01347715,  0.06358764])
    observation = np.array([ 0.94875744,  2.93648809, -0.01347715,  0.06358764])   # fix the initial position and final destination for comparison (not needed for general testing)
    env.set_state(observation[0:2], observation[2:4]) ############################################################ same as the last line

    if args.conditional:
        print('Resetting target')
        env.set_target()

    ## set conditioning xy position to be the goal
    target = env._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    env_step = env.max_episode_steps
    # env_step = 384 #env.max_episode_steps # real need is 384 because the plan horizon is only 384
    for t in range(env_step):

        state = env.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:

            cond[0] = observation
            action, samples, diffusion_paths, safe1, safe2, elbo, num_trap, iter_time, c_smooth, s_smooth = policy(cond, batch_size=args.batch_size)
            elbo_batch.append(elbo)
            safe1_val, safe2_val = safe1, safe2
            
            actions = samples.actions[0]
            sequence = samples.observations[0]
            diffusion_paths = diffusion_paths[0]
            
            ##################################################start saving videos/images
            # Save the composite image of all trajectories
            if True: # if you want to save result trajectories
                fullpath = join(args.savepath, f'results/{iter}.png')
                os.makedirs(dirname(fullpath), exist_ok=True)
                renderer.composite(fullpath, samples.observations, ncol=1)
                
                # Save the next step composite image of all trajectoiries
                """
                for checking velocity of each way-points, 
                """
                # fullpath = join(args.savepath, f'{iter}_next.png')
                # next_time_observation = samples.observations.copy()
                # for i in range(next_time_observation.shape[1]-1):
                #     next_time_observation[:, i, :2] += samples.observations[:, i+1, 2:4] * 1e-2   # next pos = curr pos + next vel * 1e-2(time interval)
                # renderer.composite(fullpath, next_time_observation, ncol=1)

            # Save the diffusion process as a video
            if False: # if you want to save diffusion process video
                renderer.render_diffusion(join(args.savepath, f'diffusion_{iter}.mp4'), diffusion_paths)
            
            # Save individual frames of the diffusion process
            # diff_step = diffusion_paths.shape[0]  
            # makedirs(join(args.savepath, 'png'))
            # for kk in range(diff_step):
            #     imgpath = join(args.savepath, f'png/{kk}.png')
            #     renderer.composite(imgpath, diffusion_paths[kk:kk+1], ncol=1)
            
            # Save individual state's movement of the diffusion process
            # diff_step = diffusion_paths.shape[0]  
            # makedirs(join(args.savepath, 'state'))
            # for k in range(258):
            #     imgpath = join(args.savepath, f'state/{k}.png')
            #     renderer.composite_state(imgpath, diffusion_paths[:,k], ncol=1)
            #################################################end saving videos/images

        ##################################################start planning
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            
        # can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        if t==0:
            prev_action = action.copy()
            prev_state = state.copy()
        # if t < 400:
        #     print(f"t: {t} | action: {action} | next_waypoint: {next_waypoint} | state: {state}")
        #     print(f"v dt: {(state[2] - prev_state[2]) / prev_action[0]} / {(state[3] - prev_state[3]) /prev_action[1]}")
        #     print(f"x dt: {(state[0] - prev_state[0]) / state[2]} / {(state[1] - prev_state[1]) / state[3]}")
        prev_action = action.copy()
        prev_state = state.copy()

        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        ##################################################end planning

        ##################################################start logging
        # print(
        #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        #     f'{action}'
        # )

        # if 'maze2d' in args.dataset:
        #     xy = next_observation[:2]
        #     goal = env.unwrapped._target
        #     print(
        #         f'maze | pos: {xy} | goal: {goal}'
        #     )

        ## update rollout observations
        rollout.append(next_observation.copy())
        #print(len(rollout), rollout[0].shape, [rollout[0]])

        # logger.log(score=score, step=t)
        ##################################################end logging

        ##################################################start saving videos/images
        # if t % args.vis_freq == 0 or terminal:
        #     fullpath = join(args.savepath, f'{t}.png')

        #     if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


        #     # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        #     ## save rollout thus far
        #     renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)   ## debug

        #     # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        #     # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)
        ##################################################end saving videos/images

        if terminal:
            break

        observation = next_observation
    
    ######################
    arr = np.expand_dims(np.stack(rollout), axis=0)
    if False: # if you want to save rollout image
        fullpath = join(args.savepath, f'rollout_{iter}.png')
        renderer.composite(fullpath, arr, ncol=1)
    
    
    ##################################################start statistics calculation
    if reward > 0.95:
        num_success = num_success + 1

    if num_trap >= 1:
        is_trap1 = 1
        num_trap1 += 1
    else:
        is_trap1 = 0
    
    if num_trap >= 2:
        is_trap2 = 1
        num_trap2 += 1
    else:
        is_trap2 = 0

    safe1_batch.append(safe1_val)
    safe2_batch.append(safe2_val)
    score_batch.append(score)
    iter_time_batch.append(iter_time)
    c_smooth_batch.append(c_smooth.item())
    s_smooth_batch.append(s_smooth.item())
    ##################################################end statistics calculation

    # logger.finish(t, env.max_episode_steps, score=score, value=0)

    ##################################################start per-iter statistics calculation
    # per-iter statistics calculation
    score_val    = float(score)
    itertime_val = float(iter_time[0])
    success_val  = 1 if reward > 0.95 else 0

    # for time saving skip it
    # append mode to write one line to csv file 
    # with open(csv_path, 'a', newline='') as csv_file: 
    #     writer = csv.writer(csv_file)
    #     writer.writerow([
    #         iter,
    #         safe1_val,
    #         safe2_val,
    #         is_trap1,
    #         is_trap2,
    #         score_val,
    #         itertime_val,
    #         success_val
    #     ])
    ##################################################end per-iter statistics calculation

# --------------------------------- statistics calculation ----------------------------------#

iter_time_batch = np.array(iter_time_batch)
iter_time_avg = np.mean(iter_time_batch, axis=0)

# pdb.set_trace()
print("======================results======================")
# elbo_batch = np.array(elbo_batch)
# print("elbo mean: ", np.mean(elbo_batch))
# print("elbo std: ", np.std(elbo_batch))

score_batch = np.array(score_batch)
print(f"safe1: min: {np.min(safe1_batch):.4f}/ {np.mean(safe1_batch):.4f} ± {np.std(safe1_batch):.4f}")
print(f"safe2: min: {np.min(safe2_batch):.4f}/ {np.mean(safe2_batch):.4f} ± {np.std(safe2_batch):.4f}")
print(f"trap1: {num_trap1} / {TOTAL_TEST_ITER}")
print(f"trap2: {num_trap2} / {TOTAL_TEST_ITER}")
print(f"c-smooth: {np.mean(c_smooth_batch):.4f} ± {np.std(c_smooth_batch):.4f}")
print(f"s-smooth: {np.mean(s_smooth_batch):.4f} ± {np.std(s_smooth_batch):.4f}")
print(f"score: {np.mean(score_batch):.5f} ± {np.std(score_batch):.5f}")
print(f"avg iter time: {iter_time_avg[0]}")
print(f"number of success: {num_success} / {TOTAL_TEST_ITER}")
print("=======================end=========================")

# --------------------------------- plot ----------------------------------#
try:
    import matplotlib.pyplot as plt
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot safe1 data
    ax1.set_title('Safety Specification 1')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Safety Value')
    ax1.plot(range(len(safe1_batch)), safe1_batch, 'b-', label='Safe1')
    ax1.grid(True)
    ax1.legend()
    
    # Plot safe2 data
    ax2.set_title('Safety Specification 2')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Safety Value')
    ax2.plot(range(len(safe2_batch)), safe2_batch, 'r-', label='Safe2')
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    imgpath = join(args.savepath, 'safety_stats.png')
    plt.savefig(imgpath, dpi=300, bbox_inches='tight')
    plt.close('all')  # Properly close all figures
    
except Exception as e:
    print(f"Warning: Could not create plots: {str(e)}")

# Save result as a json file
try:
    json_path = join(args.savepath, 'rollout.json')
    json_data = {
        'safety_stats': {
            'safe1_mean': float(np.mean(safe1_batch)),
            'safe2_mean': float(np.mean(safe2_batch)),
            'safe1_std': float(np.std(safe1_batch)),
            'safe2_std': float(np.std(safe2_batch)),
            'safe1_min': float(np.min(safe1_batch)),
            'safe2_min': float(np.min(safe2_batch))
        },
        'smooth_stats': {
            'c_smooth_mean': float(np.mean(c_smooth_batch)),
            's_smooth_mean': float(np.mean(s_smooth_batch)),
            'c_smooth_std': float(np.std(c_smooth_batch)),
            's_smooth_std': float(np.std(s_smooth_batch)),
        },
        'score_stats': {
            'mean': float(np.mean(score_batch)),
            'std': float(np.std(score_batch)),
            'min': float(np.min(score_batch)),
            'max': float(np.max(score_batch))
        },
        'iter_time_batch': {
            'mean': float(np.mean(iter_time_batch)),
            'std': float(np.std(iter_time_batch)),
            'min': float(np.min(iter_time_batch)),
            'max': float(np.max(iter_time_batch))
        },
        'local_trap1': f'{num_trap1}/{TOTAL_TEST_ITER}',
        'local_trap2': f'{num_trap2}/{TOTAL_TEST_ITER}',
        'denoising_step': args.n_diffusion_steps,
        # 'step': int(t),
        # 'return': float(total_reward),
        # 'term': bool(terminal),
        # 'epoch_diffusion': int(diffusion_experiment.epoch),
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, sort_keys=True)
except Exception as e:
    print(f"Warning: Could not save json file: {str(e)}")
