#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:10:17 2023

@author: oscar
"""

#!/usr/bin/env python

import sys
sys.path.append('/home/regmed/dregmed/vis_to_nav/src/vis_nav/vis_nav')


import os
import time
import yaml
import numpy as np
from tqdm import tqdm
from collections import deque
# from sklearn.metrics import mean_squared_error
import torch
#from SAC.DRL import SAC
from DRL import SAC
import rclpy
import threading
from env_lab import GazeboEnv, Image_subscriber, Odom_subscriber, LaserScan_subscriber, Velodyne_subscriber, DepthImage_subscriber



def main():
    rclpy.init(args=None)

    path = os.getcwd()
    yaml_path = os.path.join(path, 'src/vis_nav/vis_nav/config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ##### Individual parameters for each model ######
    model = 'SAC'
    mode_param = config[model]

    model_name = mode_param['name']  # gtrl
    policy_type = mode_param['actor_type'] # GaussianTransformer
    critic_type = mode_param['critic_type'] # CNN
    transformer_block = mode_param['block'] # 2
    transformer_head = mode_param['head'] # 4

    ###### Default parameters for DRL ######
    max_steps = config['MAX_STEPS'] # 300
    max_episodes = config['MAX_EPISODES'] #500
    lr_a = config['LR_A'] # 0.001
    lr_c = config['LR_C'] # 0.001
    gamma = config['GAMMA'] # 0.999 
    tau = config['TAU'] # 0.005
    policy_freq = config['ACTOR_FREQ'] # 1
    buffer_size = config['BUFFER_SIZE'] # 20000
    frame_stack = config['FRAME_STACK'] # 4


    ##### Attention #####
    policy_attention_fix = config['P_ATTENTION_FIX'] #False whether fix the weights and bias of policy attention
    critic_attention_fix = config['C_ATTENTION_FIX'] #False whether fix the weights and bias of value attention

    ##### Human Intervention #####
    pre_buffer = config['PRE_BUFFER'] #False Human expert buffer
    ##### Entropy ######
    auto_tune = config['AUTO_TUNE'] #True
    alpha = config['ALPHA'] # 1.0
    lr_alpha = config['LR_ALPHA'] #0.0001

    ##### Environment ######
    seed = config['SEED'] #525
    env_name = config['ENV_NAME'] # "RRC"
    linear_cmd_scale = config['L_SCALE'] # 0.5
    angular_cmd_scale = config['A_SCALE'] # 2



    env = GazeboEnv()
    odom_subscriber = Odom_subscriber()
    #image_subscriber = Image_subscriber()
    image_subscriber = DepthImage_subscriber()
    laserScan_subscriber = LaserScan_subscriber()
    velodyne_subscriber = Velodyne_subscriber()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(image_subscriber)
    executor.add_node(laserScan_subscriber)
    executor.add_node(velodyne_subscriber)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = odom_subscriber.create_rate(2)
    time.sleep(5)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env.seed(seed)

    state, _ = env.reset()
    state_dim = state.shape
    action_dim = 2
    physical_state_dim = 2 # Polar coordinate
    max_action = 1

    # Initialize the agent
    ego = SAC(action_dim, physical_state_dim, policy_type, critic_type, policy_attention_fix,
                critic_attention_fix, pre_buffer, seed, lr_c, lr_a, lr_alpha,
                buffer_size, tau, policy_freq, gamma, alpha, block=transformer_block,
                head=transformer_head, automatic_entropy_tuning=auto_tune)
    name = 'drl00_reward187_seed1991'
    env.get_logger().info(f'Let s go with {name}!!')
    ego.load (name,directory="./final_models")

    # Create evaluation data store

    ep_real = 0
    done = False
    target = False
    reward_list = []
    reward_heuristic_list = []
    reward_action_list = []
    reward_freeze_list = []
    reward_target_list = []
    reward_collision_list = []
    reward_mean_list = []

    pedal_list = []
    steering_list = []

    total_timestep = 0
    ep_real = 0
    
    # Begin the testing loop
    try:
        while rclpy.ok():
            for ep in tqdm(range(0, max_episodes), ascii=True):
                episode_reward = 0
                episode_heu_reward = 0.0
                episode_act_reward = 0.0
                episode_tar_reward = 0.0
                episode_col_reward = 0.0
                episode_fr_reward = 0.0
                s_list = deque(maxlen=frame_stack)
                if not target : 
                    s, goal = env.reset()
                else:
                    env.delete_entity('goal')
                    env.change_goal()
                    env.spawn_entity()
                #s, goal = env.reset()
                for i in range(4):
                    s_list.append(s)

                state = np.concatenate((s_list[-4], s_list[-3], s_list[-2], s_list[-1]), axis=-1)

                for timestep in range(max_steps):
                    # On termination of episode
                    if timestep == 0:
                        action = ego.choose_action(np.array(state), np.array(goal[:2])).clip(-max_action, max_action)
                        a_in = [(action[0] + 1) * linear_cmd_scale, action[1] * angular_cmd_scale]
                        s_, _, _, _, _, _ , reward, done, goal, target = env.step(a_in, timestep)        
                        state = np.concatenate((s_, s_, s_, s_), axis=-1)
                        for i in range(4):
                            s_list.append(s_)           

                        if done:
                            print("Bad Initialization, skip this episode.")
                            break

                        continue
                    
                    if done or timestep == max_steps-1:
                        ep_real += 1
            
                        done = False

                        reward_list.append(episode_reward)
                        reward_mean_list.append(np.mean(reward_list[-20:]))
                        reward_heuristic_list.append(episode_heu_reward)
                        reward_action_list.append(episode_act_reward)
                        reward_target_list.append(episode_tar_reward)
                        reward_collision_list.append(episode_col_reward)
                        reward_freeze_list.append(episode_fr_reward)


                        pedal_list.clear()
                        steering_list.clear()
                        total_timestep += timestep 
                        print('\n',
                            '\n',
                            'Robot: ', 'Scout',
                            'Episode:', ep_real,
                            'Step:', timestep,
                            'Tottal Steps:', total_timestep,
                            'R:', episode_reward,
                            'Overak R:', reward_mean_list[-1],
                            'Lr_a:', lr_a,
                            'Lr_c', lr_c,
                            'seed:', seed,
                            'Env:', env_name,
                            "Filename:", model_name,
                            '\n')
                        break

                    
                    action = ego.choose_action(np.array(state), np.array(goal[:2])).clip(-max_action, max_action)
                    a_in = [(action[0] + 1) * linear_cmd_scale, action[1]*angular_cmd_scale]
                    pedal_list.append(round((action[0] + 1)/2,2))
                    steering_list.append(round(action[1],2))

                    #last_goal = goal
                    s_, r_h, r_a, r_f, r_c, r_t, reward, done, goal, target = env.step(a_in, timestep)

                    episode_reward += reward
                    episode_heu_reward += r_h
                    episode_act_reward += r_a
                    episode_fr_reward += r_f
                    episode_col_reward += r_c
                    episode_tar_reward += r_t
                    next_state = np.concatenate((s_list[-3], s_list[-2], s_list[-1], s_), axis=-1)
                    # Update the counters
                    state = next_state
                    s_list.append(s_)

    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
    executor_thread.join()
if __name__ == '__main__':
    main() #call the main function