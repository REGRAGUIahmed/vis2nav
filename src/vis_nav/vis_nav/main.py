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
import statistics
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from collections import deque
import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error

import torch

from DRL import SAC
import rclpy
import threading

from env_lab import GazeboEnv, Odom_subscriber, LaserScan_subscriber, DepthImage_subscriber, Image_fish_subscriber, Image_subscriber


def evaluate(env,frame_stack,network, max_steps,state,timestep, linear_cmd_scale,angular_cmd_scale, max_action,  eval_episodes=10, epoch=0):
    obs_list = deque(maxlen=frame_stack)
    env.collision = 0
    ep = 0
    avg_reward_list = []
    while ep < eval_episodes:
        count = 0
        obs, goal = env.reset()
        done = False
        avg_reward = 0.0

        for i in range(4):
            obs_list.append(obs)

        observation = np.concatenate((obs_list[-4], obs_list[-3], obs_list[-2], obs_list[-1]), axis=-1)

        while not done and count < max_steps:
            
            if count == 0:
                action = network.choose_action(np.array(state), np.array(goal[:2]), evaluate=True).clip(-max_action, max_action)
                a_in = [action[0]/4 + 0.25 , action[1]*1.0]
                #a_in = [(action[0] + 1) * linear_cmd_scale, action[1] * angular_cmd_scale]
                last_goal = goal
                obs_, _, _, _, _, _ , _, done, goal, target = env.step(a_in, timestep)        
                observation = np.concatenate((obs_, obs_, obs_, obs_), axis=-1)
                
                for i in range(4):
                    obs_list.append(obs_)           

                if done:
                    env.get_logger().info("\n..............................................")
                    env.get_logger().info("Bad Initialization, skip this episode.")
                    env.get_logger().info("..............................................")
                    ep -= 1
                    if not target :
                        env.collision -= 1
                    break

                count += 1
                continue
            
            act = network.choose_action(np.array(observation), np.array(goal[:2]), evaluate=True).clip(-max_action, max_action)
            #a_in = [(act[0] + 1) * linear_cmd_scale, act[1]*angular_cmd_scale]
            a_in = [action[0]/4 + 0.25 , action[1]*1.0]
            obs_, _, _, _, _, _, reward, done, goal, target = env.step(a_in, count)        
            avg_reward += reward
            observation = np.concatenate((obs_list[-3], obs_list[-2], obs_list[-1], obs_), axis=-1)
            obs_list.append(obs_)
            count += 1
        
        ep += 1
        avg_reward_list.append(avg_reward)
        env.get_logger().info("\n..............................................")
        env.get_logger().info("%i Loop, Steps: %i, Avg Reward: %f, Collision No. : %i " % (ep, count, avg_reward, env.collision))
        env.get_logger().info("..............................................")
    reward = statistics.mean(avg_reward_list)
    col = env.collision
    env.get_logger().info("\n..............................................")
    env.get_logger().info("Average Reward over %i Evaluation Episodes, At Epoch: %i, Avg Reward: %f, Collision No.: %i" % (eval_episodes, epoch, reward, col))
    env.get_logger().info("..............................................")
    return reward, col

def plot_animation_figure(env_name, lr_a, lr_c, ep_real, block, head,
                          reward_list, reward_mean_list,model_name,desc):

    fig = plt.figure()
    plt.title(env_name + ' ' + str("SAC") + ' Lr_a: ' + str(lr_a) + ' Lr_c: ' + str(lr_c))
    plt.xlabel('Episode')
    plt.ylabel('Overall Reward')
    plt.plot(np.arange(ep_real), reward_list)
    plt.plot(np.arange(ep_real), reward_mean_list)
    plt.tight_layout()

    plt.savefig('/home/regmed/dregmed/vis_to_nav/results/plot_'+ model_name +str(block)+str(head)+'_'+desc+'.png')
    plt.close(fig)



    
def main():
    rclpy.init(args=None)
    path = os.getcwd()
    yaml_path = os.path.join(path, 'src/vis_nav/vis_nav/config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ##### Individual parameters for each model ######
    model = 'GoT-SAC' #'GoT-SAC'
    mode_param = config[model]
    desc=config['DESC'] #describe what are you dowing
    model_name = mode_param['name']  # gtrl
    policy_type = mode_param['actor_type'] # GaussianTransformer
    critic_type = mode_param['critic_type'] # CNN
    transformer_block = mode_param['block'] # 2
    transformer_head = mode_param['head'] # 4
    ###### Default parameters for DRL ######
    max_steps = config['MAX_STEPS'] # 300
    max_episodes = config['MAX_EPISODES'] #500
    batch_size = config['BATCH_SIZE'] #32
    lr_a = config['LR_A'] # 0.001
    lr_c = config['LR_C'] # 0.001
    gamma = config['GAMMA'] # 0.999 
    tau = config['TAU'] # 0.005
    policy_freq = config['ACTOR_FREQ'] # 1
    buffer_size = config['BUFFER_SIZE'] # 20000
    frame_stack = config['FRAME_STACK'] # 4
    plot_interval = config['PLOT_INTERVAL'] # 1
    ##### Evaluation #####
    save_threshold = config['SAVE_THRESHOLD'] # 50
    reward_threshold = config['REWARD_THRESHOLD'] # 100
    eval_threshold = config['EVAL_THRESHOLD'] # 95
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

    folder_name = "./final_models"    
    env = GazeboEnv()
    odom_subscriber = Odom_subscriber()
    image_subscriber = DepthImage_subscriber()
    laserScan_subscriber = LaserScan_subscriber()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(image_subscriber)
    executor.add_node(laserScan_subscriber)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = odom_subscriber.create_rate(2)
    intervention = 0
    time.sleep(5)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env.seed(seed)
    action_dim = 2
    physical_state_dim = 2 # Polar coordinate
    max_action = 1.0

    # Initialize the agent
    ego = SAC(action_dim, physical_state_dim, policy_type, critic_type, policy_attention_fix,
                critic_attention_fix, pre_buffer, seed, lr_c, lr_a, lr_alpha,
                buffer_size, tau, policy_freq, gamma, alpha, block=transformer_block,
                head=transformer_head, automatic_entropy_tuning=auto_tune)

    # Create evaluation data store
    evaluations = []
    ep_real = 0
    done = False
    target = False
    reward_list = []
    reward_mean_list = []
    pedal_list = []
    steering_list = []
    plt.ion()
    total_timestep = 0
    ep_real = 0
    cntr = 0
    cntr2 = 0
    nb_col = 0
    indice = 0
    for ep in tqdm(range(0, max_episodes), ascii=True):
        episode_reward = 0
        s_list = deque(maxlen=frame_stack)
        s, goal = env.reset()
        for i in range(4):
            s_list.append(s)

        state = np.concatenate((s_list[-4], s_list[-3], s_list[-2], s_list[-1]), axis=-1)

        for timestep in range(max_steps):
            # On termination of episode
            if timestep == 0:
                
                action = ego.choose_action(np.array(state), np.array(goal[:2]))
                action = action.clip(-max_action, max_action)
                a_in = [action[0]/4 + 0.25 , action[1]*1.0]
                last_goal = goal
                s_, _, _, _, _, _ , reward, done, goal, target = env.step(a_in, timestep)
                state = np.concatenate((s_, s_, s_, s_), axis=-1)

                for i in range(4):
                    s_list.append(s_)           

                if done:
                    env.get_logger().warn("Bad Initialization, skip this episode.")
                    break

                continue
            
            if done or timestep == max_steps-1:
                ep_real += 1
    
                done = False

                reward_list.append(episode_reward)
                reward_mean_list.append(np.mean(reward_list[-20:]))

                if reward_mean_list[-1] >= reward_threshold and ep_real > eval_threshold:
                    reward_threshold = reward_mean_list[-1]
                    env.get_logger().warn("Evaluating the Performance.")
                    avg_reward, nb_col = evaluate(env,frame_stack,ego, max_steps,state,timestep, linear_cmd_scale,angular_cmd_scale, max_action)
                    evaluations.append(avg_reward)
                    if avg_reward > save_threshold or nb_col<6:
                        indice +=1
                        ego.save(model_name+desc, directory=folder_name, reward=int(np.floor(avg_reward)), seed=seed)
                        np.save(os.path.join('final_curves', 'reward_seed' + str(seed) + '_' + model_name+desc+str(indice)), reward_mean_list, allow_pickle=True, fix_imports=True)
                        save_threshold = avg_reward
                        

                pedal_list.clear()
                steering_list.clear()
                total_timestep += timestep 
                env.get_logger().info("Reward: %.2f, Overak R: %.2f" % (episode_reward, reward_mean_list[-1]))

                if ep_real % plot_interval == 0:
                    plot_animation_figure(env_name, lr_a, lr_c, ep_real, transformer_block, transformer_head,
                          reward_list, reward_mean_list,model_name,desc)
                break

            action = ego.choose_action(np.array(state), np.array(goal[:2])).clip(-max_action, max_action)
            action_exp = None
            a_in = [action[0]/4 + 0.25 , action[1]*1.0]
            pedal_list.append(round((action[0] + 1)/2,2))
            steering_list.append(round(action[1],2))

            last_goal = goal
            s_, _, _, _, _, _, reward, done, goal, target = env.step(a_in, timestep)

            episode_reward += reward

            next_state = np.concatenate((s_list[-3], s_list[-2], s_list[-1], s_), axis=-1)

            # Save the tuple in replay buffer
            ego.store_transition(state, action, last_goal[:2], goal[:2], reward, next_state, intervention, action_exp, done)
            cntr += 1
            if(cntr==buffer_size):
                env.get_logger().warn('Buffer is already full we lose Data')
            # Train the SAC model
            ego.learn(batch_size)
            # Update the counters
            state = next_state
            s_list.append(s_)
            if(target):
                cntr2=cntr2+1
                env.get_logger().warn(f'Goal reached successfully : {cntr2} !!')
    rclpy.shutdown()
    executor_thread.join()
if __name__ == '__main__':
    main() #call the main function