import torch
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from discrete_torch_env import RIS_MISO_Env

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])
# plt.style.use(['ggplot'])

from datetime import date
import time
import os

import warnings
warnings.filterwarnings('ignore')
# NOTE UserWarning: This system does not have apparently enough memory to store the complete replay buffer 25.46GB > 24.60GB

from utils import log


def get_random_rewards(env, episodes=1, max_episode_steps=1000):
    random_act_rewards = []
    for episode in range(1, episodes + 1):
        if episode % 10 == 0: print(f"episode: {episode} / {episodes}")

        done = False
        obs = env.reset()
        for step in range(1, max_episode_steps + 1):  # while not done #
            if step % 100 == 0: print(f"step: {step} / {max_episode_steps}")
            random_action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(random_action)
            random_act_rewards.append(reward)
            
    print(f"-"*32)
    print(f"random action:")
    print(f"   mean:   {np.mean(random_act_rewards)}")
    print(f"   std:    {np.std(random_act_rewards)}")
    print(f"   max:    {np.max(random_act_rewards)}")
    print(f"   min:    {np.min(random_act_rewards)}")
    print(f"   shape:  {obs.shape}")
    print(f"-"*32)

    return random_act_rewards

def get_instant_rewards(model, episodes=1, max_episode_steps=1000):
    instant_rewards = []
    vec_env = model.get_env()
    for episode in range(1, episodes + 1):
        if episode % 10 == 0: print(f"episode: {episode} / {episodes}")
        
        obs = vec_env.reset()
        for step in range(1, max_episode_steps + 1):  # while not done #
            if step % 100 == 0: print(f"step: {step} / {max_episode_steps}")

            action, _states = model.predict(
                observation=obs, 
                deterministic=True, 
            )
            obs, reward, done, info = vec_env.step(action)
            instant_rewards.append(reward)

    print(f"-"*32)
    print(f"model inference of {model_name}:")
    print(f"   mean:   {np.mean(instant_rewards)}")
    print(f"   std:    {np.std(instant_rewards)}")
    print(f"   max:    {np.max(instant_rewards)}")
    print(f"   min:    {np.min(instant_rewards)}")
    print(f"   shape:  {obs.shape}")
    print(f"-"*32)

    return instant_rewards

def aggregate_rewards(model, episodes=1000, max_episode_steps=1, stats_every=100):
    all_actions = []

    epi_rewards = []
    aggr_epi_rewards = {'epi': [], 'avg': [], 'max': [], 'min': []}

    vec_env = model.get_env()
    print(f"-"*68)
    for episode in range(1, episodes + 1):
        if episode % 100 == 0: print(f"episode: {episode} / {episodes}")
        episode_reward = 0  # current episode reward

        obs = vec_env.reset()
        done = False
        for step in range(1, max_episode_steps + 1):  ### while not done ###
            if step % 100 == 0: print(f"step: {step} / {max_episode_steps}")
            action, _states = model.predict(
                observation=obs, 
                deterministic=True, 
            )
            obs, reward, done, info = vec_env.step(action)

            prev = vec_env.get_attr("prev_actions")
            all_actions.append(list(prev[0]))

            episode_reward += np.squeeze(reward)

        epi_rewards.append(episode_reward)
        if episode > 0 and episode % stats_every == 0:
            average_reward = sum(epi_rewards[-stats_every:]) / len(epi_rewards[-stats_every:])  # running average of past 'STATS_EVERY' number of rewards 
            aggr_epi_rewards['epi'].append(episode)
            aggr_epi_rewards['avg'].append(average_reward)
            aggr_epi_rewards['max'].append(max(epi_rewards[-stats_every:]))
            aggr_epi_rewards['min'].append(min(epi_rewards[-stats_every:]))
            # ':>5d' pad decimal with zeros (left padding, width 5), ':>4.1f' format float 1 decimal places (left padding, width 4)
            print(f"\n Episode: {episode:>5d}, average reward: {average_reward:>4.1f},", end=" ")
            print(f"current max: {max(epi_rewards[-stats_every:]):>4.1f},", end=" ")
            print(f"current min: {min(epi_rewards[-stats_every:]):>4.1f}", end=" ") 
            print(f"over past {stats_every} number of rewards \n")

    return all_actions, aggr_epi_rewards

def store_actions_to_txt(all_actions):
    print(f"Writing all actions of {model_name} to .txt file")
    with open(f"{fig_dir}/{log_index}-all_actions.txt", "w") as txt_file:
        for action in all_actions:
            print(f"{action}", file=txt_file)
    print(f"Finished!")

def plot_confidence_interval1(x, values, z=1.96, color='#2187bb', horizontal_line_width=0.25):
    mean = np.mean(values)
    std = np.std(values)
    confidence_interval = z * std / np.sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color='#f44336')

    return mean, confidence_interval

def plot_instant_rewards(instant_rewards, random_act_rewards, max_steps=1000):
    
    plt.plot([i for i in range(1, len(instant_rewards) + 1)], instant_rewards, label="instant rewards")
    plt.plot([i for i in range(1, len(random_act_rewards) + 1)], random_act_rewards, label="random action rewards")
    plt.legend(loc='best')

    plt.title(f"{model_name} Instant rewards with {max_steps} steps")
    plt.xlabel(f"episodes")
    plt.ylabel(f"rewards")
    plt.grid(True)

    # plt.savefig(f"{fig_dir}/{log_index}-Instant-Rewards-{max_steps}-steps.png")
    # plt.close()
    plt.show()

    plt.title('Confidence Interval')
    plt.xticks([0, 1], ['agent', 'random'])
    plt.ylabel("sum-rate")
    plot_confidence_interval1(0, instant_rewards)
    plot_confidence_interval1(1, random_act_rewards)
    # plt.savefig(f"{fig_dir}/{log_index}-Confidence-Interval-{max_steps}-steps.png")
    # plt.close()
    plt.show()


def plot_aggr_epi_rewards(aggr_epi_rewards, episodes):
    # Plot the reward figure 
    plt.plot(aggr_epi_rewards['epi'], aggr_epi_rewards['avg'], label="avg rewards")
    plt.plot(aggr_epi_rewards['epi'], aggr_epi_rewards['max'], label="max rewards")
    plt.plot(aggr_epi_rewards['epi'], aggr_epi_rewards['min'], label="min rewards")
    plt.legend(loc='best')

    plt.title(f"{model_name} Avg-Max-Min-Rewards with {episodes} episodes")
    plt.xlabel(f"episodes")
    plt.ylabel(f"rewards")
    plt.grid(True)
    
    # plt.savefig(f"{fig_dir}/{log_index}-Avg-Max-Min-Rewards-{episodes}-episodes.png")
    # plt.close()
    plt.show()

def test_inference(model, episodes=2, max_episode_steps=3):
    vec_env = model.get_env()
    for episode in range(1, episodes + 1):
        print(f"-"*12)
        print(f" episide: {episode}")
        print(f"-"*12)
        obs = vec_env.reset()
        for step in range(1, max_episode_steps + 1):  # while not done #
            print(f"   step:   {step}/{max_episode_steps}")
            action, _states = model.predict(
                observation=obs, 
                deterministic=True, 
            )
            print(f"   action:  {np.squeeze(action)}")

            obs, reward, done, info = vec_env.step(action)

            print(f"   reward:  {np.squeeze(reward)}")
            Ns = np.squeeze(vec_env.get_attr("Ns"))
            print(f"   obs:     {np.squeeze(obs)[-Ns:]}")
            print(f"-"*44)



if __name__ == "__main__":
    print(f"-"*68)
    tic = time.perf_counter()

    model_name = f"DQN-2023-11-06-2-4-4"
    models_dir = f"models/{model_name}"
    log_index = 200_000
    model_path = f"{models_dir}/{log_index}.zip"
    fig_dir = f"figures/{model_name}/"

    if not os.path.exists(models_dir):
        print(f"\n file {model_path} doesn't exits!\n ")
    else:
        print(f"\n ...loading \'{model_path}\' model... \n")

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)


    TIMESTEPS = 10000
    env = RIS_MISO_Env(
        num_users=2,
        num_UE_antennas=1,
        num_BS_antennas=4,
        num_RIS_elements=4,
        beta_min=0.2,
        mu_PDA=0.1,
        kappa_PDA=1.5,
        location_mu=0.43*np.pi,
        concentration_kappa=1.5,   
        uncertainty_factor=0.001,
        AWGN_var=0.0001,
        Tx_power_dBm=5,
        bits=2,
        max_episode_steps=TIMESTEPS,
        seed=33,
    )
    env.reset()
    
    # NOTE the loaded env setting has to be exactly the same as the saved one, 
    # or we will get 'ValueError: Observation spaces do not match'.
    model = DQN.load(path=model_path, env=env)

    episodes = 1000
    stats_every = 100
    # all_actions, aggr_epi_rewards = aggregate_rewards(model, episodes=episodes, max_episode_steps=1, stats_every=stats_every)
    # plot_aggr_epi_rewards(aggr_epi_rewards, episodes=episodes)
    # store_actions_to_txt(all_actions)

    max_steps = 1000
    random_act_rewards = get_random_rewards(env, episodes=1, max_episode_steps=max_steps)
    instant_rewards = get_instant_rewards(model, episodes=1, max_episode_steps=max_steps)
    plot_instant_rewards(instant_rewards, random_act_rewards, max_steps)
    
    # test_inference(model)

    toc = time.perf_counter()
    duration = (toc - tic)
    print(f"duration: {duration:0.4f} sec")
    print(f"model:    {model_name}/{log_index}.zip")
