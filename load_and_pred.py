
import numpy as np
from stable_baselines3 import PPO
from torch_env import RIS_MISO_Env

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


def get_random_rewards(Nk, Nt, Ns, env, total_steps, mini_steps):
    random_act_rewards, actuals = [], []
    obs = env.reset()
    for step in range(1, total_steps + 1):  # while not done #
        if step % mini_steps == 0: 
            env.compute_channels(L=L)

        random_action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(random_action)
        random_act_rewards.append(-1*reward)

        actual = env.random_passive_beamforming()
        actuals.append(-1*actual)

    print(f"random action of ({Nk}, {Nt}, {Ns}): ")
    print(f"  (mean, std):  [{np.mean(random_act_rewards)}, {np.std(random_act_rewards)}]")
    print(f"  (mean, std):  [{np.mean(actuals)}, {np.std(actuals)}]")


def get_optimal_rewards(Nk, Nt, Ns, env, total_steps, mini_steps):
    env.reset()
    optimal_act_rewards = [[] for _ in range(Nk)]
    for step in range(1, total_steps + 1):  # while not done #
        if step % mini_steps == 0: 
            env.compute_channels(L=L)

        actual = env.optimal_passive_beamforming()
        for k in range(Nk):
            optimal_act_rewards[k].append(-1*actual[k])

    for k in range(Nk):
        # if k >= 2: continue
        print(f"optimal of ({Nk}, {Nt}, {Ns}) for use {k}:")
        print(f"  (mean, std):  [{np.mean(optimal_act_rewards[k])}, {np.std(optimal_act_rewards[k])}]")


def get_instant_rewards(env, model, total_steps, mini_steps):
    instant_rewards = []
    vec_env = model.get_env()
    obs = vec_env.reset()
    for step in range(1, total_steps + 1):  # while not done #
        if step % mini_steps == 0: 
            env.compute_channels(L=L)
            
        action, _states = model.predict(
            observation=obs, 
            deterministic=True, 
        )
        obs, reward, done, info = vec_env.step(action)
        instant_rewards.append(-1*reward)
        # mse_vector = vec_env.env_method("compute_raw_MSE")
        # instant_rewards.append(np.sum(mse_vector))

    print(f"model inference of {model_name}/{log_index}:")
    print(f"  (mean, std):  [{np.mean(instant_rewards)}, {np.std(instant_rewards)}]")



def aggregate_rand_rewards(env, episodes=1000, max_episode_steps=1, stats_every=100):
    rand_rewards = []
    aggr_rand_rewards = {'epi': [], 'avg': [], 'max': [], 'min': []}

    print(f"-"*16)
    print(f"episide:  1/{episodes}")
    print(f"-"*16)
    for episode in range(1, episodes + 1):
        if episode % 10 == 0:
            print(f"-"*16)
            print(f"episode:  {episode}/{episodes}")
            print(f"-"*16)
        
        episode_reward = 0  # current episode reward
        obs = env.reset()
        done = False
        for step in range(1, max_episode_steps + 1):  ### while not done ###
            if step % 100 == 0: 
                print(f"  step: {step}/{max_episode_steps}")
            
            random_action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(random_action)

            episode_reward += np.squeeze(reward)

        rand_rewards.append(episode_reward / max_episode_steps)
        if episode > 0 and episode % stats_every == 0:
            average_rand_reward = sum(rand_rewards[-stats_every:]) / len(rand_rewards[-stats_every:])  # running average of past 'STATS_EVERY' number of rewards 
            aggr_rand_rewards['epi'].append(episode)
            aggr_rand_rewards['avg'].append(average_rand_reward)
            aggr_rand_rewards['max'].append(max(rand_rewards[-stats_every:]))
            aggr_rand_rewards['min'].append(min(rand_rewards[-stats_every:]))

    return aggr_rand_rewards

def aggregate_agent_rewards(model, episodes=1000, max_episode_steps=1, stats_every=100):
    epi_rewards = []
    aggr_epi_rewards = {'epi': [], 'avg': [], 'max': [], 'min': []}

    vec_env = model.get_env()
    print(f"-"*16)
    print(f"episide:  1/{episodes}")
    print(f"-"*16)
    for episode in range(1, episodes + 1):
        if episode % 10 == 0:
            print(f"-"*16)
            print(f"episode:  {episode}/{episodes}")
            print(f"-"*16)
        
        episode_reward = 0  # current episode reward
        obs = vec_env.reset()
        done = False
        for step in range(1, max_episode_steps + 1):  ### while not done ###
            if step % 100 == 0: 
                print(f"  step: {step}/{max_episode_steps}")
            
            action, _states = model.predict(
                observation=obs, 
                deterministic=True, 
            )
            obs, reward, done, info = vec_env.step(action)

            episode_reward += np.squeeze(reward)

        epi_rewards.append(episode_reward / max_episode_steps)
        if episode > 0 and episode % stats_every == 0:
            average_epi_reward = sum(epi_rewards[-stats_every:]) / len(epi_rewards[-stats_every:])  # running average of past 'STATS_EVERY' number of rewards 
            aggr_epi_rewards['epi'].append(episode)
            aggr_epi_rewards['avg'].append(average_epi_reward)
            aggr_epi_rewards['max'].append(max(epi_rewards[-stats_every:]))
            aggr_epi_rewards['min'].append(min(epi_rewards[-stats_every:]))
            # ':>5d' pad decimal with zeros (left padding, width 5), ':>4.1f' format float 1 decimal places (left padding, width 4)
            # print(f"\n Episode: {episode:>5d}, average reward: {average_reward:>4.1f},", end=" ")
            # print(f"current max: {max(epi_rewards[-stats_every:]):>4.1f},", end=" ")
            # print(f"current min: {min(epi_rewards[-stats_every:]):>4.1f}", end=" ") 
            # print(f"over past {stats_every} number of rewards \n")

    return aggr_epi_rewards

def plot_instant_rewards(instant_rewards, random_act_rewards, max_steps=1000, show=True):
    
    plt.plot([i for i in range(1, len(instant_rewards) + 1)], instant_rewards, label="instant rewards")
    plt.plot([i for i in range(1, len(random_act_rewards) + 1)], random_act_rewards, label="random action rewards")
    plt.legend(loc='lower right')

    plt.title(f"{model_name} Instant rewards with {max_steps} steps")
    plt.xlabel(f"episodes")
    plt.ylabel(f"rewards")
    plt.grid(True)

    if show:
        plt.show()
    else:
        plt.savefig(f"{fig_dir}/{model_name}-{log_index}-Instant-Rewards-{max_steps}-steps.png")
        plt.close()

    plt.title('Confidence Interval')
    plt.xticks([0, 1], ['agent', 'random'])
    plt.ylabel("rewards")
    plot_confidence_interval1(0, instant_rewards)
    plot_confidence_interval1(1, random_act_rewards)

    if show:
        plt.show()
    else:
        plt.savefig(f"{fig_dir}/{model_name}-{log_index}-Confidence-Interval-{max_steps}-steps.png")
        plt.close()

def plot_aggregate_rewards(aggr_epi_rewards, episodes, show=True):
    # Plot the reward figure 
    plt.plot(aggr_epi_rewards['epi'], aggr_epi_rewards['avg'], label="avg rewards")
    plt.plot(aggr_epi_rewards['epi'], aggr_epi_rewards['max'], label="max rewards")
    plt.plot(aggr_epi_rewards['epi'], aggr_epi_rewards['min'], label="min rewards")
    plt.legend(loc='lower right')

    plt.title(f"{model_name} Avg-Max-Min-Rewards with {episodes} episodes")
    plt.xlabel(f"episodes")
    plt.ylabel(f"rewards")
    plt.grid(True)
    
    if show:
        plt.show()
    else:
        plt.savefig(f"{fig_dir}/{model_name}-{log_index}-Avg-Max-Min-Rewards-{episodes}-episodes.png")
        plt.close()
    
    plt.title('Confidence Interval')
    plt.xticks([0, 1, 2], ['average', 'maximum', 'minimum'])
    plt.ylabel("rewards")
    plot_confidence_interval1(0, aggr_epi_rewards['avg'])
    plot_confidence_interval1(1, aggr_epi_rewards['max'])
    plot_confidence_interval1(2, aggr_epi_rewards['min'])

    if show:
        plt.show()
    else:
        plt.savefig(f"{fig_dir}/{model_name}-{log_index}-Confidence-Interval-{episodes}-episodes.png")
        plt.close()


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

    # plt.ylim(-25, 0)

    return mean, confidence_interval



if __name__ == "__main__":
    print(f"-"*68)
    
    # Nk = 2
    # Nt = 16
    # Ns = 16
    TIMESTEPS = 500
    mini_steps = 1
    seed = 3407
    L = 4

    def main(Nk, Nt, Ns, beta=0.9, psi=0.001):
        tic = time.perf_counter()

        global model_name
        model_name = f"PPO-2024-02-09-{Nk}-{Nt}-{Ns}"
        global models_dir
        models_dir = f"models/{model_name}"

        global log_index
        log_index = 1024000
        model_path = f"{models_dir}/{log_index}.zip"
        if not os.path.exists(model_path): raise Exception(f"file {model_path} doesn't exits!")

        global fig_dir
        fig_dir = f"figures/{model_name}/{log_index}/"
        # if not os.path.exists(fig_dir): os.makedirs(fig_dir)

        env1 = RIS_MISO_Env(Nk, 1, Nt, Ns, beta_min=beta, uncertainty_factor=psi, max_episode_steps=TIMESTEPS, seed=seed, L=L)
        env2 = RIS_MISO_Env(Nk, 1, Nt, Ns, beta_min=beta, uncertainty_factor=psi, max_episode_steps=TIMESTEPS, seed=seed, L=L)
        
        # NOTE the loaded env setting has to be exactly the same as the saved one, or we will get 'ValueError: Observation spaces do not match'.
        model = PPO.load(path=model_path, env=env2)
        # print(model.policy)
        
        def test(status=0, num_env1_resets=0, num_env2_resets=0):
            # status = 0  # [0, 1, 2]
            print(f"\n({Nk}, {Nt}, {Ns}, {beta}, {psi}) status {status} with {TIMESTEPS}*{TIMESTEPS//mini_steps}: ")
            if status == 0:
                # num_env1_resets = 0
                if num_env1_resets == 0:
                    get_random_rewards(Nk, Nt, Ns, env1, TIMESTEPS, mini_steps)
                    get_instant_rewards(env2, model, TIMESTEPS, mini_steps)
                else:
                    print(f"number of env1 resets:  {num_env1_resets}")
                    for _ in range(num_env1_resets): env1.reset()
                    get_random_rewards(Nk, Nt, Ns, env1, TIMESTEPS, mini_steps)
                    get_instant_rewards(env2, model, TIMESTEPS, mini_steps)
            elif status == 1:
                # num_env1_resets = 7
                if num_env1_resets == 0:
                    get_optimal_rewards(Nk, Nt, Ns, env1, TIMESTEPS, mini_steps)
                    get_instant_rewards(env2, model, TIMESTEPS, mini_steps)
                else:
                    print(f"number of env1 resets:  {num_env1_resets}")
                    for _ in range(num_env1_resets): env1.reset()
                    get_optimal_rewards(Nk, Nt, Ns, env1, TIMESTEPS, mini_steps)
                    get_instant_rewards(env2, model, TIMESTEPS, mini_steps)
            elif status == 2:
                # num_env2_resets = 0
                if num_env2_resets == 0:
                    get_instant_rewards(env2, model, TIMESTEPS, mini_steps)
                else:
                    print(f"number of env2 resets:  {num_env2_resets}")
                    for _ in range(num_env2_resets): env2.reset()
                    get_instant_rewards(env2, model, TIMESTEPS, mini_steps)
        
        # rng = np.random.default_rng()
        # for i in range(3):
        #     # test(i, rng.integers(0, 200), rng.integers(0, 200))
        #     test(i, 0, 0)
        
        test(2, 0, 0)

        toc = time.perf_counter()
        duration = (toc - tic)
        print(f"duration: {duration:0.4f} sec\n")


    Nk_to_MSE = [3, 6, 8, 10] 
    Nt_to_MSE = [8, 16, 32, 64]
    Ns_to_MSE = [16, 36, 64, 100]
    beta_mins = [i/10 for i in range(0, 11, 2)]
    psi = [0] + [1/10**i for i in range(4, 0, -1)]
    # psi = [0.001*i for i in range(0, 11, 2)]

    for i in psi:
        main(2, 16, 16, 0.9, i)

    # main(2, 16, 16, 0.9, 0.1)
    print(psi)
