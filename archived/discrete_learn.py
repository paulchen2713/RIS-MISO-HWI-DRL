import numpy as np
from stable_baselines3 import DQN, PPO, A2C

from discrete_torch_env import RIS_MISO_Env

from datetime import date
import os

from utils import log



if __name__ == "__main__":
    print(f"-"*64)

    date_today = f"2023-11-09"  # date.today()
    print(date_today)

    # NOTE On RTX 2060 + i7-8700 with the setting of (Nk, Nt, Ns, obs.shape) = (4, 36, 36, 3276)
    # UserWarning: This system does not have apparently enough memory to store the complete replay buffer 26.22GB > 22.35GB

    # NOTE On RTX 3060Ti + i7-12700 with the setting of (Nk, Nt, Ns, obs.shape) = (56, 56, 4, 7180)
    # UserWarning: This system does not have apparently enough memory to store the complete replay buffer 59.39GB > 59.25GB

    Nk = 4
    Nt = 16
    Ns = 4
    episodes = 100
    TIMESTEPS = 10000

    model_name = "DQN"
    logs_dir = f"logs/{model_name}-{date_today}/"
    models_dir = f"models/{model_name}-{date_today}-{Nk}-{Nt}-{Ns}/"

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    env = RIS_MISO_Env(
        num_users=Nk,
        num_UE_antennas=1,
        num_BS_antennas=Nt,
        num_RIS_elements=Ns,
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
    )
    env.reset()

    # model = DQN(
    #     policy='MlpPolicy', 
    #     env=env, 
    #     learning_rate=0.0001, 
    #     buffer_size=1000000,
    #     learning_starts=50000,         # how many steps of the model to collect transitions for before learning starts
    #     batch_size=64,
    #     tau=1,                         # the soft update coefficient (between 0 and 1) default 1 for hard update
    #     gamma=0.99,                    # the discount factor
    #     train_freq=(1, "step"),        # pdate the model every (1, "step") or (1, "episode")
    #     gradient_steps=1,              # how many gradient steps to do after each rollout
    #     target_update_interval=10000,  # update the target network every episode (environment steps)
    #     exploration_fraction=0.1,      # fraction of entire training period over which the exploration rate is reduced
    #     exploration_initial_eps=1,     # initial value of random action probability
    #     exploration_final_eps=0.05,    # final value of random action probability
    #     verbose=1,                     # 0 for no output, 1 for info messages, 2 for debug messages
    #     tensorboard_log=logs_dir,
    # )

    model = DQN(policy='MlpPolicy', env=env, verbose=1, tensorboard_log=logs_dir)

    for i in range(1, episodes + 1):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{model_name}")
        model.save(f"{models_dir}/{TIMESTEPS*i}")


    print(date_today)

