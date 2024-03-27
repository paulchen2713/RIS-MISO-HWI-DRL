import torch
import numpy as np
from stable_baselines3 import PPO

from torch_env import RIS_MISO_Env

from datetime import date
import os

from utils import log



if __name__ == "__main__":
    print(f"-"*64)

    date_today = f"2024-02-19"  # date.today()
    print(date_today)

    Nk = 2
    Nt = 16
    Ns = 16
    episodes = 100
    TIMESTEPS = 20480
    seed = 33
    L = 4
    layers = [Ns*2**i for i in range(5, 0, -1)]

    model_name = "PPO"
    logs_dir = f"logs/{model_name}-{date_today}/"
    models_dir = f"models/{model_name}-{date_today}-{Nk}-{Nt}-{Ns}/"

    if not os.path.exists(logs_dir): os.makedirs(logs_dir)

    if not os.path.exists(models_dir): os.makedirs(models_dir)

    env = RIS_MISO_Env(
        num_users=Nk,
        num_BS_antennas=Nt,
        num_RIS_elements=Ns,
        beta_min=0.9,  # 0.9,
        mu_PDA=0.21,   # 0.21,
        kappa_PDA=3.4,
        location_mu=0.6*np.pi,
        concentration_kappa=1.2,   
        uncertainty_factor=0.1,  # 0.0001, -10 dBm
        AWGN_var=0.000001,  # -30 dBm
        Tx_power=1,         #  30 dBm
        bits=1,
        max_episode_steps=TIMESTEPS,
        seed=seed,
        L=L,
    )
    
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,  # torch.nn.ReLU, torch.nn.Tanh
        net_arch=dict(
            pi=layers,
            vf=layers,
        )
    )

    for _ in range(0): env.reset()
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, ent_coef=0.01, verbose=1, tensorboard_log=logs_dir)

    for i in range(1, episodes + 1):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{model_name}")
        model.save(f"{models_dir}/{TIMESTEPS*i}")

    print(model.policy)
    print(date_today)

