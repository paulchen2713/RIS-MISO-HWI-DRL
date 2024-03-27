import torch

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env as gym_check_env

import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env as sb3_check_env

import numpy as np
from numpy import linalg
from sympy import sin, cos, pi  # 
from scipy.stats import vonmises_line

import matplotlib.pyplot as plt
# plt.style.use(['ggplot'])
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])

import time
from datetime import date
import os
date_today = date.today()
print(date_today)


def retrieve_name(var):
    import inspect
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def log(*argv):
    for arg in argv:
        print(f"-"*75)
        print(f"{retrieve_name(arg)}")
        print(f"content: ")
        print(arg)
        print(f"type: {type(arg)}")
        if isinstance(arg, np.ndarray) or isinstance(arg, torch.Tensor): 
            print(f"shape: {arg.shape}")
        elif isinstance(arg, list) or isinstance(arg, str) or isinstance(arg, dict):
            print(f"len: {len(arg)}")


class RIS_MISO_Env(gym.Env):
    __slots__ = ('Nk', 'Nr', 'Nt', 'Ns', 'beta_min', 'mu_PDA', 'kappa_PDA', 'loc_mu', 'kappa_PE', 'psi', \
                 'awgn_var', 'Pt', 'bits', 'n_actions', 'action_dim', 'angle_set_deg', 'angle_set_rad', \
                 'splits', 'state_dim', 'H_1', 'H_2', 'H_3', 'F', 'Phi', 'episode_t', '_max_episode_steps', )

    metadata = {"render_modes": ["console"]}

    def __init__(self, 
                 num_users=4,
                 num_UE_antennas=1,
                 num_BS_antennas=4,
                 num_RIS_elements=16,       
                 beta_min=0.2,
                 mu_PDA=0.0,
                 kappa_PDA=1.5,
                 location_mu=0.43*np.pi,
                 concentration_kappa=1.5,   
                 uncertainty_factor=1e-3,
                 AWGN_var=1e-2,
                 Tx_power_dBm=30,
                 bits=3,
                 max_episode_steps=10000,
        ):
        super(RIS_MISO_Env, self).__init__()

        # Downlink RIS-assisted MU-MISO system parameters  
        self.Nk = num_users         # N_k users
        self.Nr = num_UE_antennas   # N_r receive antenna (single-antenna)
        self.Nt = num_BS_antennas   # N_t transmit antenna
        self.Ns = num_RIS_elements  # N_s reflective elements

        assert self.Nt == self.Nk

        # HWI parameters
        #   Phase Depemdent Amplitude
        self.beta_min = beta_min    # \beta_{min}
        self.mu_PDA = mu_PDA        # \mu
        self.kappa_PDA = kappa_PDA  # \kappa
        #   Phase Error 
        self.loc_mu = location_mu            # Von Mises Location factor \mu
        self.kappa_PE = concentration_kappa  # Von Mises Concentration factor \kappa
        #   Channel uncertainty factor
        self.psi = uncertainty_factor        # uncertainty factor \psi

        # 
        self.awgn_var = AWGN_var  # sigma_n^2
        self.Pt = Tx_power_dBm    # total transmit power / beamforming power consumption

        # Ddiscrete actions
        #   action: RIS matrix
        self.bits = bits
        self.n_actions = 2 ** bits
        self.action_dim = self.Ns

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)

        spacing_degree = 360. / self.n_actions
        act = [i for i in range(self.n_actions)]
        deg = [spacing_degree*i - 180. - 15. for i in range(1, self.n_actions + 1)]
        rad = np.radians(deg).tolist()
        self.angle_set_deg = {
            key:val for (key, val) in zip(act, deg)
        }
        self.angle_set_rad = {
            key:val for (key, val) in zip(act, rad)
        }
        self.splits = [-1 + 2 / self.n_actions * i for i in range(0, self.n_actions + 1)]  # [-1, ..., 1]

        # Continuous observation space
        #   state: H_1 + H_2 + H_3 + previous RIS matrix
        self.state_dim = 2 * (self.Ns * self.Nt + self.Nk * self.Ns + self.Nk * self.Nt) + 2*self.Ns + self.action_dim
        # log(self.state_dim)  # 336

        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(self.state_dim,), dtype=np.float32)
        # print(self.observation_space.shape)  # shape: (336,)

        self.Phi = np.eye(self.Ns, dtype=complex)

        self._max_episode_steps = max_episode_steps


    def _compute_PDA(self, angle_rad):
        # beta(\theta) = (1 - \beta_{min}) * ((sin(\theta - \mu) + 1) / 2)^\kappa + \beta_{min}
        beta_PDA = (1 - self.beta_min) * ((sin(angle_rad - self.mu_PDA) + 1) / 2) ** self.kappa_PDA + self.beta_min
        return beta_PDA

    def _Eulers_formula(self, amplitude, angle_rad):
        return amplitude * (cos(angle_rad) + 1j*sin(angle_rad))

    def _compute_Phi_entries(self, angles):
        actual_Phi_entries = np.zeros(self.action_dim, dtype=complex)
        for i, angle in enumerate(angles):
            actual_Phi_entries[i] = self._Eulers_formula(self._compute_PDA(angle), angle)
        return actual_Phi_entries

    def _action2phase(self, indices, unit='radian'):
        actual_phases = np.zeros(self.action_dim)
        if unit == 'radian':
            for i, index in enumerate(indices):
                actual_phases[i] = self.angle_set_rad[index]
            return actual_phases
        elif unit == 'degree':
            for i, index in enumerate(indices):
                actual_phases[i] = self.angle_set_deg[index]
            return actual_phases

    def _linear_interpolation(self, x):
        # we have a value x between x0 and x1 and we want a value y between y0 and y1, y is caluculated as follows:
        #   y = y0 + (x - x0) / (x1 - x0) * (y1 - y0), where [x0, x1] := [-1, 1] and [y0, y1] = [0, 2**bits - 1]
        y = (x + 1) / 2 * (self.n_actions - 1)
        return y

    def _intervals_to_indices(self, act):
        for j in range(0, self.n_actions):
            if self.splits[j] <= act < self.splits[j + 1]: 
                # print(f"{j}: {self.splits[j]}   {act}   {self.splits[j + 1]}")
                return j
        return np.random.randint(low=0, high=self.n_actions, size=1)

    def _rescale(self, actions):
        # method 1: rounding
        #   np.round(self._linear_interpolation(actions)).astype(int)
        
        # method 2: convert intervals to indices
        indices = np.zeros(self.action_dim, dtype=int)  # 2**bits
        for i, act in enumerate(actions):
            indices[i] = self._intervals_to_indices(act)
        return indices

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        info = {}
        
        self.episode_t = 0

        self.H_1 = np.random.normal(0, np.sqrt(0.5), (self.Ns, self.Nt)) + 1j*np.random.normal(0, np.sqrt(0.5), (self.Ns, self.Nt))

        H_2_est = np.random.normal(0, np.sqrt(0.5), (self.Nk, self.Ns)) + 1j*np.random.normal(0, np.sqrt(0.5), (self.Nk, self.Ns))
        delta_H_2_entry = np.random.normal(0, 1, (self.Nk, self.Ns)) + 1j*np.random.normal(0, 1, (self.Nk, self.Ns))
        delta_H_2 = self.psi * (delta_H_2_entry / linalg.norm(delta_H_2_entry, 'fro'))
        self.H_2 = H_2_est + delta_H_2

        H_3_est = np.random.normal(0, np.sqrt(0.5), (self.Nk, self.Nt)) + 1j*np.random.normal(0, np.sqrt(0.5), (self.Nk, self.Nt))
        delta_H_3_entry = np.random.normal(0, 1, (self.Nk, self.Nt)) + 1j*np.random.normal(0, 1, (self.Nk, self.Nt))
        delta_H_3 = self.psi * (delta_H_3_entry / linalg.norm(delta_H_3_entry, 'fro'))
        self.H_3 = H_3_est + delta_H_3

        # Max Ration Transmission (MRT)
        complex_power = np.ones((self.Nt, self.Nk)) + 1j*np.ones((self.Nt, self.Nk))
        normalized_complex_power = complex_power / linalg.norm(complex_power, 'fro')
        self.F = np.sqrt(self.Pt / self.Nk) * normalized_complex_power

        # TODO maybe we could use 'zero-forcing' or 'SVD' to compute the beamforming matrix F later

        # RIS 
        # version 1. np.random.randint(low=0, high=self.n_actions, size=(self.action_dim,))
        # version 2. self._rescale(actions=self.action_space.sample())
        raw_action = -1 + 2*np.random.random_sample(size=(self.action_dim,))
        init_action = self._rescale(actions=raw_action)

        est_rad_phases = self._action2phase(indices=init_action, unit='radian')
        est_Phi_entries = self._compute_Phi_entries(angles=est_rad_phases)
        np.fill_diagonal(self.Phi, est_Phi_entries)
        # log(init_action, self.angle_set_rad, est_rad_phases, est_Phi_entries, np.diag(self.Phi))
        
        H_1_real, H_1_imag = np.real(self.H_1).reshape(-1), np.imag(self.H_1).reshape(-1)  # shape: (1, Ns * Nt) --> (Ns * Nt,)
        H_2_real, H_2_imag = np.real(self.H_2).reshape(-1), np.imag(self.H_2).reshape(-1)  # shape: (1, Nk * Ns) --> (Nk * Ns,)
        H_3_real, H_3_imag = np.real(self.H_3).reshape(-1), np.imag(self.H_3).reshape(-1)  # shape: (1, Nk * Nt) --> (Nk * Nt,)
        Phi_real, Phi_imag = np.real(np.diag(self.Phi)).reshape(-1), np.imag(np.diag(self.Phi)).reshape(-1)  # shape: (1, Ns) --> (Ns,)

        observation = np.concatenate((H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, 
                                      Phi_real, Phi_imag, init_action), axis=0, dtype=np.float32)
        # print(observation.shape)  # (336,)
        # log(observation)
        return observation, info

    def _compute_SumRate_reward(self, Phi):
        reward, opt_reward = 0, 0

        for k in range(self.Nk):
            h_2_k = self.H_2[k, :].reshape(-1, 1)  # shape: (self.Nk, self.Ns)
            f_k = self.F[:, k].reshape(-1, 1)
            # log(k, h_2_k, f_k)

            x = np.abs(h_2_k.T @ Phi @ self.H_1 @ f_k) ** 2
            x = x.item()

            F_removed = np.delete(self.F, k, axis=1)

            interference = np.sum(np.abs(h_2_k.T @ Phi @ self.H_1 @ F_removed) ** 2)
            y = interference + (self.Nk - 1) * self.awgn_var

            rho_k = x / y

            reward += np.log(1 + rho_k) / np.log(2)  # log_2() = log_10() / log(2)
            opt_reward += np.log(1 + x / ((self.Nk - 1) * self.awgn_var)) / np.log(2)

        return reward, opt_reward
    
    def _compute_min_DownLinkRate_reward(self):
        actual_rates, optimal_rates = np.zeros(self.Nk), np.zeros(self.Nk)

        for k in range(self.Nk):
            h_2_k = self.H_2[k, :].reshape(-1, 1)  # shape: (self.Nk, self.Ns)
            f_k = self.F[:, k].reshape(-1, 1)
            # log(k, h_2_k, f_k)

            x = np.abs(h_2_k.T @ self.Phi @ self.H_1 @ f_k) ** 2
            # log(x)      # e.g. [[14.63812291]], type: <class 'numpy.ndarray'>, shape: (1, 1)
            x = x.item()
            # log(x)      # e.g. 14.638122905233896, type: <class 'float'>

            F_removed = np.delete(self.F, k, axis=1)

            interference = np.sum(np.abs(h_2_k.T @ self.Phi @ self.H_1 @ F_removed) ** 2)
            y = interference + (self.Nk - 1) * self.awgn_var

            rho_k = x / y

            actual_rates[k] = np.log(1 + rho_k) / np.log(2)  # log_2() = log_10() / log(2)
            optimal_rates[k] = np.log(1 + x / ((self.Nk - 1) * self.awgn_var)) / np.log(2)

        true_reward, opt_reward = np.min(actual_rates), np.min(optimal_rates)
        reward = true_reward - opt_reward
        
        # log(actual_rates, optimal_rates, true_reward, opt_reward, reward)
        return reward, true_reward, opt_reward

    def _compute_H_tilde(self):
        return self.H_2 @ self.Phi @ self.H_1 + self.H_3
    
    def _compute_MSE_matrix(self):
        H_tilde = self._compute_H_tilde()

        awgn_var_I = np.identity(self.Nk, dtype=complex)
        np.fill_diagonal(awgn_var_I, self.awgn_var)

        equivalent_I = np.identity(self.Nk, dtype=complex) + awgn_var_I
        
        H_tilde_F = H_tilde @ self.F
        H_tilde_F_H = H_tilde_F.conjugate().T

        # log(H_tilde, self.F, awgn_var_I, equivalent_I, H_tilde_F, H_tilde_F_H.conjugate().T, H_tilde_F_H)
        return equivalent_I - H_tilde_F - H_tilde_F_H + (H_tilde_F @ H_tilde_F_H)
    
    def step(self, raw_action):
        info = {'max MSE': -1}
        self.episode_t += 1

        action = self._rescale(actions=raw_action)
        # log(raw_action, action)

        # convert the 'action,' represented as an integer ranging from 0 to 2^bits - 1, into a discrete phase shift
        est_rad_phases = self._action2phase(indices=action, unit='radian')
        phase_errors = vonmises_line(loc=self.loc_mu, kappa=self.kappa_PE).rvs(self.action_dim)
        actual_rad_phases = np.add(est_rad_phases, phase_errors)
        # log(est_rad_phases, phase_errors, actual_rad_phases)

        # construct the Phi matrix using the discrete phase and the phase error
        actual_Phi_entries = self._compute_Phi_entries(angles=actual_rad_phases)
        np.fill_diagonal(self.Phi, actual_Phi_entries)
        # log(self.Phi, actual_Phi_entries)

        # compute the MSE for all users, identify the largest one, and multiply this max value by -1 to obtain the reward
        mse_matrix = self._compute_MSE_matrix()
        user_MSEs = np.diag(mse_matrix).reshape(-1)
        current_max_MSE = np.max(np.real(user_MSEs))
        info['max MSE'] = max(info['max MSE'], current_max_MSE)
        # log(mse_matrix, user_MSEs, np.real(user_MSEs), np.max(np.real(user_MSEs)))

        H_1_real, H_1_imag = np.real(self.H_1).reshape(-1), np.imag(self.H_1).reshape(-1)  # shape: (1, Ns * Nt) --> (Ns * Nt,)
        H_2_real, H_2_imag = np.real(self.H_2).reshape(-1), np.imag(self.H_2).reshape(-1)  # shape: (1, Nk * Ns) --> (Nk * Ns,)
        H_3_real, H_3_imag = np.real(self.H_3).reshape(-1), np.imag(self.H_3).reshape(-1)  # shape: (1, Nk * Nt) --> (Nk * Nt,)
        Phi_real, Phi_imag = np.real(np.diag(self.Phi)).reshape(-1), np.imag(np.diag(self.Phi)).reshape(-1)  # shape: (1, Ns) --> (Ns,)

        observation = np.concatenate((H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, 
                                      Phi_real, Phi_imag, action), axis=0, dtype=np.float32)
        # log(observation)
        
        opt_reward = 230
        reward = opt_reward - current_max_MSE
        # log(reward, opt_reward)

        truncated = (self.episode_t >= self._max_episode_steps)
        done = (opt_reward == reward) or truncated
        return observation, reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        pass


def checking_env(env):
    print(f"checking if the environment follows the Gym and SB3 interfaces\n")
    # NOTE UserWarning: WARN: For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). 
    #                   See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.
    #      UserWarning: WARN: Not able to test alternative render modes due to the environment not having a spec. 
    #                   Try instantialising the environment through gymnasium.make
    gym_check_env(env)

    # NOTE UserWarning: We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) 
    #                   cf. https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
    #      UserWarning: Your action space has dtype int32, we recommend using np.float32 to avoid cast errors.
    sb3_check_env(env)

def test_env(env, episodes=4):
    for episode in range(1, episodes + 1):
        print(f"-"*12)
        print(f" episide: {episode}")
        print(f"-"*12)
        done = False
        obs = env.reset()
        while not done:
            print(f"   step:     {env.episode_t + 1}/{env._max_episode_steps}")
            random_action = env.action_space.sample()
            print(f"   action:   {random_action}")
            obs, reward, done, truncated, info = env.step(random_action)
            print(f"   reward:   {reward}")
            indices = env._rescale(random_action)
            print(f"   indices:  {indices}")
            print(f"   obs:      {np.squeeze(obs)[-16:-1]}")  # fetch the previous action of last step
            print(f"-"*44)

def sample(env, episodes=1):
    for episode in range(1, episodes + 1):
        print(f"{episode}")
        env.reset()
        random_action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(random_action)

def plot_MSE_dist(env, episodes=100):
    num_bins = 100

    mse_samples = np.zeros(episodes)
    for episode in range(1, episodes + 1):
        print(f"{episode}")
        env.reset()
        random_action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(random_action)
        mse_samples[episode - 1] = info['max MSE']
    print(f"\nmin MSE: {np.min(mse_samples)}")
    print(f"max MSE: {np.max(mse_samples)}")
    print(f"average: {np.mean(mse_samples)}")
    print(f"median:  {np.median(mse_samples)}")

    with open(f"system_simulation/mse_dist/mse_samples-{episodes}.txt", "w") as txt_file:
        print(f"\nwriting mse_smaples to text file...")
        for i, mse in enumerate(mse_samples):
            print(f"{mse}", file=txt_file)
        
        print("", file=txt_file)
        print(f"min MSE: {np.min(mse_samples)}", file=txt_file)
        print(f"max MSE: {np.max(mse_samples)}", file=txt_file)
        print(f"average: {np.mean(mse_samples)}", file=txt_file)
        print(f"median:  {np.median(mse_samples)}", file=txt_file)

    plt.figure(figsize=(9, 6))
    plt.axvline(x=np.mean(mse_samples), linewidth=1.5, color='red', linestyle='dashed', label='Average MSE')
    # plt.axvline(x=np.min(mse_samples), linewidth=1.5, linestyle='dashed', color='blue', label='Min MSE')
    # plt.axvline(x=np.max(mse_samples), linewidth=1.5, linestyle='dashed', color='green', label='Max MSE')
    plt.axvline(x=np.median(mse_samples), linewidth=1.5, color='yellow', linestyle='dashdot', label='Median MSE')
    plt.hist(
        mse_samples, 
        bins=num_bins, 
        # histtype='step', 
        label='MSE values'
    )
    plt.legend()
    
    plt.title("MSE distribution")
    plt.xlabel("MSE values")
    plt.ylabel("number of samples")

    plt.savefig(f"system_simulation/mse_dist/mse-{episodes}.png", dpi=200)



if __name__ == "__main__":
    print(f"-"*64)
    env = RIS_MISO_Env(
        num_users=1,
        num_UE_antennas=1,
        num_BS_antennas=1,
        num_RIS_elements=16,       
        beta_min=0.2,
        mu_PDA=0.1,
        kappa_PDA=1.5,
        location_mu=0.43*np.pi,
        concentration_kappa=1.5,   
        uncertainty_factor=0.001,
        AWGN_var=0.01,
        Tx_power_dBm=30,
        bits=3,
        max_episode_steps=1,
    )
    env.reset()
    episodes = 4

    tic = time.perf_counter()

    # checking_env(env=env)

    test_env(env=env)
    
    # plot_MSE_dist(env=env, episodes=100)

    toc = time.perf_counter()
    duration = (toc - tic)
    print(f"duration: {duration:0.4f} sec")
    # env.close()
    
    
    
    

    
