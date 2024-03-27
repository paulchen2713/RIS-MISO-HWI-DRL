import torch

import gym
# from gym.utils.env_checker import check_env as gym_check_env
# from gym import spaces

# import stable_baselines3 as sb3
# from stable_baselines3.common.env_checker import check_env as sb3_check_env

import numpy as np

import inspect


def retrieve_name(var):
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


class RIS_MISO(gym.Env):
    def __init__(self,
                 num_antennas=4,
                 num_RIS_elements=16,
                 num_users=4,
                 channel_est_error=False,
                 AWGN_var=1e-2,
                 channel_noise_var=1e-2):
        super(RIS_MISO, self).__init__()

        self.M = num_antennas
        self.L = num_RIS_elements
        self.K = num_users

        self.channel_est_error = channel_est_error

        assert self.M == self.K

        self.awgn_var = AWGN_var
        self.channel_noise_var = channel_noise_var

        power_size = 2 * self.K

        channel_size = 2 * (self.L * self.M + self.L * self.K)

        self.action_dim = 2 * self.M * self.K + 2 * self.L
        self.state_dim = power_size + channel_size + self.action_dim

        self.H_1 = None
        self.H_2 = None
        self.G = np.eye(self.M, dtype=complex)
        self.Phi = np.eye(self.L, dtype=complex)

        self.state = None
        self.done = None

        self.episode_t = None

        self.dummy_action = None

    def _compute_H_2_tilde(self):
        return self.H_2.T @ self.Phi @ self.H_1 @ self.G

    def reset(self):
        self.episode_t = 0

        self.H_1 = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.M))
        self.H_2 = np.random.normal(0, np.sqrt(0.5), (self.L, self.K)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.K))

        init_action_G = np.hstack((np.real(self.G.reshape(1, -1)), np.imag(self.G.reshape(1, -1))))
        init_action_Phi = np.hstack((np.real(np.diag(self.Phi)).reshape(1, -1), np.imag(np.diag(self.Phi)).reshape(1, -1)))

        init_action = np.hstack((init_action_G, init_action_Phi))

        self.dummy_action = init_action

        Phi_real = init_action[:, -2 * self.L:-self.L]
        Phi_imag = init_action[:, -self.L:]

        self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag)

        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        H_2_tilde = self._compute_H_2_tilde()
        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2

        H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)

        self.state = np.hstack((init_action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        return self.state

    def get_dummy_action(self):
        return self.dummy_action

    def _compute_reward(self, Phi):
        reward = 0
        opt_reward = 0

        for k in range(self.K):
            h_2_k = self.H_2[:, k].reshape(-1, 1)  # (self.L, self.K)
            g_k = self.G[:, k].reshape(-1, 1)      # (self.M, self.K)
            log(k, h_2_k, g_k)

            x = np.abs(h_2_k.T @ Phi @ self.H_1 @ g_k) ** 2

            x = x.item()

            G_removed = np.delete(self.G, k, axis=1)

            interference = np.sum(np.abs(h_2_k.T @ Phi @ self.H_1 @ G_removed) ** 2)
            y = interference + (self.K - 1) * self.awgn_var

            rho_k = x / y

            reward += np.log(1 + rho_k) / np.log(2)
            opt_reward += np.log(1 + x / ((self.K - 1) * self.awgn_var)) / np.log(2)

        return reward, opt_reward

    def step(self, action):
        self.episode_t += 1

        action = action.reshape(1, -1)

        G_real = action[:, :self.M ** 2]
        G_imag = action[:, self.M ** 2:2 * self.M ** 2]

        Phi_real = action[:, -2 * self.L:-self.L]
        Phi_imag = action[:, -self.L:]

        self.G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)

        self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag)

        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        H_2_tilde = self._compute_H_2_tilde()

        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2

        H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)

        self.state = np.hstack((action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        reward, opt_reward = self._compute_reward(self.Phi)

        done = (opt_reward == reward)

        return self.state, reward, done, {}

    def close(self):
        pass



if __name__ == "__main__":
    print(f"checking if the environment follows the Gym and SB3 interfaces\n")

    env = RIS_MISO()

    # gym_check_env(env)
    # sb3_check_env(env)

    env.reset()

    dummy_action = env.get_dummy_action()
    env.step(dummy_action)


