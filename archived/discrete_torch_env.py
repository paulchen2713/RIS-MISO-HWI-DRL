import torch

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env as gym_check_env

from stable_baselines3.common.env_checker import check_env as sb3_check_env

import numpy as np

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])
# plt.style.use(['ggplot'])

import time
from datetime import date
import os
from collections import deque

date_today = date.today()
# print(f"date_today: {date_today}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"device: {device}")

torch.backends.cudnn.benchmark = True  # Set to True should increase the speed of the code if the input sizes don't change

from utils import log


class RIS_MISO_Env(gym.Env):
    __slots__ = ('Nk', 'Nr', 'Nt', 'Ns', 'beta_min', 'mu_PDA', 'kappa_PDA', 'loc_mu', 'kappa_PE', 'psi', \
                 'awgn_var', 'Pt', 'bits', 'n_actions', 'action_dim', 'angle_set_deg', 'angle_set_rad', \
                 'state_dim', 'H_1', 'H_2', 'H_3', 'F', 'Phi', 'episode_t', '_max_episode_steps', 'prev_actions', \
                 'seed')

    metadata = {"render_modes": ["console"]}

    def __init__(self, 
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
            max_episode_steps=10000,
            seed=33,
        ):
        super(RIS_MISO_Env, self).__init__()

        assert num_BS_antennas >= num_users

        # Downlink RIS-assisted MU-MISO system parameters  
        self.Nk = num_users         # N_k users
        self.Nr = num_UE_antennas   # N_r receive antenna (single-antenna)
        self.Nt = num_BS_antennas   # N_t transmit antenna
        self.Ns = num_RIS_elements  # N_s reflective elements

        # HWI parameters
        #   Phase Depemdent Amplitude
        self.beta_min = torch.tensor(beta_min, dtype=torch.float32, device=device)
        self.mu_PDA = torch.tensor(mu_PDA, dtype=torch.float32, device=device)
        self.kappa_PDA = torch.tensor(kappa_PDA, dtype=torch.float32, device=device)
        #   Phase Error 
        self.loc_mu = torch.tensor(location_mu, dtype=torch.float32, device=device)
        self.kappa_PE = torch.tensor(concentration_kappa, dtype=torch.float32, device=device)
        #   Channel uncertainty factor
        self.psi = torch.tensor(uncertainty_factor, dtype=torch.float32, device=device)  # uncertainty factor

        # SNR
        self.awgn_var = torch.tensor(AWGN_var, dtype=torch.float32, device=device)  # sigma_n^2
        self.Pt = torch.tensor(Tx_power_dBm, dtype=torch.float32, device=device)    # beamforming power consumption

        # Discrete actions
        #   action: RIS matrix
        self.bits = bits
        self.n_actions = 2**bits
        self.action_dim = self.Ns

        self.action_space = spaces.Discrete(start=0, n=self.n_actions)  # {start, ..., start + n - 1}
        self.prev_actions = deque(maxlen=self.action_dim)

        spacing_degree = 360. / self.n_actions
        act = [i for i in range(self.n_actions)]
        deg = [spacing_degree*i - 180. for i in range(1, self.n_actions + 1)]
        rad = np.radians(deg).tolist()
        self.angle_set_deg = {
            key:val for (key, val) in zip(act, deg)
        }
        self.angle_set_rad = {
            key:val for (key, val) in zip(act, rad)
        }

        # Continuous observation space
        #   state: H_1 + H_2 + H_3 + RIS matrix
        self.state_dim = 2 * (self.Ns * self.Nt + self.Nk * self.Ns + self.Nk * self.Nt) + 2*self.Ns + self.action_dim
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.state_dim,), dtype=np.float32)
        
        self.Phi = torch.eye(self.Ns, dtype=torch.complex64, device=device)

        self._max_episode_steps = max_episode_steps
        self.seed_everything(seed=seed)

    def seed_everything(self, seed=None):
        print(f"seed_everything() is being called with random seed set to {seed}")
        # import random
        # random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True  # this will make the code runs extremely slow
        # torch.backends.cudnn.benchmark = False     # I don't know if I should disable this when seeding everything

    def compute_PDA(self, angle_rad: torch.FloatTensor) -> torch.Tensor:
        beta_PDA = (1 - self.beta_min) * ((torch.sin(angle_rad - self.mu_PDA) + 1) / 2)**self.kappa_PDA + self.beta_min
        return beta_PDA

    def Eulers_formula(self, amplitude: torch.FloatTensor, angle_rad: torch.FloatTensor) -> torch.Tensor:
        return amplitude * (torch.cos(angle_rad).to(device) + 1j*torch.sin(angle_rad).to(device))

    def compute_Phi_entries(self, angles: torch.FloatTensor) -> torch.Tensor:
        actual_Phi_entries = torch.zeros(self.action_dim, dtype=torch.complex64, device=device)
        for i, angle in enumerate(angles):
            actual_Phi_entries[i] = self.Eulers_formula(self.compute_PDA(angle), angle)
        return actual_Phi_entries

    def action2phase(self, indices, unit='radian') -> torch.Tensor:
        actual_phases = torch.zeros(self.action_dim, device=device)
        if unit == 'radian':
            for i, index in enumerate(indices):
                actual_phases[i] = torch.tensor(self.angle_set_rad[index], dtype=torch.float32, device=device)
            return actual_phases
        elif unit == 'degree':
            for i, index in enumerate(indices):
                actual_phases[i] = torch.tensor(self.angle_set_deg[index], dtype=torch.float32, device=device)
            return actual_phases

    def torch_array_response(angle1: torch.FloatTensor, angle2: torch.FloatTensor, 
                             num: torch.ByteTensor, antenna_array: str) -> torch.Tensor:
        """
        Generate ULA and UPA steering vectors
        """
        # NOTE The UPA mode only works with perfect square numbers; otherwise, 
        #      it is rounded down to the nearest perfect square number.

        assert num > 0
        PI = torch.tensor(np.pi, dtype=torch.float32, device=device)

        array_response = torch.zeros((num, 1), dtype=torch.complex64, device=device)

        if antenna_array == 'UPA':
            num_sqrt = int(torch.sqrt(num))
            assert num_sqrt * num_sqrt == int(num)

            for m in range(num_sqrt):
                for n in range(num_sqrt):
                    array_response[m * num_sqrt + n] \
                        = torch.exp(1j*PI*(m*torch.sin(angle1).to(device)*torch.cos(angle2).to(device) 
                                           + n*torch.cos(angle2).to(device))).to(device)
        elif antenna_array == 'ULA':
            for n in range(num):
                array_response[n] = torch.exp(1j*PI*(n*torch.sin(angle1).to(device))).to(device)
    
        array_response = array_response / torch.sqrt(num).to(device)
        return array_response

    def torch_ULA_response(self, angle: torch.FloatTensor, num_antennas: torch.ByteTensor) -> torch.Tensor:
        """
        Return the ULA steering vector

        Keyword arguments:
        angle:         the angles of arrival(AoA) or angle of departure (AoD) in radian
        num_antennas:  the number of Tx or Rx antennas
        """
        assert num_antennas > 0
        PI = torch.tensor(np.pi, dtype=torch.float32, device=device)
        
        array_response = torch.zeros((num_antennas, 1), dtype=torch.complex64, device=device)
        
        for n in range(0, num_antennas):
            array_response[n] = torch.exp(1j*PI*(n*torch.sin(angle).to(device))).to(device)
        
        array_response = array_response / torch.sqrt(num_antennas).to(device)
        return array_response

    def torch_UPA_response(self, azimuth: torch.FloatTensor, elevation: torch.FloatTensor, 
                           M_y: torch.ByteTensor, M_z: torch.ByteTensor) -> torch.Tensor:
        """
        Return the UPA steering vector

        Keyword arguments:
        azimuth:    the azimuth AoA or AoD in radian
        elevation:  the elevation AoA or AoD in radian
        M_y:        the number horizontal antennas of Tx or Rx 
        M_z:        the number vertical antennas of Tx or Rx 
        """
        assert M_y > 0 and M_z > 0
        PI = torch.tensor(np.pi, dtype=torch.float32, device=device)
        
        array_response = torch.zeros((M_y * M_z, 1), dtype=torch.complex64, device=device)
        
        for m in range(0, M_y):
            for n in range(0, M_z):
                array_response[m * int(M_z) + n] = \
                    torch.exp(1j*PI*(m*torch.sin(azimuth).to(device)*torch.cos(elevation).to(device) 
                                     + n*torch.cos(elevation).to(device))).to(device)
        array_response = array_response / torch.sqrt(M_y * M_z).to(device)
        return array_response

    def torch_USPA_response(self, azimuth: torch.FloatTensor, elevation: torch.FloatTensor, 
                            num_antennas: torch.ByteTensor) -> torch.Tensor:
        """
        Return the Uniform Square Planar Array (USPA) steering vector

        Keyword arguments:
        azimuth:       the azimuth AoA or AoD in radian
        elevation:     the elevation AoA or AoD in radian
        num_antennas:  the total number of the Tx or Rx antennas
        """
        assert num_antennas > 0
        PI = torch.tensor(np.pi, dtype=torch.float32, device=device)
        
        array_response = torch.zeros((num_antennas, 1), dtype=torch.complex64, device=device)
        
        num_sqrt = int(torch.sqrt(num_antennas))
        assert num_sqrt * num_sqrt == int(num_antennas)

        for m in range(0, num_sqrt):
            for n in range(0, num_sqrt):
                array_response[m * num_sqrt + n] = \
                    torch.exp(1j*PI*(m*torch.sin(azimuth).to(device)*torch.cos(elevation).to(device) 
                                    + n*torch.cos(elevation).to(device))).to(device)
        array_response = array_response / torch.sqrt(num_antennas).to(device)
        return array_response

    def get_ULA_sample(self, num: int) -> torch.Tensor:
        return self.torch_ULA_response(
            torch.deg2rad(torch.randint(low=0, high=360, size=(1,), dtype=torch.int16, device=device)).to(device),
            torch.tensor(num, dtype=torch.uint8, device=device), 
        )

    def get_USPA_sample(self, num: int) -> torch.Tensor:
        return self.torch_USPA_response(
            torch.deg2rad(torch.randint(low=0, high=360, size=(1,), dtype=torch.int16, device=device)).to(device),
            torch.deg2rad(torch.randint(low=0, high=360, size=(1,), dtype=torch.int16, device=device)).to(device),
            torch.tensor(num, dtype=torch.uint8, device=device), 
    )

    def torch_CN(self, mean=0.0, std=1.0, size=None) -> torch.Tensor:
        return torch.normal(mean, std, size).to(device) + 1j*torch.normal(mean, std, size).to(device)

    def compute_H_tilde(self):
        return self.H_2 @ self.Phi @ self.H_1 + self.H_3

    def reset(self, seed=33, options=None):
        super().reset(seed=seed) 
        self.episode_t = 0

        a_BS_1 = self.get_ULA_sample(self.Nt)
        a_RIS_1 = self.get_USPA_sample(self.Ns)
        H_1_channel_gain = self.torch_CN(0, 1, (1,))
        self.H_1 = H_1_channel_gain * (a_RIS_1 @ a_BS_1.conj().T)  # NOTE H_1 shape: (Ns, Nt)
        del a_BS_1, a_RIS_1, H_1_channel_gain

        a_RIS_2 = torch.zeros((self.Nk, self.Ns), dtype=torch.complex64, device=device)
        for i in range(self.Nk):
            a_RIS_2[i] = self.get_USPA_sample(self.Ns).conj().T
        H_2_channel_gain = self.torch_CN(0, 1, (1,))
        H_2_est = H_2_channel_gain * a_RIS_2                       # NOTE H_2 shape: (Nk, Ns)
        
        delta_H_2_entry = self.torch_CN(0, 1, (self.Nk, self.Ns))
        delta_H_2 = self.psi * (delta_H_2_entry / torch.norm(delta_H_2_entry, 'fro')).to(device)
        self.H_2 = H_2_est + delta_H_2 
        del a_RIS_2, H_2_channel_gain, H_2_est, delta_H_2_entry, delta_H_2  

        a_BS_3 = torch.zeros((self.Nk, self.Nt), dtype=torch.complex64, device=device)
        for i in range(self.Nk):
            a_BS_3[i] = self.get_ULA_sample(self.Nt).conj().T
        H_3_channel_gain = self.torch_CN(0, 1, (1,))
        H_3_est = H_3_channel_gain * a_BS_3                        # NOTE H_3 shape: (Nk, Nt)

        delta_H_3_entry = self.torch_CN(0, 1, (self.Nk, self.Nt))
        delta_H_3 = self.psi * (delta_H_3_entry / torch.norm(delta_H_3_entry, 'fro')).to(device)
        self.H_3 = H_3_est + delta_H_3
        del a_BS_3, H_3_channel_gain, H_3_est, delta_H_3_entry, delta_H_3

        # Max Ratio Transmission (MRT)
        # complex_power = torch.ones((self.Nt, self.Nk), device=device) + 1j*torch.ones((self.Nt, self.Nk), device=device)
        # self.F = torch.sqrt(self.Pt / self.Nk).to(device) * (complex_power / torch.norm(complex_power, 'fro').to(device))

        H_tilde = self.compute_H_tilde()                           # NOTE F shape: (Nk, Nt)
        self.F = torch.sqrt(self.Pt / self.Nk).to(device) * (H_tilde.conj().T / torch.norm(H_tilde, 'fro').to(device))
        del H_tilde

        # TODO maybe we could use 'zero-forcing' or 'SVD' to compute the beamforming matrix F later

        # RIS 
        # method 1. random action initialization
        #   torch.randint(low=0, high=self.n_actions, size=(self.action_dim,), dtype=torch.uint8, device=device)

        # method 2. same initial action 
        init_action = torch.ones(size=(self.action_dim,), dtype=torch.uint8, device=device)
        for act in init_action:
            self.prev_actions.appendleft(act.item())

        est_rad_phases  = self.action2phase(indices=self.prev_actions, unit='radian')
        est_Phi_entries = self.compute_Phi_entries(angles=est_rad_phases)
        self.Phi = torch.diagonal_scatter(self.Phi, est_Phi_entries)
        del est_rad_phases, est_Phi_entries
        
        H_1_real, H_1_imag = torch.real(self.H_1).reshape(-1).to(device), torch.imag(self.H_1).reshape(-1).to(device)
        H_2_real, H_2_imag = torch.real(self.H_2).reshape(-1).to(device), torch.imag(self.H_2).reshape(-1).to(device)
        H_3_real, H_3_imag = torch.real(self.H_3).reshape(-1).to(device), torch.imag(self.H_3).reshape(-1).to(device)
        Phi_real, Phi_imag = torch.real(torch.diag(self.Phi)).to(device), torch.imag(torch.diag(self.Phi)).to(device)  

        observation = torch.cat((H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, 
                                 Phi_real, Phi_imag, init_action), dim=0)
        del H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, Phi_real, Phi_imag, init_action
        
        return np.array(observation.cpu(), dtype=np.float32), {}

    def compute_interference(self) -> torch.FloatTensor:
        interference = 0

        for k in range(self.Nk):
            h_2_k = self.H_2[k, :].reshape(1, -1)  # H_2: (Nk, Ns) --> h_2_k: (1, Ns)
            h_3_k = self.H_3[k, :].reshape(1, -1)  # H_3: (Nk, Nt) --> h_3_k: (1, Nt)
            f_k   = self.F[:, k].reshape(-1, 1)    # F:   (Nt, Nk) --> f_k:   (Nt, 1)
            
            x = torch.abs((h_2_k @ self.Phi @ self.H_1 + h_3_k) @ f_k)**2
            x = torch.tensor(x.item(), dtype=torch.float32, device=device)
            interference += x 
        
        return interference

    def compute_SumRate(self):
        sum_rate, opt_rate = 0, 0
        interference = self.compute_interference()
        
        for k in range(self.Nk):
            h_2_k = self.H_2[k, :].reshape(1, -1)  # H_2: (Nk, Ns) --> h_2_k: (1, Ns)
            h_3_k = self.H_3[k, :].reshape(1, -1)  # H_3: (Nk, Nt) --> h_3_k: (1, Nt)
            f_k   = self.F[:, k].reshape(-1, 1)    # F:   (Nt, Nk) --> f_k:   (Nt, 1)
            
            x = torch.abs((h_2_k @ self.Phi @ self.H_1 + h_3_k) @ f_k)**2
            x = torch.tensor(x.item(), dtype=torch.float32, device=device)

            y = interference - x + self.awgn_var 
            rho_k = x / y
            
            # log_2() = log_10() / log(2)
            LOG2 = torch.log10(torch.tensor(2, dtype=torch.uint8, device=device))
            sum_rate += torch.log10(1 + rho_k) / LOG2
            opt_rate += torch.log10(1 + (x / self.awgn_var)) / LOG2

        del interference
        return sum_rate.item(), opt_rate.item()
    
    def compute_min_DownLinkRate(self):
        actual_rates = torch.zeros(self.Nk, dtype=torch.float32, device=device)
        optimal_rates = torch.zeros(self.Nk, dtype=torch.float32, device=device)
        interference = self.compute_interference()

        for k in range(self.Nk):
            h_2_k = self.H_2[k, :].reshape(1, -1)  # H_2: (Nk, Ns) --> h_2_k: (1, Ns)
            h_3_k = self.H_3[k, :].reshape(1, -1)  # H_3: (Nk, Nt) --> h_3_k: (1, Nt)
            f_k   = self.F[:, k].reshape(-1, 1)    # F:   (Nt, Nk) --> f_k:   (Nt, 1)
            
            x = torch.abs((h_2_k @ self.Phi @ self.H_1 + h_3_k) @ f_k)**2
            x = torch.tensor(x.item(), dtype=torch.float32, device=device)

            y = interference - x + self.awgn_var 
            rho_k = x / y
            
            # log_2() = log_10() / log(2)
            LOG2 = torch.log10(torch.tensor(2, dtype=torch.uint8, device=device))
            actual_rates[k]  = torch.log10(1 + rho_k) / LOG2
            optimal_rates[k] = torch.log10(1 + x / self.awgn_var) / LOG2

        del interference
        return actual_rates, optimal_rates
    
    def compute_MSE_matrix(self):
        H_tilde = self.compute_H_tilde()
        
        awgn_var_I = torch.zeros(size=(self.Nk, self.Nk), dtype=torch.complex64, device=device)
        awgn_var_I.fill_diagonal_(self.awgn_var)

        equivalent_I = torch.eye(self.Nk, dtype=torch.complex64, device=device) + awgn_var_I
        
        H_tilde_F = H_tilde @ self.F
        H_tilde_F_H = H_tilde_F.conj().T

        return equivalent_I - H_tilde_F - H_tilde_F_H + (H_tilde_F @ H_tilde_F_H)
    
    def compute_max_min_rate_reward(self):
        actual_rates, optimal_rates = self.compute_min_DownLinkRate()
        # log(actual_rates, optimal_rates, show=True)

        min_actual_rate, min_optimal_rate = torch.min(actual_rates).item(), torch.min(optimal_rates).item()
        # log(min_actual_rate, min_optimal_rate, show=True)

        return min_actual_rate, min_optimal_rate

    def compute_min_max_MSE_reward(self, shift=10, scale=10000):
        # compute the MSE for all users, identify the largest one, and multiply this max value by -1 to obtain the reward
        mse_matrix = self.compute_MSE_matrix()
        user_MSEs = torch.diag(mse_matrix).reshape(-1).to(device)
        current_max_MSE = torch.max(torch.real(user_MSEs)).item()
        
        optimal_mse = shift
        scaled_mse = np.log10(optimal_mse - current_max_MSE)*scale
        return scaled_mse, optimal_mse

    def step_with_actions(self, actions):
        H_tilde = self.compute_H_tilde()
        self.F = torch.sqrt(self.Pt / self.Nk).to(device) * (H_tilde.conj().T / torch.norm(H_tilde, 'fro').to(device))
        del H_tilde

        # convert the 'action,' represented as an integer ranging from 0 to 2^bits - 1, into a discrete phase shift
        est_rad_phases = self.action2phase(indices=actions, unit='radian')

        from torch import distributions as dist
        phase_errors = dist.VonMises(self.loc_mu, self.kappa_PE).sample((self.action_dim,)).to(device)

        actual_rad_phases = torch.add(est_rad_phases, phase_errors).to(device)

        # construct the Phi matrix using the discrete phase and the phase error
        actual_Phi_entries = self.compute_Phi_entries(angles=actual_rad_phases)
        self.Phi = torch.diagonal_scatter(self.Phi, actual_Phi_entries)
        del est_rad_phases, phase_errors, actual_rad_phases, actual_Phi_entries

        reward, opt_reward = self.compute_SumRate()
        return reward, opt_reward

    def step(self, action: np.uint8):
        self.episode_t += 1

        H_tilde = self.compute_H_tilde()
        self.F = torch.sqrt(self.Pt / self.Nk).to(device) * (H_tilde.conj().T / torch.norm(H_tilde, 'fro').to(device))
        del H_tilde

        self.prev_actions.appendleft(action)
        actions = torch.tensor(self.prev_actions, dtype=torch.uint8, device=device)

        # convert the 'action,' represented as an integer ranging from 0 to 2^bits - 1, into a discrete phase shift
        est_rad_phases = self.action2phase(indices=self.prev_actions, unit='radian')

        from torch import distributions as dist
        phase_errors = dist.VonMises(self.loc_mu, self.kappa_PE).sample((self.action_dim,)).to(device)

        actual_rad_phases = torch.add(est_rad_phases, phase_errors).to(device)

        # construct the Phi matrix using the discrete phase and the phase error
        actual_Phi_entries = self.compute_Phi_entries(angles=actual_rad_phases)
        self.Phi = torch.diagonal_scatter(self.Phi, actual_Phi_entries)
        del est_rad_phases, phase_errors, actual_rad_phases, actual_Phi_entries

        H_1_real, H_1_imag = torch.real(self.H_1).reshape(-1).to(device), torch.imag(self.H_1).reshape(-1).to(device)
        H_2_real, H_2_imag = torch.real(self.H_2).reshape(-1).to(device), torch.imag(self.H_2).reshape(-1).to(device)
        H_3_real, H_3_imag = torch.real(self.H_3).reshape(-1).to(device), torch.imag(self.H_3).reshape(-1).to(device)
        Phi_real, Phi_imag = torch.real(torch.diag(self.Phi)).to(device), torch.imag(torch.diag(self.Phi)).to(device)

        observation = torch.cat((H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, 
                                 Phi_real, Phi_imag, actions), dim=0)
        
        reward, opt_reward = self.compute_SumRate()
        # print(f"   opt_reward: {opt_reward}")
        
        truncated = (self.episode_t >= self._max_episode_steps)
        done = (opt_reward == reward) or truncated

        del H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, Phi_real, Phi_imag, action

        return np.array(observation.cpu(), dtype=np.float32), reward, done, truncated, {}

    def render(self):
        pass

    def close(self):
        pass


def checking_env(env):
    print(f"checking if the environment follows the Gym and SB3 interfaces\n")

    import warnings
    warnings.filterwarnings('ignore')
    
    # NOTE UserWarning: WARN: For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). 
    #                   See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.
    #      UserWarning: WARN: Not able to test alternative render modes due to the environment not having a spec. 
    #                   Try instantialising the environment through gymnasium.make
    gym_check_env(env)

    # NOTE UserWarning: We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) 
    #                   cf. https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
    #      UserWarning: Your action space has dtype int32, we recommend using np.float32 to avoid cast errors.
    sb3_check_env(env)

    print(f"Success!\n")

def test_env(env, episodes=4):
    rewards = []
    print(f"-"*16)
    print(f"episide:  1/{episodes}")
    print(f"-"*16)
    for episode in range(1, episodes + 1):
        if episode % 10 == 0: 
            print(f"-"*16)
            print(f"episide: {episode}/{episodes}")
            print(f"-"*16)
        
        done = False
        obs = env.reset()
        while not done:
            step = env.episode_t + 1
            if step % 100 == 0: 
                print(f"  step:  {step}/{env._max_episode_steps}")
            
            random_action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(random_action)
            rewards.append(reward)
            # print(f"   action:  {random_action}")
            # print(f"   reward:  {reward}")
            # print(f"   obs:     {np.squeeze(obs)[-env.Ns:]}")  # fetch the previous action of last step
            # print(f"-"*44)

    print(f"-"*32)
    print(f"random action of ({Nk}, {Nt}, {Ns}):")
    print(f"   mean:   {np.mean(rewards)}")
    print(f"   std:    {np.std(rewards)}")
    print(f"   max:    {np.max(rewards)}")
    print(f"   min:    {np.min(rewards)}")
    print(f"   action: {np.squeeze(obs)[-env.Ns:]}")
    print(f"   shape:  {obs.shape}")
    print(f"-"*32)

    return rewards

def plot_MSE_dist(env, episodes=100):
    import math
    num_bins = int(math.sqrt(episodes))

    mse_samples = torch.zeros(episodes).to(device)
    for episode in range(1, episodes + 1):
        if episode % num_bins == 0: print(f"{episode}")
        env.reset()
        random_action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(random_action)
        mse_samples[episode - 1] = info['mse']
    print(f"\nwriting mse_smaples to text file...\n")

    print(f"min MSE: {torch.min(mse_samples).cpu()}")
    print(f"max MSE: {torch.max(mse_samples).cpu()}")
    print(f"average: {torch.mean(mse_samples).cpu()}")
    print(f"median:  {torch.median(mse_samples).cpu()}\n")

    mse_dir = f"system_simulation/mse_dist-{date_today}/"
    if not os.path.exists(mse_dir):
        os.makedirs(mse_dir)

    with open(mse_dir + f"discrete-mse_samples-{episodes}.txt", "w") as txt_file:
        for i, mse in enumerate(mse_samples.cpu().numpy()):
            print(f"{mse}", file=txt_file)
        
        print("", file=txt_file)
        print(f"min MSE: {torch.min(mse_samples).cpu()}", file=txt_file)
        print(f"max MSE: {torch.max(mse_samples).cpu()}", file=txt_file)
        print(f"average: {torch.mean(mse_samples).cpu()}", file=txt_file)
        print(f"median:  {torch.median(mse_samples).cpu()}", file=txt_file)

    plt.figure(figsize=(9, 6))
    plt.axvline(x=torch.mean(mse_samples).cpu(), linewidth=1.5, color='#d62728', linestyle='dashed', label='Average MSE')
    # plt.axvline(x=np.min(mse_samples), linewidth=1.5, linestyle='dashed', color='blue', label='Min MSE')
    # plt.axvline(x=np.max(mse_samples), linewidth=1.5, linestyle='dashed', color='green', label='Max MSE')
    plt.axvline(x=torch.median(mse_samples).cpu(), linewidth=1.5, color='#2ca02c', linestyle='dashdot', label='Median MSE')
    plt.hist(
        mse_samples.cpu(), 
        bins=num_bins, 
        color='skyblue',
        # histtype='step', 
        label='MSE values'
    )
    plt.legend()
    
    plt.title(f"The distribution of MSE for {episodes} samples")
    plt.xlabel("MSE values")
    plt.ylabel("number of samples")

    plt.savefig(mse_dir + f"discrete-mse-{episodes}.png", dpi=200)

# ['blue',    'orange',  'green',   'red',     'purple',  'brown',   'pink',    'grey',    'darkyellow', 'cyan']
# ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',    '#17becf']

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

def brute_force_check(env):
    Ns = 4
    bits = 2
    n_actions = 2 ** bits
    size = n_actions**Ns
    actions = [] 
    for j in range(n_actions):
        for k in range(n_actions):
            for m in range(n_actions):
                for n in range(n_actions):
                    actions.append([j, k, m, n])
    actions = np.array(actions, dtype=np.uint8)
    
    rewards, opt_rewards = [0]*size, [0]*size
    for i, action in enumerate(actions):
        reward, opt_reward = env.step_with_actions(action)

        rewards[i] = reward
        opt_rewards[i] = opt_reward

    # log(actions, rewards, opt_rewards, show=False)

    plt.title('Confidence Interval')
    plt.xticks([0, 1], ['actual', 'optimal'])
    plt.xlabel("rewards")
    plt.ylabel("sum-rate")
    plot_confidence_interval1(0, rewards)
    plot_confidence_interval1(1, opt_rewards)
    plt.show()

    plt.title('actual vs. optimal')
    plt.xlabel('actions')
    plt.ylabel('sum-rate')
    
    plt.plot([i for i in range(size)], rewards, linestyle='-', color='#f44336', linewidth=2, label='actual')
    plt.plot([i for i in range(size)], opt_rewards, linestyle='-', color='#2187bb', linewidth=2, label='optimal')
    plt.legend(loc='best')
    plt.show()



if __name__ == "__main__":
    print(f"-"*64)

    # NOTE On RTX 2060 + i7-8700 with the setting of (Nk, Nt, Ns, obs.shape) = (4, 36, 36, 3276)
    # UserWarning: This system does not have apparently enough memory to store the complete replay buffer 26.22GB > 22.35GB

    # NOTE On RTX 3060Ti + i7-12700 with the setting of (Nk, Nt, Ns, obs.shape) = (56, 56, 4, 7180)
    # UserWarning: This system does not have apparently enough memory to store the complete replay buffer 59.39GB > 59.25GB

    Nk = 4
    Nt = 16
    Ns = 4
    episodes = 1
    TIMESTEPS = 300

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
        seed=33,
    )
    
    tic = time.perf_counter()

    test_env(env=env, episodes=episodes)
    # checking_env(env=env)  # NOTE Warnings have been ignored!
    
    toc = time.perf_counter()
    duration = (toc - tic)
    print(f"duration: {duration:0.4f} sec")
    

    
