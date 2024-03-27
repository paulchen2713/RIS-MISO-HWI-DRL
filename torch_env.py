import torch

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env as gym_check_env

from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3 import PPO

import numpy as np

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])
# plt.style.use(['ggplot'])

import random
import time
from datetime import date
import os


date_today = date.today()
# print(f"date_today: {date_today}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"device: {device}")

torch.backends.cudnn.benchmark = True  # Set to True should increase the speed of the code if the input sizes don't change

from utils import log


class RIS_MISO_Env(gym.Env):
    __slots__ = ('Nk', 'Nr', 'Nt', 'Ns', 'beta_min', 'mu_PDA', 'kappa_PDA', 'loc_mu', 'kappa_PE', 'psi', \
                 'awgn_var', 'Pt', 'bits', 'n_actions', 'action_dim', 'angle_set_deg', 'angle_set_rad', \
                 'splits', 'state_dim', 'H_1', 'H_2', 'H_2_est', 'H_3', 'H_3_est', 'F', 'Phi', 'Phi_est', \
                 'episode_t', '_max_episode_steps', 'seed', 'L')

    metadata = {"render_modes": ["console"]}

    def __init__(self, 
            num_users=2,
            num_UE_antennas=1,
            num_BS_antennas=16,
            num_RIS_elements=16,
            beta_min=0.09,  # 0.9,
            mu_PDA=0.21,   # 0.21,
            kappa_PDA=3.4,
            location_mu=0.6*np.pi,
            concentration_kappa=1.2,   
            uncertainty_factor=0.1,  # -10 dBm
            AWGN_var=0.000001,  # -30 dBm
            Tx_power=1,         #  30 dBm
            bits=1,
            max_episode_steps=20480,
            seed=33,
            L=4,
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
        self.Pt = torch.tensor(Tx_power, dtype=torch.float32, device=device)        # beamforming power consumption

        # Discrete actions
        #   action: RIS matrix
        self.bits = bits
        self.n_actions = 2**bits
        self.action_dim = self.Ns

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        
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
        self.splits = [-1 + 2 / self.n_actions * i for i in range(0, self.n_actions + 1)]

        # Continuous observation space
        #   state: H_1 + H_2 + H_3 + RIS matrix
        self.state_dim = 2 * (self.Ns * self.Nt + self.Nk * self.Ns + self.Nk * self.Nt) + 2*self.Ns + self.action_dim
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.state_dim,), dtype=np.float32)
        
        self.Phi_est = torch.eye(self.Ns, dtype=torch.complex64, device=device)
        self.Phi = torch.eye(self.Ns, dtype=torch.complex64, device=device)

        self._max_episode_steps = max_episode_steps
        self.seed = seed
        self.seed_everything(seed=self.seed)
        self.L = L

    def seed_everything(self, seed=None):
        print(f"seed_everything() is being called with random seed set to {seed}")

        random.seed(seed)
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

    def action2phase(self, indices: torch.ByteTensor, unit='radian') -> torch.Tensor:
        actual_phases = torch.zeros(self.action_dim, device=device)
        if unit == 'radian':
            for i, index in enumerate(indices):
                actual_phases[i] = torch.tensor(self.angle_set_rad[int(index)], dtype=torch.float32, device=device)
            return actual_phases
        elif unit == 'degree':
            for i, index in enumerate(indices):
                actual_phases[i] = torch.tensor(self.angle_set_deg[int(index)], dtype=torch.float32, device=device)
            return actual_phases

    def linear_interpolation(self, x) -> torch.ByteTensor:
        # we have a value x between x0 and x1 and we want a value y between y0 and y1, y is caluculated as follows:
        #   y = y0 + (x - x0) / (x1 - x0) * (y1 - y0), where [x0, x1] := [-1, 1] and [y0, y1] = [0, 2**bits - 1]
        y = torch.tensor((x + 1) / 2 * (self.n_actions - 1), dtype=torch.int8, device=device)
        return y

    def intervals_to_indices(self, act) -> torch.ByteTensor:
        for j in range(0, self.n_actions):
            if self.splits[j] < act <= self.splits[j + 1]: 
                return torch.tensor(j, dtype=torch.int8, device=device)
        return torch.randint(low=0, high=self.n_actions, size=(1,), dtype=torch.int16, device=device)

    def rescale(self, actions: torch.FloatTensor) -> torch.Tensor:
        # method 1: rounding
        #   torch.from_numpy(np.round(self._linear_interpolation(actions)).astype(int))
        
        # method 2: convert intervals to indices
        indices = torch.zeros(self.action_dim, dtype=torch.uint8, device=device)  # 2**bits
        for i, act in enumerate(actions):
            indices[i] = self.intervals_to_indices(act)
        return indices

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

    def compute_channels(self, L: int):
        a_BS_1 = self.get_ULA_sample(self.Nt)
        a_RIS_1 = self.get_USPA_sample(self.Ns)
        Rayleigh_1 = 0
        for _ in range(L):
            Rayleigh_1 += self.torch_CN(0, 0.01, (1,))
        H_1_channel_gain = self.torch_CN(0, 0.1, (1,)) + Rayleigh_1
        self.H_1 = np.sqrt(self.Nt * self.Ns / (L + 1)) * H_1_channel_gain * (a_RIS_1 @ a_BS_1.conj().T)  # NOTE H_1 shape: (Ns, Nt)
        del a_BS_1, a_RIS_1, Rayleigh_1, H_1_channel_gain

        a_RIS_2 = torch.zeros((self.Nk, self.Ns), dtype=torch.complex64, device=device)
        for i in range(self.Nk):
            a_RIS_2[i] = self.get_USPA_sample(self.Ns).conj().T
        Rayleigh_2 = 0
        for _ in range(L):
            Rayleigh_2 += self.torch_CN(0, 0.01, (1,))
        H_2_channel_gain = self.torch_CN(0, 0.1, (1,)) + Rayleigh_2
        self.H_2_est = np.sqrt(self.Ns / (L + 1))  * H_2_channel_gain * a_RIS_2                       # NOTE H_2 shape: (Nk, Ns)
        
        delta_H_2_entry = self.torch_CN(0, 1, (self.Nk, self.Ns))
        delta_H_2 = self.psi * (delta_H_2_entry / torch.norm(delta_H_2_entry, 'fro')).to(device)
        self.H_2 = self.H_2_est + delta_H_2    
        del a_RIS_2, Rayleigh_2, H_2_channel_gain, delta_H_2_entry, delta_H_2

        a_BS_3 = torch.zeros((self.Nk, self.Nt), dtype=torch.complex64, device=device)
        for i in range(self.Nk):
            a_BS_3[i] = self.get_ULA_sample(self.Nt).conj().T
        H_3_channel_gain = 0
        for _ in range(L + 1):
            H_3_channel_gain += self.torch_CN(0, 0.01, (1,))
        self.H_3_est = np.sqrt(self.Nt / (L + 1)) * H_3_channel_gain * a_BS_3                         # NOTE H_3 shape: (Nk, Nt)

        delta_H_3_entry = self.torch_CN(0, 1, (self.Nk, self.Nt))
        delta_H_3 = self.psi * (delta_H_3_entry / torch.norm(delta_H_3_entry, 'fro')).to(device)
        self.H_3 = self.H_3_est + delta_H_3
        del a_BS_3, H_3_channel_gain, delta_H_3_entry, delta_H_3


    # TODO maybe we could use 'zero-forcing' or 'SVD' to compute the beamforming matrix F later
    def compute_MRT_precoder(self):
        H_tilde = self.H_2_est @ self.Phi @ self.H_1 + self.H_3_est                           # NOTE F shape: (Nk, Nt)
        self.F = torch.sqrt(self.Pt / self.Nk).to(device) * (H_tilde.conj().T / torch.norm(H_tilde, 'fro').to(device))
        del H_tilde
    
    def add_phase_error(self, phase_shift: torch.Tensor):
        self.Phi_est = torch.diagonal_scatter(self.Phi_est, phase_shift)

        phase_errors = torch.distributions.VonMises(self.loc_mu, self.kappa_PE).sample((self.action_dim,)).to(device)
        actual_phase = torch.add(phase_shift, phase_errors).to(device)        
        actual_Phi_entries = self.compute_Phi_entries(angles=actual_phase)

        self.Phi = torch.diagonal_scatter(self.Phi, actual_Phi_entries)
        # log(actual_Phi_entries, self.Phi, show=True)

        del phase_errors, actual_phase, actual_Phi_entries

    def init_RIS_matrix(self):
        # method 1. random action initialization
        raw_action = -1 + 2*torch.rand(size=(self.action_dim,), dtype=torch.float32, device=device)

        # method 2. same initial action 
        #   torch.ones(size=(self.action_dim,), dtype=torch.float32, device=device)
        init_action = self.rescale(actions=raw_action)
        del raw_action

        estimated_phase_shifts = self.action2phase(indices=init_action, unit='radian')
        self.add_phase_error(estimated_phase_shifts)
        del estimated_phase_shifts
        
        return init_action

    def get_state_dimension(self) -> int:
        return self.state_dim

    def find_max_eigenvalue_index(self, eigenvalues):
        l2_norms = torch.zeros(size=(self.Ns,), dtype=torch.float32, device=device)
        for i, eigenvalue in enumerate(eigenvalues):
            l2_norms[i] = torch.sqrt(torch.real(eigenvalue)**2 + torch.imag(eigenvalue)**2).to(torch.float32)
        # log(l2_norms, show=True)
        return torch.argmax(l2_norms)

    def optimal_phase_shifts(self) -> torch.Tensor:
        phase_shifts = torch.zeros(size=(self.Nk, self.Ns), dtype=torch.float32, device=device)

        for k in range(self.Nk):
            h_2_k = self.H_2_est[k, :].reshape(1, -1).squeeze(0)  # H_2: (Nk, Ns) --> h_2_k: (1, Ns) --> (Ns,)
            diag_h = torch.zeros(size=(self.Ns, self.Ns), dtype=torch.complex64, device=device)  # (Ns, Ns)
            diag_h = torch.diagonal_scatter(diag_h, h_2_k)
            V = diag_h @ self.H_1                             # H_1: (Ns, Nt) --> V: (Ns, Nt)
            R = V @ V.T                                       # R:   (Ns, Ns)
            
            eigenvalues, eigenvectors = torch.linalg.eig(R)
            index = self.find_max_eigenvalue_index(eigenvalues)

            eigenvector_real, eigenvector_imag = torch.real(eigenvectors[index]), torch.imag(eigenvectors[index])
            phase_shift = torch.atan(eigenvector_imag / eigenvector_real)
            # log(eigenvalues, eigenvectors, index, phase_shift, show=True)

            phase_shifts[k] = phase_shift
        return phase_shifts

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.episode_t = 0

        # Initialize H_1, H_2, and H_3 channels
        self.compute_channels(L=self.L)

        # Max Ration Transmission (MRT)
        self.compute_MRT_precoder()

        # RIS 
        init_action = self.init_RIS_matrix()
        
        H_1_real, H_1_imag = torch.real(self.H_1).reshape(-1).to(device), torch.imag(self.H_1).reshape(-1).to(device)  # shape: (Ns * Nt,)
        H_2_real, H_2_imag = torch.real(self.H_2_est).reshape(-1).to(device), torch.imag(self.H_2_est).reshape(-1).to(device)  # shape: (Nk * Ns,)
        H_3_real, H_3_imag = torch.real(self.H_3_est).reshape(-1).to(device), torch.imag(self.H_3_est).reshape(-1).to(device)  # shape: (Nk * Nt,)
        Phi_real, Phi_imag = torch.real(torch.diag(self.Phi_est)).reshape(-1).to(device), torch.imag(torch.diag(self.Phi_est)).reshape(-1).to(device)
        
        observation = torch.cat((H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, 
                                 Phi_real, Phi_imag, init_action), dim=0).to(device)
        del H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, Phi_real, Phi_imag, init_action
    
        return np.array(observation.cpu(), dtype=np.float32), {}

    def compute_interference(self) -> torch.FloatTensor:
        interference = 0

        for k in range(self.Nk):
            h_2_k = self.H_2_est[k, :].reshape(1, -1)  # H_2: (Nk, Ns) --> h_2_k: (1, Ns)
            h_3_k = self.H_3_est[k, :].reshape(1, -1)  # H_3: (Nk, Nt) --> h_3_k: (1, Nt)
            f_k   = self.F[:, k].reshape(-1, 1)    # F:   (Nt, Nk) --> f_k:   (Nt, 1)
            
            x = torch.abs((h_2_k @ self.Phi @ self.H_1 + h_3_k) @ f_k)**2
            x = torch.tensor(x.item(), dtype=torch.float32, device=device)
            interference += x 
        
        return interference

    def compute_SumRate(self):
        sum_rate, opt_rate = 0, 0
        interference = self.compute_interference()
        
        for k in range(self.Nk):
            h_2_k = self.H_2_est[k, :].reshape(1, -1)  # H_2: (Nk, Ns) --> h_2_k: (1, Ns)
            h_3_k = self.H_3_est[k, :].reshape(1, -1)  # H_3: (Nk, Nt) --> h_3_k: (1, Nt)
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
    
    def compute_SumRate2(self):
        sum_rate, opt_rate = 0, 0

        for k in range(self.Nk):
            h_2_k = self.H_2_est[k, :].reshape(1, -1)  # H_2: (Nk, Ns) --> h_2_k: (1, Ns)
            h_3_k = self.H_3_est[k, :].reshape(1, -1)  # H_3: (Nk, Nt) --> h_3_k: (1, Nt)
            f_k   = self.F[:, k].reshape(-1, 1)    # F:   (Nt, Nk) --> f_k:   (Nt, 1)
            
            x = torch.abs((h_2_k @ self.Phi @ self.H_1 + h_3_k) @ f_k)**2
            x = torch.tensor(x.item(), dtype=torch.float32, device=device)

            # remove k-th column of F
            F_removed = torch.cat((self.F[:, :k], self.F[:, (k+1):]), dim=1)

            interference = torch.sum(torch.abs((h_2_k @ self.Phi @ self.H_1 + h_3_k) @ F_removed)**2, dtype=torch.float32)
            y = interference + self.awgn_var 
            rho_k = x / y
            
            # log_2() = log_10() / log(2)
            LOG2 = torch.log10(torch.tensor(2, dtype=torch.uint8, device=device))
            sum_rate += torch.log10(1 + rho_k) / LOG2
            opt_rate += torch.log10(1 + (x / self.awgn_var)) / LOG2

        return sum_rate.item(), opt_rate.item()

    def compute_min_DownLinkRate(self):
        actual_rates = torch.zeros(self.Nk, dtype=torch.float32, device=device)
        optimal_rates = torch.zeros(self.Nk, dtype=torch.float32, device=device)
        interference = self.compute_interference()

        for k in range(self.Nk):
            h_2_k = self.H_2_est[k, :].reshape(1, -1)  # H_2: (Nk, Ns) --> h_2_k: (1, Ns)
            h_3_k = self.H_3_est[k, :].reshape(1, -1)  # H_3: (Nk, Nt) --> h_3_k: (1, Nt)
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
        
    def compute_H_tilde(self):
        return self.H_2 @ self.Phi @ self.H_1 + self.H_3

    def compute_MSE_matrix(self) -> torch.Tensor:
        H_tilde = self.compute_H_tilde()
        
        awgn_var_I = torch.zeros(size=(self.Nk, self.Nk), dtype=torch.complex64, device=device)
        awgn_var_I.fill_diagonal_(self.awgn_var)

        equivalent_I = torch.eye(self.Nk, dtype=torch.complex64, device=device) + awgn_var_I
        
        H_tilde_F = H_tilde @ self.F
        H_tilde_F_H = H_tilde_F.conj().T

        return equivalent_I - H_tilde_F - H_tilde_F_H + (H_tilde_F @ H_tilde_F_H)
    
    def compute_raw_MSE(self):
        H_tilde = self.compute_H_tilde()

        transmit_signal_x = self.torch_CN(0, 1, size=(self.Nk, 1)) 

        for i in range(self.Nk):
            transmit_signal_x[i] = transmit_signal_x[i] / torch.sqrt((transmit_signal_x[i])**2)
        
        noise_vector_n = self.torch_CN(0, 0.001, size=(self.Nk, 1))
        received_signal_y = H_tilde @ self.F @ transmit_signal_x + noise_vector_n
        # log(transmit_signal_x, received_signal_y, noise_vector_n, show=True)

        mse_vector = []
        for k in range(self.Nk):
            x_real, x_imag = torch.real(transmit_signal_x[k]), torch.imag(transmit_signal_x[k])
            y_real, y_imag = torch.real(received_signal_y[k]), torch.imag(received_signal_y[k])
            mse_k = torch.abs(x_real - y_real)**2 + torch.abs(x_imag - y_imag)**2
            # print(f"MSE_{k}: {mse_k.item()}")
            mse_vector.append(mse_k.item())
        # log(mse_vector, show=True)
        return np.max(mse_vector) 


    def compute_SumRate_reward(self, scale=1000):
        sum_rate, opt_rate = self.compute_SumRate()

        # log(sum_rate, opt_rate, show=True)
        return sum_rate*scale, opt_rate*scale
    
    def compute_max_min_rate_reward(self, scale=1):
        actual_rates, optimal_rates = self.compute_min_DownLinkRate()
        
        min_actual_rate, min_optimal_rate = torch.min(actual_rates).item(), torch.min(optimal_rates).item()

        # log(actual_rates, optimal_rates, min_actual_rate, min_optimal_rate, show=True)
        return min_actual_rate*scale, min_optimal_rate*scale

    def compute_MMSE_reward(self, shift=0, scale=1):
        # compute the MSE for all users, identify the largest one, and multiply this max value by -1 to obtain the reward
        mse_matrix = self.compute_MSE_matrix()
        user_MSEs = torch.diag(mse_matrix).reshape(-1).to(device)
        current_max_MSE = torch.max(torch.real(user_MSEs)).item()

        return shift - current_max_MSE*scale

    def compute_STMSE_reward(self):
        worst_case_mse = self.compute_raw_MSE()
        return -1*worst_case_mse

    def random_passive_beamforming(self):
        # Initialize H_1, H_2, and H_3 channels
        # self.compute_channels(L=self.L)

        # Max Ration Transmission (MRT)
        self.compute_MRT_precoder()

        # Compute random phase shift
        self.init_RIS_matrix()

        actual = self.compute_MMSE_reward()
        # actual = self.compute_STMSE_reward()
        return actual

    def optimal_passive_beamforming(self):
        actuals = []

        # Initialize H_1, H_2, and H_3 channels
        # self.compute_channels(L=self.L)

        # Max Ration Transmission (MRT)
        self.compute_MRT_precoder()

        # Obtain the optimal phase shifts
        phase_shifts = self.optimal_phase_shifts()
        # log(phase_shifts, show=True)

        for i, phase_shift in enumerate(phase_shifts):
            self.add_phase_error(phase_shift)

            actual = self.compute_MMSE_reward()
            # actual = self.compute_STMSE_reward()
            actuals.append(actual)

        return actuals

    def step(self, raw_action):
        info = {'rewards':[]}
        self.episode_t += 1

        # Max Ration Transmission (MRT)
        self.compute_MRT_precoder()
        
        action = self.rescale(actions=raw_action)
        
        # convert the 'action,' represented as an integer ranging from 0 to 2^bits - 1, into a discrete phase shift
        estimated_phase_shifts = self.action2phase(indices=action, unit='radian')
        self.add_phase_error(estimated_phase_shifts)
        del estimated_phase_shifts
        
        H_1_real, H_1_imag = torch.real(self.H_1).reshape(-1).to(device), torch.imag(self.H_1).reshape(-1).to(device)  # shape: (Ns * Nt,)
        H_2_real, H_2_imag = torch.real(self.H_2_est).reshape(-1).to(device), torch.imag(self.H_2_est).reshape(-1).to(device)  # shape: (Nk * Ns,)
        H_3_real, H_3_imag = torch.real(self.H_3_est).reshape(-1).to(device), torch.imag(self.H_3_est).reshape(-1).to(device)  # shape: (Nk * Nt,)
        Phi_real, Phi_imag = torch.real(torch.diag(self.Phi_est)).reshape(-1).to(device), torch.imag(torch.diag(self.Phi_est)).reshape(-1).to(device)
        
        observation = torch.cat((H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, 
                                 Phi_real, Phi_imag, action), dim=0).to(device)
        
        reward = self.compute_MMSE_reward()
        # reward = self.compute_STMSE_reward()
        info['rewards'].append(reward)
        
        truncated = (self.episode_t >= self._max_episode_steps)
        done = truncated  # might need to find another optimal stopping criteria

        del H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, Phi_real, Phi_imag, action
        
        return np.array(observation.cpu(), dtype=np.float32), reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        pass

# ---------------------------------------------------------------------------------------------------------------------------------------------- #



def checking_env(env):
    print(f"checking if the environment follows the Gym and SB3 interfaces\n")
    # env.reset()

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

def test_env(Nk, Nt, Ns, env, total_steps, mini_steps):
    
    actuals = []

    # done = False
    obs = env.reset()
    for i in range(1, total_steps + 1):
        if i % mini_steps == 0: 
            env.compute_channels(L=L)
        
        random_action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(random_action)
        
        actuals.append(-1*reward)

        # mse_vector = env.compute_raw_MSE()
        # actuals.append(np.sum(mse_vector))

    print(f"-"*32)
    print(f"random action of ({Nk}, {Nt}, {Ns}):")
    print(f"   (mean, std):  [{np.mean(actuals)}, {np.std(actuals)}]")
    print(f"-"*32)


def plot_MSE_dist(env, episodes=100):
    import math
    num_bins = int(math.sqrt(episodes))

    mse_samples = torch.zeros(episodes).to(device)
    for episode in range(1, episodes + 1):
        if episode % num_bins == 0: print(f"{episode}")
        env.reset()
        random_action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(random_action)
        mse_samples[episode - 1] = info['max MSE']
    print(f"\nwriting mse_smaples to text file...\n")

    print(f"min MSE: {torch.min(mse_samples).cpu()}")
    print(f"max MSE: {torch.max(mse_samples).cpu()}")
    print(f"average: {torch.mean(mse_samples).cpu()}")
    print(f"median:  {torch.median(mse_samples).cpu()}\n")

    mse_dir = f"system_simulation/mse_dist-{date_today}/"
    if not os.path.exists(mse_dir):
        os.makedirs(mse_dir)

    with open(mse_dir + f"mse_samples-{episodes}.txt", "w") as txt_file:
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

    plt.savefig(mse_dir + f"mse-{episodes}.png", dpi=200)

# ['blue',    'orange',  'green',   'red',     'purple',  'brown',   'pink',    'grey',    'darkyellow', 'cyan']
# ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',    '#17becf']

def test_random_passive_beamforming(Nk, Nt, Ns, test_random_env, total_steps, mini_steps):
    test_random_env.reset()

    actuals = []       
    for i in range(1, total_steps + 1):
        if i % mini_steps == 0: 
            test_random_env.compute_channels(L=L)

        actual = test_random_env.random_passive_beamforming()
        actuals.append(-1*actual)

    print(f"-"*32)
    print(f"random action of ({Nk}, {Nt}, {Ns}):")
    print(f"   (mean, std):  [{np.mean(actuals)}, {np.std(actuals)}]")
    print(f"-"*32)

def test_optimal_beamforming(Nk, Nt, Ns, test_optimal_env, total_steps, mini_steps):
    test_optimal_env.reset()

    actuals = [[] for _ in range(Nk)]
    for i in range(1, total_steps + 1):
        if i % mini_steps == 0: 
            test_optimal_env.compute_channels(L=L)

        actual = test_optimal_env.optimal_passive_beamforming()
        # mse_vector = test_optimal_env.compute_raw_MSE()
        for k in range(Nk):
            actuals[k].append(-1*actual[k])
            # actuals[k].append(np.sum(mse_vector))
    
    for k in range(Nk):
        if k >= 2: continue
        print(f"-"*32)
        print(f"optimal of ({Nk}, {Nt}, {Ns}) for use {k}:")
        print(f"   (mean, std):  [{np.mean(actuals[k])}, {np.std(actuals[k])}]")
        print(f"-"*32)


def validate_autocorrelation(Nk, env, num_samples=10000):
    # Check if the autocorrelation of the transmit signals is equal to the identity matrix.
    # a = E[x*x^H]
    a = np.zeros(shape=(Nk,Nk), dtype=np.float32)
    for i in range(num_samples):
        if i % 1000 == 0: print(f"#{i}")
        transmit_signal_x = env.torch_CN(0, 1, size=(Nk, 1))

        for j in range(Nk):
            transmit_signal_x[j] = transmit_signal_x[j] / torch.sqrt((transmit_signal_x[j])**2)
        
        autocorrelation = transmit_signal_x * transmit_signal_x.conj().T
        # log(transmit_signal_x, autocorrelation, show=True)
        
        for j in range(Nk):
            for k in range(Nk):
                a[j][k] += torch.real(autocorrelation[j][k]).item() / num_samples
    
    log(a, show=True)



if __name__ == "__main__":
    print(f"-"*64)
    
    # Nk = 2   # [1, 2, 3, 4], [4, 6, 8, 10]
    # Nt = 16  # [16, 24, 32, 40]
    # Ns = 100  # [16, 36, 64, 100]
    TIMESTEPS = 20000
    mini_steps = 1000
    seed = 33
    L = 4
    
    def main(Nk, Nt, Ns):
        tic = time.perf_counter()

        layers = [Ns*2**i for i in range(5, 0, -1)]
        
        env1 = RIS_MISO_Env(Nk, 1, Nt, Ns, max_episode_steps=TIMESTEPS, seed=seed, L=L)
        policy_kwargs = dict(
            activation_fn=torch.nn.Tanh,  # torch.nn.ReLU, torch.nn.Tanh
            net_arch=dict(
                pi=layers,
                vf=layers,
            )
        )
        model = PPO('MlpPolicy', env1, policy_kwargs=policy_kwargs, ent_coef=0.001, verbose=1)
        state_dim = 2 * (Ns*Nt + Nk*Ns + Nk*Nt) + 3*Ns
        # print(state_dim, Ns*2**5)
        # print(model.policy)  
        
        checking_env(env=env1)  # NOTE Warnings have been ignored!

        def test_random(num_env1_resets=0):
            if num_env1_resets == 0: 
                test_env(Nk, Nt, Ns, env1, TIMESTEPS, mini_steps)
            else:
                print(f"number of env1 resets:  {num_env1_resets}")
                for _ in range(num_env1_resets): env1.reset()
                test_env(Nk, Nt, Ns, env1, TIMESTEPS, mini_steps)
        
        test_random()

        env3 = RIS_MISO_Env(Nk, 1, Nt, Ns, max_episode_steps=TIMESTEPS, seed=seed, L=L)
        def test_optimal(num_env3_resets=0):
            if num_env3_resets == 0:
                test_optimal_beamforming(Nk, Nt, Ns, env3, TIMESTEPS, mini_steps)
            else:
                print(f"number of env3 resets:  {num_env3_resets}")
                for _ in range(num_env3_resets): env3.reset()
                test_optimal_beamforming(Nk, Nt, Ns, env3, TIMESTEPS, mini_steps)
        
        test_optimal()
        
        toc = time.perf_counter()
        duration = (toc - tic)
        print(f"duration: {duration:0.4f} sec\n")

    Ns_to_MSE = [i**2 for i in range(2, 11)]
    Ns_to_MSE = [16, 36, 64, 100]

    # for i in Ns_to_MSE:
    #     main(2, 16, i)

    main(2, 4, 16)
    