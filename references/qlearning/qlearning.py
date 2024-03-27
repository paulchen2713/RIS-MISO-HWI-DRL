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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from collections import deque
import time
from datetime import date
import os


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
                 'state_dim', 'H_1', 'H_2', 'H_3', 'F', 'Phi', 'episode_t', '_max_episode_steps', 'prev_actions')

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
        self.beta_min = torch.tensor(beta_min, dtype=torch.float32).to(device)    # \beta_{min}
        self.mu_PDA = torch.tensor(mu_PDA, dtype=torch.float32).to(device)        # \mu
        self.kappa_PDA = torch.tensor(kappa_PDA, dtype=torch.float32).to(device)  # \kappa
        #   Phase Error 
        self.loc_mu = location_mu            # Von Mises Location factor \mu
        self.kappa_PE = concentration_kappa  # Von Mises Concentration factor \kappa
        #   Channel uncertainty factor
        self.psi = torch.tensor(uncertainty_factor, dtype=torch.float32).to(device)        # uncertainty factor \psi

        # 
        self.awgn_var = torch.tensor(AWGN_var, dtype=torch.float32).to(device)  # sigma_n^2
        self.Pt = torch.tensor(Tx_power_dBm, dtype=torch.float32).to(device)    # total transmit power / beamforming power consumption

        # Discrete actions
        #   action: RIS matrix
        self.bits = bits
        self.n_actions = 2 ** bits
        self.action_dim = self.Ns

        self.action_space = spaces.Discrete(start=0, n=self.n_actions)  # {start, ..., start + n - 1}
        self.prev_actions = deque(maxlen=self.action_dim)

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

        # Continuous observation space
        #   state: H_1 + H_2 + H_3 + previous RIS matrix
        self.state_dim = 2*self.Ns 
        # self.state_dim = 2 * (self.Ns * self.Nt + self.Nk * self.Ns + self.Nk * self.Nt) + 2*self.Ns + self.action_dim
        # log(self.state_dim)  # 336

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.state_dim,), dtype=np.float32)
        # print(self.observation_space.shape)  # shape: (336,)

        self.Phi = torch.eye(self.Ns, dtype=torch.complex64).to(device)

        self._max_episode_steps = max_episode_steps


    def _compute_PDA(self, angle_rad):
        # beta(\theta) = (1 - \beta_{min}) * ((sin(\theta - \mu) + 1) / 2)^\kappa + \beta_{min}
        beta_PDA = (1 - self.beta_min) * ((torch.sin(angle_rad - self.mu_PDA).to(device) + 1) / 2) ** self.kappa_PDA + self.beta_min
        return beta_PDA

    def _Eulers_formula(self, amplitude, angle_rad):
        return amplitude * (torch.cos(angle_rad).to(device) + 1j*torch.sin(angle_rad).to(device))

    def _compute_Phi_entries(self, angles):
        actual_Phi_entries = torch.zeros(self.action_dim, dtype=torch.complex64).to(device)
        for i, angle in enumerate(angles):
            actual_Phi_entries[i] = self._Eulers_formula(self._compute_PDA(angle), angle)
        return actual_Phi_entries

    def _action2phase(self, indices, unit='radian'):
        actual_phases = torch.zeros(self.action_dim).to(device)
        if unit == 'radian':
            for i, index in enumerate(indices.cpu().numpy()):
                actual_phases[i] = self.angle_set_rad[index]
            return actual_phases
        elif unit == 'degree':
            for i, index in enumerate(indices.cpu().numpy()):
                actual_phases[i] = self.angle_set_deg[index]
            return actual_phases

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        info = {}
        
        self.episode_t = 0

        self.H_1 = torch.normal(0, np.sqrt(0.5), (self.Ns, self.Nt)).to(device) \
                 + 1j*torch.normal(0, np.sqrt(0.5), (self.Ns, self.Nt)).to(device)

        H_2_est = torch.normal(0, np.sqrt(0.5), (self.Nk, self.Ns)).to(device) \
                + 1j*torch.normal(0, np.sqrt(0.5), (self.Nk, self.Ns)).to(device)
        delta_H_2_entry = torch.normal(0, 1, (self.Nk, self.Ns)).to(device) \
                        + 1j*torch.normal(0, 1, (self.Nk, self.Ns)).to(device)
        delta_H_2 = self.psi * (delta_H_2_entry / torch.norm(delta_H_2_entry, 'fro')).to(device)
        self.H_2 = H_2_est + delta_H_2

        H_3_est = torch.normal(0, np.sqrt(0.5), (self.Nk, self.Nt)).to(device) \
                + 1j*torch.normal(0, np.sqrt(0.5), (self.Nk, self.Nt)).to(device)
        delta_H_3_entry = torch.normal(0, 1, (self.Nk, self.Nt)).to(device) \
                        + 1j*torch.normal(0, 1, (self.Nk, self.Nt)).to(device)
        delta_H_3 = self.psi * (delta_H_3_entry / torch.norm(delta_H_3_entry, 'fro')).to(device)
        self.H_3 = H_3_est + delta_H_3
        # log(self.H_1, delta_H_2_entry, delta_H_2, H_2_est, self.H_2, delta_H_3_entry, delta_H_3, H_3_est, self.H_3)

        # Max Ration Transmission (MRT)
        complex_power = torch.ones((self.Nt, self.Nk)).to(device) + 1j*torch.ones((self.Nt, self.Nk)).to(device)
        normalized_complex_power = complex_power / torch.norm(complex_power, 'fro').to(device)
        self.F = torch.sqrt(self.Pt / self.Nk).to(device) * normalized_complex_power
        # log(complex_power, normalized_complex_power, self.F)

        # TODO maybe we could use 'zero-forcing' or 'SVD' to compute the beamforming matrix F later

        # RIS 
        init_action = np.random.randint(low=0, high=self.n_actions, size=(self.action_dim,))
        for act in init_action:
            self.prev_actions.append(act)
        actions = torch.from_numpy(np.array(list(self.prev_actions), dtype=np.float32)).to(device)
        # log(self.prev_actions, actions)

        est_rad_phases = self._action2phase(indices=actions, unit='radian')
        est_Phi_entries = self._compute_Phi_entries(angles=est_rad_phases)
        self.Phi = torch.diagonal_scatter(self.Phi, est_Phi_entries)
        # log(init_action, self.angle_set_rad, est_rad_phases, est_Phi_entries, torch.diagonal(self.Phi))
        
        # H_1_real, H_1_imag = torch.real(self.H_1).reshape(-1).to(device), torch.imag(self.H_1).reshape(-1).to(device)  # shape: (Ns * Nt,)
        # H_2_real, H_2_imag = torch.real(self.H_2).reshape(-1).to(device), torch.imag(self.H_2).reshape(-1).to(device)  # shape: (Nk * Ns,)
        # H_3_real, H_3_imag = torch.real(self.H_3).reshape(-1).to(device), torch.imag(self.H_3).reshape(-1).to(device)  # shape: (Nk * Nt,)
        Phi_real, Phi_imag = torch.real(torch.diag(self.Phi)).reshape(-1).to(device), torch.imag(torch.diag(self.Phi)).reshape(-1).to(device)
        # log(H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, Phi_real, Phi_imag)

        observation = torch.cat((Phi_real, Phi_imag), dim=0).to(device)
        # observation = torch.cat((H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, 
        #                          Phi_real, Phi_imag, actions), dim=0).to(device)
        # print(observation.shape)  # (336,)
        # log(observation)
        return np.array(observation.cpu(), dtype=np.float32), info

    def _compute_H_tilde(self):
        return self.H_2 @ self.Phi @ self.H_1 + self.H_3
    
    def _compute_MSE_matrix(self):
        H_tilde = self._compute_H_tilde()
        
        awgn_var_I = torch.zeros(size=(self.Nk, self.Nk), dtype=torch.complex64).to(device)
        awgn_var_I.fill_diagonal_(self.awgn_var)

        equivalent_I = torch.eye(self.Nk, dtype=torch.complex64).to(device) + awgn_var_I
        
        H_tilde_F = H_tilde @ self.F
        H_tilde_F_H = H_tilde_F.conj().T
        # log(H_tilde, self.F, awgn_var_I, equivalent_I, H_tilde_F, H_tilde_F_H.conj().T, H_tilde_F_H)

        return equivalent_I - H_tilde_F - H_tilde_F_H + (H_tilde_F @ H_tilde_F_H)
    
    def step(self, action):
        info = {'max MSE': -1}
        self.episode_t += 1
        self.prev_actions.append(action)
        actions = torch.from_numpy(np.array(list(self.prev_actions), dtype=np.float32)).to(device)

        # convert the 'action,' represented as an integer ranging from 0 to 2^bits - 1, into a discrete phase shift
        est_rad_phases = self._action2phase(indices=actions, unit='radian')
        phase_errors = torch.from_numpy(vonmises_line(loc=self.loc_mu, kappa=self.kappa_PE).rvs(self.action_dim)).to(device)
        actual_rad_phases = torch.add(est_rad_phases, phase_errors).to(device)
        # log(est_rad_phases, phase_errors, actual_rad_phases)

        # construct the Phi matrix using the discrete phase and the phase error
        actual_Phi_entries = self._compute_Phi_entries(angles=actual_rad_phases)
        self.Phi = torch.diagonal_scatter(self.Phi, actual_Phi_entries)
        # log(self.Phi, actual_Phi_entries, torch.diagonal(self.Phi))

        # compute the MSE for all users, identify the largest one, and multiply this max value by -1 to obtain the reward
        mse_matrix = self._compute_MSE_matrix()
        user_MSEs = torch.diag(mse_matrix).reshape(-1).to(device)
        current_max_MSE = torch.max(torch.real(user_MSEs)).to(device)
        info['max MSE'] = max(info['max MSE'], current_max_MSE)
        # log(user_MSEs, np.real(user_MSEs), np.max(np.real(user_MSEs)))

        # H_1_real, H_1_imag = torch.real(self.H_1).reshape(-1).to(device), torch.imag(self.H_1).reshape(-1).to(device)  # shape: (Ns * Nt,)
        # H_2_real, H_2_imag = torch.real(self.H_2).reshape(-1).to(device), torch.imag(self.H_2).reshape(-1).to(device)  # shape: (Nk * Ns,)
        # H_3_real, H_3_imag = torch.real(self.H_3).reshape(-1).to(device), torch.imag(self.H_3).reshape(-1).to(device)  # shape: (Nk * Nt,)
        Phi_real, Phi_imag = torch.real(torch.diag(self.Phi)).reshape(-1).to(device), torch.imag(torch.diag(self.Phi)).reshape(-1).to(device)
        # log(H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, Phi_real, Phi_imag)

        observation = torch.cat((Phi_real, Phi_imag), dim=0).to(device)
        # observation = torch.cat((H_1_real, H_1_imag, H_2_real, H_2_imag, H_3_real, H_3_imag, 
        #                          Phi_real, Phi_imag, actions), dim=0).to(device)
        # log(observation)
        
        opt_reward = 230
        reward = opt_reward - current_max_MSE.item()
        # log(reward, opt_reward)

        truncated = (self.episode_t >= self._max_episode_steps)
        done = (opt_reward == reward) or truncated
        return np.array(observation.cpu(), dtype=np.float32), reward, done, truncated, info


env = RIS_MISO_Env(
        num_users=1,
        num_UE_antennas=1,
        num_BS_antennas=1,
        num_RIS_elements=4,
        beta_min=0.2,
        mu_PDA=0.1,
        kappa_PDA=1.5,
        location_mu=0.43*np.pi,
        concentration_kappa=1.5,   
        uncertainty_factor=0.001,
        AWGN_var=0.01,
        Tx_power_dBm=30,
        bits=2,
        max_episode_steps=10000,
)

# print(env.reset())  # calling env.reset() will give us a random state, e.g. [-0.59585701  0.        ]

# Q-Learning settings
LEARNING_RATE = 0.1  # [0.1, 0.0001]
DISCOUNT = 0.95      # [0.95, 0.99]
EPISODES = 100000     # 25_000
# SHOW_EVERY = 10000    # 1000

# Quantization settings
# DISCRETE_OS_SIZE = [20, 20]  # use 20 groups/buckets for each range. (20 units) 
DISCRETE_OS_SIZE = [8] * len(env.observation_space.high)
# [(0.6 - (-1.2)) / 20, (0.07 - (-0.07)) / 20] == [1.8 / 20, 0.14 / 20] == [0.09, 0.007]
discrete_os_window_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Epsilon-Greedy Exploration settings
epsilon = 1  # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING + 1)

# For stats 
epi_rewards = [] 
aggr_epi_rewards = {'epi': [], 'avg': [], 'max': [], 'min': []}
STATS_EVERY = 1000
SAVE_QTABLE_EVERY = 10000


# ---------------------------------------------------------------------------------------------- #
print(f"number of actions: {env.action_space.n}")  # 3 
# there are "3" actions we can pass: 0 means push left, 1 is stay still, and 2 means push right
print(f"random action: {env.action_space.sample()}")  # 0, 1, or 2

# we can query the enviornment to find out the possible ranges for each of these state values
print(f"state values range: ") 
print(f"  {env.observation_space.high}")  # [0.6  0.07]
print(f"  {env.observation_space.low}")   # [-1.2  -0.07]

print(f"len(env.observation_space.high): {len(env.observation_space.high)}")  # 
print(f"discrete observation space window size: {discrete_os_window_size}")   # [0.09  0.007]

size = DISCRETE_OS_SIZE + [env.action_space.n]
print(f"q_table size: {size}")  # [20, 20, 3] is a 20x20x3 shape
# ---------------------------------------------------------------------------------------------- #


q_table = np.random.uniform(low=-1, high=1, size=DISCRETE_OS_SIZE + [env.action_space.n])

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_window_size
    # we use this tuple to look up the 3 Q-values for the available actions in the 'q_table'
    return tuple(discrete_state.astype(np.int8))  


# ---------------------------------------------------------------------------------------------- #
test_state = get_discrete_state(env.observation_space.sample())
print(type(test_state))                       # <class 'tuple'>
print(test_state)                             # (6, 10)
test_action = np.argmax(q_table[test_state])  
print(type(test_state + (test_action, )))     # <class 'tuple'>
print(test_state + (test_action, ))           # (6, 10, 2)
# ---------------------------------------------------------------------------------------------- #

tic = time.perf_counter()
env.reset()
for episode in range(EPISODES):
    episode_reward = 0  # current episode reward
    
    # if episode % SHOW_EVERY == 0:
    #     render = True
    #     print(episode)
    # else:
    #     render = False
    
    # Get the initial state values from env.reset() and store it to 'discrete_state'
    discrete_state = get_discrete_state(env.observation_space.sample())
    done = False
    while not done:
        # Take an action
        if epsilon > 0 and np.random.random() > epsilon:
            # Get action from Q-table
            action = np.argmax(q_table[discrete_state])  # get the index of the greatest Q-value in the q_table
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)
        
        new_state, reward, done, truncated, info = env.step(action=action)
        # e.g. reward: -1.0, state := [position, velocity] == [-0.5519343  -0.01300341]
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        # if episode % SHOW_EVERY == 0:
        #     print(reward, new_state)  
        #     # env.render()

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q-value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])  # get the max value, not the index

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action:
            #   Q_{new}(s_t, a_t) <-- (1 - \alpha) \cdot Q(s_t, a_t) +    \alpha      \cdot (  r_t   +    \gamma      \cdot \max_{a} Q(s_{t + 1}, a))
            #       new-Q-value                            old-value    learning-rate         reward   discount-fator       estimate-of-future-value
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q


        # Simulation ended (for any reson) - if goal position is achieved - update Q-value with reward directly
        elif reward >= 230:
            # q_table[discrete_state + (action, )] = reward
            q_table[discrete_state + (action,)] = reward
            print(f"reached max reward: ")
            print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
        
        # Updating the old state with the new state
        discrete_state = new_discrete_state
    
    # Decaying is being done every episode if episode number is within decaying range
    if START_EPSILON_DECAYING <= episode <= END_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Save stats for further analysis
    epi_rewards.append(episode_reward)
    if episode > 0 and episode % STATS_EVERY == 0:
        average_reward = sum(epi_rewards[-STATS_EVERY:]) / STATS_EVERY  # running average of past 'STATS_EVERY' number of rewards 
        aggr_epi_rewards['epi'].append(episode)
        aggr_epi_rewards['avg'].append(average_reward)
        aggr_epi_rewards['max'].append(max(epi_rewards[-STATS_EVERY:]))
        aggr_epi_rewards['min'].append(min(epi_rewards[-STATS_EVERY:]))
        # ':>5d' pad decimal with zeros (left padding, width 5), ':>4.1f' format float 1 decimal places (left padding, width 4)
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

    # Save the q_table for every SAVE_QTABLE_EVERY number of episodes
    # if episode > 0 and episode % SAVE_QTABLE_EVERY == 0 and epsilon <= 0:
    #     np.save(f"qtables/{episode}-qtable.npy", q_table) 

# env.close()
toc = time.perf_counter()
duration = (toc - tic)
print(f"\nduration: {duration:0.4f} sec")

# Plot the reward figure 
plt.plot(aggr_epi_rewards['epi'], aggr_epi_rewards['avg'], label="avg rewards")
plt.plot(aggr_epi_rewards['epi'], aggr_epi_rewards['max'], label="max rewards")
plt.plot(aggr_epi_rewards['epi'], aggr_epi_rewards['min'], label="min rewards")

plt.legend(loc='best')
plt.xlabel(f"episodes")
plt.ylabel(f"rewards")
plt.grid(True)

# plt.show()
plt.savefig(f"figures/avg-max-min-rewards.png")

