# RIS-MISO-DRL
DRL-based RIS Configuration in RIS-assisted MU-MISO mmWave Systems for Min-Max MSE Optimization under HWI of Phase errors, Phase-dependent amplitude response model, and Imperfect CSI

~This is a story of switching research topics three times, starting a completely new project from scratch, and finishing it within six months—getting rejected by top conferences several times, and finally getting accepted by one.~\
I just got lucky to be accepted. My writing style isn’t meant to please the academic world—there’s no real genius in the research, and it’s completely useless in practice. But I tried really hard to explain everything I know as clearly as possible and to make the paper beginner-friendly. I hope it can benefit graduate students just like me.


## Paper
Working paper on IEEE International Conference on Communications\
Rejected by IEEE VTC2024-Spring\
Working paper on IEEE Wireless Communication Letters\
Rejected by IEEE Globecom 2024\
Working paper on IEEE Wireless Communications and Networking Conference\
Accepted by IEEE WCNC 2025

<img src='https://github.com/user-attachments/assets/8fd5c4ff-3869-4388-8ecd-75c2d35da6a7' width=65% height=65%>


## Note
~I'll upload the code once I graduate or the paper gets accepted~ Never mind.\
The code is a mess btw.\
This code is no longer maintained, and this is all that's left.


## Installing
- Install `Anaconda`
- Import the environment (though the YAML string-based config will be deprecated in the future)
  ```
  conda env create --file sb3.yaml --name sb3
  ```


## **Meeting Notes**
### **Spring 2024**
- meeting 03/12 [Current Progress](https://hackmd.io/@Shao-Heng/S1mISX_aa)
  - psi-to-MSE
- meeting 03/05 [Current Progress](https://hackmd.io/@Shao-Heng/SkO1sq6h6)
  - MSE-Matrix vs Signal-Tx 
  - Nk-to-MSE 
    - ```PPO-[3, 6, 8, 10]-16-16```
  - Nt-to-MSE 
    - ```PPO-2-[8, 16, 32, 64]-16```
  - Ns-to-MSE
    - ```PPO-2-16-[16, 36, 64, 100]```
  - beta_min-to-MSE
    - ```PPO-2-16-36```
  - psi-to-MSE
    - ```PPO-2-16-16```
- meeting 02/26 [Current Progress](https://hackmd.io/@Shao-Heng/ByN6NZm3T)
  - Bugs fixing
  - Validate self-identity
  - Ns-to-MSE
- meeting 02/20 [Current Progress](https://hackmd.io/@Shao-Heng/r1QyUde26)
  - Ns-to-MSE
- meeting 01/23 [Current Progress](https://hackmd.io/@Shao-Heng/SJxOIgKK6)
  - Nk-to-MSE
- meeting 01/09 [Current Progress](https://hackmd.io/@Shao-Heng/SJAcI8gO6)
  - Baseline method
    - ```Dominant Eigenvector Matching (DEM) heuristic``` for RIS Configuration
      - Performance: ```SDR``` > ```DEM``` > ```Power method```
      - Speed: ```DEM``` > ```Power method``` > ```SDR```
    - ```Max Ratio Transmission (MRT)``` for Precoder Design
  - Appendix
    - Validate MSE values with ```compute_raw_MSE()```
  - Reference
    - N. K. Kundu and M. R. McKay, "[RIS-Assisted MISO Communication: Optimal Beamformers and Performance Analysis](https://ieeexplore.ieee.org/abstract/document/9367504)," *2020 IEEE Globecom Workshops (GC Wkshps*, Taipei, Taiwan, 2020, pp. 1-6. (Cited by 13)
    - S. Ragi, E. K. P. Chong and H. D. Mittelmann, "[Polynomial-Time Methods to Solve Unimodular Quadratic Programs With Performance Guarantees](https://ieeexplore.ieee.org/document/8534389)," in *IEEE Transactions on Aerospace and Electronic Systems*, vol. 55, no. 5, pp. 2118-2127, Oct. 2019. (Cited by 6)
    - J. Gao, C. Zhong, X. Chen, H. Lin and Z. Zhang, "[Unsupervised Learning for Passive Beamforming](https://ieeexplore.ieee.org/document/8955968)," in *IEEE Communications Letters*, vol. 24, no. 5, pp. 1052-1056, May 2020.
- meeting 01/02 [Current Progress](https://hackmd.io/@Shao-Heng/BkAwzOAUa)
  - Inference result: ```PPO-2-16-[4, 16, 36, 64, 100]```
  - Confidence Interval: ```Random``` vs. ```Agent```
 

### **Fall 2023**
- meeting 12/19 [Current Progress](https://hackmd.io/@Shao-Heng/ryGop4WIT)
  - ```PPO-2-16-[4, 9, 16, 25, 36, 64]```
  - Comparison of different settings
- meeting 12/14 [Current Progress](https://hackmd.io/@Shao-Heng/SJwFurwLT)
  - M. -A. Badiu and J. P. Coon, "[Communication Through a Large Reflecting Surface With Phase Errors](https://ieeexplore.ieee.org/abstract/document/8869792)," in *IEEE Wireless Communications Letters*, vol. 9, no. 2, pp. 184-188, Feb. 2020.
  - R. Kozlica, S. Wegenkittl and S. Hiränder, "[Deep Q-Learning versus Proximal Policy Optimization: Performance Comparison in a Material Sorting Task](https://ieeexplore.ieee.org/abstract/document/10228056)," *2023 IEEE 32nd International Symposium on Industrial Electronics (ISIE)*, Helsinki, Finland, 2023, pp. 1-6.
- meeting 12/13 [Current Progress](https://hackmd.io/@Shao-Heng/Byf8mc6HT)
  - ```PPO-2-16-[4, 36]```
  - ```PPO-2-16-9```
  - ```PPO-2-16-25```
  - ```PPO-[2, 4, 6, 8, 10]-16-16```
  - ```PPO-10-16-36```
- meeting 12/05 [Current Progress](https://hackmd.io/@Shao-Heng/ryVx8HEra)
  - System validation: Brute force check
    - Try every possible combination of actions
    - Plot the Sum-Rate for every possible actions
  - Update ```Max Ratio Transmission (MRT)```
    - J. Gao, C. Zhong, X. Chen, H. Lin and Z. Zhang, "[Unsupervised Learning for Passive Beamforming](https://ieeexplore.ieee.org/document/8955968)," in *IEEE Communications Letters*, vol. 24, no. 5, pp. 1052-1056, May 2020.
    - D. Tse and P. Viswanath, Fundamentals of Wireless Communication, Cambridge, *U.K.:Cambridge Univ. Press*, 2005.
  - Training results
  - Inference results
  - Plotting functions
  - Future works
    - Adding more neurons in each layer
    - Deepen the network architecture
    - ```PPO``` default network architecture is ```[64, 64]``` for both actor and critic networks
- meeting 11/28 [Current Progress](https://hackmd.io/@Shao-Heng/r1FPxqhE6)
  - New feature: ```seed_everything()```
  - Bug fixing
  - Training results
    - ```PPO``` (1-4-4 to 4-4-4, and 4-16-16)
    - ```A2C``` (1-4-4 to 4-4-4)
  - Training of more complex settings with ```PPO (4-16-16)```
  - Training of more episodes with ```PPO``` (1000 episodes)
  - Comparison of all continuous agents (```TD3, DDPG, A2C, PPO, SAC```)
- meeting 11/21 [Current Progress](https://hackmd.io/@Shao-Heng/r1vOF-qm6)
  - Training results
    - Scaling rewards doesn't actually work
  - Channel model
    - General Communication Systems
  - Problem formulations
    - Max-min downlink rate
    - Sum-Rate Maximization
  - Future works
    - Go back to Box discrete
- meeting 11/16 [Summary](https://hackmd.io/@Shao-Heng/SyfqFZqQa)
  - System model
    - Downlink RIS-aided MU-MISO System
  - Channel model
    - mmWave Systems
    - General Communication Systems
  - Steering vectors
    - ULA, UPA, USPA
    - Array response implementations in torch
  - Problem formulations
    - Min-max MSE
    - Max-min downlink rate
    - Sum-Rate Maximization
- meeting 11/14 [Channel model - mmWave Systems](https://hackmd.io/@Shao-Heng/HJV4tZqmp)
  - P. Wang, J. Fang, L. Dai and H. Li, "[Joint Transceiver and Large Intelligent Surface Design for Massive MIMO mmWave Systems](https://ieeexplore.ieee.org/document/9234098)," in *IEEE Transactions on Wireless Communications*, vol. 20, no. 2, pp. 1052-1064, Feb. 2021. (Cited by 80)
  - K. Ying, Z. Gao, S. Lyu, Y. Wu, H. Wang and M. -S. Alouini, "[GMD-Based Hybrid Beamforming for Large Reconfigurable Intelligent Surface Assisted Millimeter-Wave Massive MIMO](https://ieeexplore.ieee.org/abstract/document/8964330)," in *IEEE Access*, vol. 8, pp. 19530-19539, 2020. (Cited by 91)
- meeting 11/07 [Steering vectors](https://hackmd.io/@Shao-Heng/SykqdaeXp)
  - K. Ying, Z. Gao, S. Lyu, Y. Wu, H. Wang and M. -S. Alouini, "[GMD-Based Hybrid Beamforming for Large Reconfigurable Intelligent Surface Assisted Millimeter-Wave Massive MIMO](https://ieeexplore.ieee.org/abstract/document/8964330)," in *IEEE Access*, vol. 8, pp. 19530-19539, 2020. (Cited by 91)
  - J. Yuan, Y. -C. Liang, J. Joung, G. Feng and E. G. Larsson, "[Intelligent Reflecting Surface-Assisted Cognitive Radio System](https://ieeexplore.ieee.org/document/9235486)," in *IEEE Transactions on Communications*, vol. 69, no. 1, pp. 675-687, Jan. 2021. (Cited by 130)
- meeting 10/31 [Random action rewards](https://hackmd.io/@Shao-Heng/HyJrsCnfa)
  - Random action rewards
  - TODO list
    - Inference
    - more anttenas do help
    - more bits don't actually help
- meeting 10/30 [Current Progress](https://hackmd.io/@Shao-Heng/Sy0RDJ5fa)
  - Training results
  - Inference results
- meeting 10/24 [Current Progress](https://hackmd.io/@Shao-Heng/S1Rj9u1Mp)
  - Bugs fixing
  - Training results
    - ```PPO, A2C DQN```
    - Compare differenct models with their best performance
    - Compare different numbers of users
    - Compare the complexity of different settings
- meeting 10/17 [Current Progress](https://hackmd.io/@Shao-Heng/S1NA0l6xa)
  - True ```Discrete``` action space version
  - Normalize ```Box``` action space
  - Apply ```GPU``` acceleration
  - Learn and Save
  - Load and Predict
- meeting 10/03 [Custom Gym Environment](https://hackmd.io/@Shao-Heng/r1oOJaMg6)
  - Environment built
  - Able to train
  - Future works
- meeting 09/26 [MU-MISO system model](https://hackmd.io/@Shao-Heng/ByD4m4lyp)
  - System model
  - Problem formulation
  - MSE derivation
- meeting 09/14 [MU-MIMO system model and possible methods](https://hackmd.io/@Shao-Heng/BksO3Akk6)
- meeting 09/12 [MSE derivation](https://hackmd.io/@Shao-Heng/ryFtN-jCh)
  - K. -Y. Chen, H. -Y. Chang, R. Y. Chang and W. -H. Chung, "[Hybrid Beamforming in mmWave MIMO-OFDM Systems via Deep Unfolding](https://ieeexplore.ieee.org/document/9860467)," *2022 IEEE 95th Vehicular Technology Conference: (VTC2022-Spring)*, Helsinki, Finland, 2022, pp. 1-7.
  - X. Zhao, T. Lin, Y. Zhu and J. Zhang, "[Partially-Connected Hybrid Beamforming for Spectral Efficiency Maximization via a Weighted MMSE Equivalence](https://ieeexplore.ieee.org/document/9467491)," in *IEEE Transactions on Wireless Communications*, vol. 20, no. 12, pp. 8218-8232, Dec. 2021.
- meeting 09/05 [Paper reading](https://hackmd.io/@Shao-Heng/S1WPuSJCn)
  - W. -Y. Chen, C. -Y. Wang, R. -H. Hwang, W. -T. Chen and S. -Y. Huang, "[Impact of Hardware Impairment on the Joint Reconfigurable Intelligent Surface and Robust Transceiver Design in MU-MIMO System](https://ieeexplore.ieee.org/document/10149520)," in *IEEE Transactions on Mobile Computing*.
- meeting 08/29 [Paper reading](https://hackmd.io/@Shao-Heng/r1OU-EVTh)
  - C. Huang, R. Mo and C. Yuen, "[Reconfigurable Intelligent Surface Assisted Multiuser MISO Systems Exploiting Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9110869)," in *IEEE Journal on Selected Areas in Communications*, vol. 38, no. 8, pp. 1839-1850, Aug. 2020. (Cited by 397)
- meeting 08/22 [Paper reading](https://hackmd.io/@Shao-Heng/ByzqNx-63)
  - Saglam Baturay, Doga Gurgunoglu, and Suleyman S. Kozat. "[Deep Reinforcement Learning Based Joint Downlink Beamforming and RIS Configuration in RIS-aided MU-MISO Systems Under Hardware Impairments and Imperfect CSI](https://arxiv.org/abs/2211.09702)." *arXiv preprint arXiv:2211.09702* (2022).
    - which was accepted to *2023 IEEE International Conference on Communications the 5th Workshop on Data Driven Intelligence for Networks and Systems (DDINS)*.

