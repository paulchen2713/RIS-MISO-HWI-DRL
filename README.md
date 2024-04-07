# RIS-MISO-DRL
DRL-based RIS Configuration in RIS-assisted MU-MISO mmWave Systems for Min-Max MSE Optimization under HWI of Phase errors, Phase-dependent amplitude response model, and Imperfect CSI

## Paper
Rejected by IEEE VTC2024-Spring\
Working paper on ~IEEE Wireless Coomunication Letters~

## Note
~I'll upload the code once I graduate or the paper gets accepted~ Never mind.

## Run
### 0. Dependency
$$
\caption{Hardware and Software Configuration}
\label{Tab:HW-and-SW-Settings}
\centering
\setlength{\tabcolsep}{6pt}       % Default value: 6pt
\renewcommand{\arraystretch}{1.2} % Default value: 1
\begin{tabular}{c c}
    \hline\hline
    \multicolumn{2}{c}{\textbf{Hardware Specifications}}   \\ \hline\hline
    CPU & Intel Core i7-12700 (4.9 GHz)          \\ \hline
    GPU & NVIDIA GeForce RTX 3060 Ti (8 GB)  \\ \hline
    RAM & DDR4-2400 64 GB                              \\ \hline\hline
    \multicolumn{2}{c}{\textbf{Software Versions}}  \\ \hline\hline
    python & 3.8.17  \\ \hline
    pytorch & 2.0.1  \\ \hline
    pytorch-cuda & 11.8 \\ \hline
    pytorch-mutex & 1.0  \\ \hline
    torchaudio & 2.0.2  \\ \hline
    torchvision & 0.15.2  \\ \hline
    stable-baselines3 & 2.1.0  \\ \hline
    gym & 0.21.0  \\ \hline
    gymnasium & 0.28.1  \\ \hline
    numpy & 1.24.3  \\ \hline
    matplotlib & 3.5.3  \\ \hline
    scienceplots & 2.1.0  \\ \hline
\end{tabular}
$$
### 1. Installing
- Set up a new conda environment, e.g. ```conda create -n sb3 python=3.8```
```
conda create -n [env_name] python=[version]
```

