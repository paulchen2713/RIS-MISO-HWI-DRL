import numpy as np
import torch

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])
import pandas as pd



def retrieve_name(var):
    import inspect
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def log(*argv, show=False):
    for arg in argv:
        print(f"-"*75)
        print(f"{retrieve_name(arg)}")
        if show:
            print(f"content: ")
            print(arg)
        print(f"type: {type(arg)}")
        if isinstance(arg, np.ndarray) or isinstance(arg, torch.Tensor):
            print(f"dtype: {arg.dtype}")
            print(f"shape: {arg.shape}")
        elif isinstance(arg, list) or isinstance(arg, str) or isinstance(arg, dict):
            print(f"len: {len(arg)}")


def plot_confidence_interval2(x, vals: list, num_samples=2000, z=1.96, horizontal_line_width=0.1, marker='o', color1='#2187bb', color2='#2187bb'):
    mean, std = vals
    confidence_interval = z * std / np.sqrt(num_samples)

    top, bot = mean - confidence_interval, mean + confidence_interval
    left = x - horizontal_line_width / 2
    right = x + horizontal_line_width / 2

    plt.plot(x, mean, marker=marker, color=color1)  # '#f44336'
    plt.plot([x, x], [top, bot], color=color2)   # '#2187bb'
    plt.plot([left, right], [top, top], color=color2)
    plt.plot([left, right], [bot, bot], color=color2)

    return confidence_interval

def random_vs_optimal_vs_agent(random, optimal, agent, title, xlabel, ylabel, loc, xticks, stamp):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i, data in enumerate(random):
        plot_confidence_interval2(i, data, marker='o', color1='#2ca02c')
    for i, data in enumerate(optimal):
        plot_confidence_interval2(i, data, marker='o', color1='#1f77b4')
    for i, data in enumerate(agent):
        plot_confidence_interval2(i, data, marker='o', color1='#f44336')
    
    plt.plot([i for i in range(len(random))], [data[0] for data in random], linestyle='--', color='#2ca02c', linewidth=1.5, label="Random")
    plt.plot([i for i in range(len(optimal))], [data[0] for data in optimal], linestyle='-.', color='#1f77b4', linewidth=1.5, label="Baseline")
    plt.plot([i for i in range(len(agent))], [data[0] for data in agent], linestyle='-', color='#f44336', linewidth=1.5, label="PPO")
    
    # plt.xscale('symlog')
    # plt.yscale('symlog')

    plt.xticks([i for i in range(len(agent))], xticks)
    plt.legend(loc=loc)

    # plt.savefig(f"./results/{title}-{stamp}.png", format='png')
    plt.savefig(f"./results/{title}-{stamp}.eps", format='eps')
    plt.show()


def opt_mse_matrix_vs_signal_tx(opt11, opt21, opt12, opt22, title, xlabel, ylabel, loc, xticks, stamp):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # ['blue',    'orange',  'green',   'red',     'purple',  'brown',   'pink',    'grey',    'darkyellow', 'cyan']
    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',    '#17becf']

    for i, data in enumerate(opt11):
        plot_confidence_interval2(i, data, marker='', color1='#2ca02c')
    for i, data in enumerate(opt21):
        plot_confidence_interval2(i, data, marker='', color1='#2ca02c')
    plt.plot([i for i in range(len(opt11))], [data[0] for data in opt11], 
             linestyle='-.', color='#1f77b4', linewidth=1.5, label="Simulate-MSE-Matrix-0")
    plt.plot([i for i in range(len(opt21))], [data[0] for data in opt21], 
             linestyle='-.', color='#ff7f0e', linewidth=1.5, label="Simulate-Signal-Tx-0")
    
    for i, data in enumerate(opt12):
        plot_confidence_interval2(i, data, marker='', color1='#2ca02c')
    for i, data in enumerate(opt22):
        plot_confidence_interval2(i, data, marker='', color1='#2ca02c')
    plt.plot([i for i in range(len(opt12))], [data[0] for data in opt12], 
             linestyle='-.', color='#2ca02c', linewidth=1.5, label="Simulate-MSE-Matrix-1")
    plt.plot([i for i in range(len(opt22))], [data[0] for data in opt22], 
             linestyle='-.', color='#d62728', linewidth=1.5, label="Simulate-Signal-Tx-1")

    plt.xticks([i for i in range(len(opt11))], xticks)
    plt.legend(loc=loc)

    # plt.savefig(f"./results/{title}-{stamp}.png", format='png')
    plt.savefig(f"./results/{title}-{stamp}.eps", format='eps')
    plt.show()

def mse_matrix_vs_signal_tx(x1, x2, title, xlabel, ylabel, loc, xticks, stamp):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # ['blue',    'orange',  'green',   'red',     'purple',  'brown',   'pink',    'grey',    'darkyellow', 'cyan']
    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',    '#17becf']

    for i, data in enumerate(x1):
        plot_confidence_interval2(i, data, marker='', color1='#d62728')
    for i, data in enumerate(x2):
        plot_confidence_interval2(i, data, marker='', color1='#2ca02c')
    plt.plot([i for i in range(len(x1))], [data[0] for data in x1], 
             linestyle='-.', color='#d62728', linewidth=1.5, label="Simulate-MSE-Matrix")
    plt.plot([i for i in range(len(x2))], [data[0] for data in x2], 
             linestyle='-.', color='#2ca02c', linewidth=1.5, label="Simulate-Signal-Tx")
    
    plt.xticks([i for i in range(len(x1))], xticks)
    plt.legend(loc=loc)

    plt.savefig(f"./results/{title}-{stamp}.eps", format='eps')
    plt.show()


def read_csv_and_extract_value(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Extract the 'Value' column and store it in a list
    xvalues = df['Step'].to_numpy()
    yvalues = df['Value'].to_numpy()

    return xvalues, yvalues

def plot_instant_reward(title, filename, xscale, yscale, stamp):
    # Specify the CSV file name
    csv_file_name = f'./data/{filename}.csv'

    # Call the function to read and extract 'Value' data
    xvalues, yvalues = read_csv_and_extract_value(csv_file_name)
    print(xvalues.shape, yvalues.shape)

    # Plotting the graph
    plt.plot(xvalues / xscale, yvalues / yscale, marker='', linestyle='-')
    # plt.yscale("symlog")
    plt.yticks([-0.5*i for i in range(1, 13, 2)])

    # Adding labels and title
    plt.xlabel('Training episodes')
    plt.ylabel('Mean episode reward rollout (2.048e4)')
    # plt.title('Instant Reward Graph')

    # Display and save the plot
    plt.savefig(f"./results/{title}-{stamp}.eps", format='eps')
    plt.show()
