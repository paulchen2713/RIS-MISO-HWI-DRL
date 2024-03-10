import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
# plt.style.use(['seaborn-whiltegrid'])

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
        if isinstance(arg, np.ndarray):
            print(f"shape: {arg.shape}")
        elif isinstance(arg, list) or isinstance(arg, str) or isinstance(arg, dict):
            print(f"len: {len(arg)}")


np.random.seed(0)

uncertainty_factor = 1e-3

Nt, Nk = 2, 2
Ns = 4

print(f"printing out the channel uncertainties in matrices")
H_2_est = np.random.normal(0, np.sqrt(0.5), (Nk, Ns)) + 1j*np.random.normal(0, np.sqrt(0.5), (Nk, Ns))
H_3_est = np.random.normal(0, np.sqrt(0.5), (Nk, Nt)) + 1j*np.random.normal(0, np.sqrt(0.5), (Nk, Nt))

delta_H_2_entry = np.random.normal(0, 1, (Nk, Ns)) + 1j*np.random.normal(0, 1, (Nk, Ns))
delta_H_2_norm = linalg.norm(delta_H_2_entry)
delta_H_2 = uncertainty_factor * (delta_H_2_entry / delta_H_2_norm)
H_2 = H_2_est + delta_H_2
log(H_2_est, delta_H_2_entry, delta_H_2_norm, delta_H_2, H_2)

delta_H_3_entry = np.random.normal(0, 1, (Nk, Nt)) + 1j*np.random.normal(0, 1, (Nk, Nt))
delta_H_3_norm = linalg.norm(delta_H_3_entry)
delta_H_3 = uncertainty_factor * (delta_H_3_entry / linalg.norm(delta_H_3_entry))
H_3 = H_3_est + delta_H_3
log(H_3_est, delta_H_3_entry, delta_H_3_norm, delta_H_3, H_3)


print(f"printing out the channel uncertainties in matrices")
h2_est = np.random.normal(0, np.sqrt(0.5), (Ns,)) + 1j*np.random.normal(0, np.sqrt(0.5), (Ns,))
delta_h2_entry = np.random.normal(0, 1, (Ns,)) + 1j*np.random.normal(0, 1, (Ns,))
delta_h2_norm = linalg.norm(delta_h2_entry)
delta_h2 = uncertainty_factor * (delta_h2_entry / delta_h2_norm)
h2 = h2_est + delta_h2
log(h2_est, delta_h2_entry, delta_h2_norm, delta_h2, h2)

h3_est = np.random.normal(0, np.sqrt(0.5), (Nt,)) + 1j*np.random.normal(0, np.sqrt(0.5), (Nt,))
delta_h3_entry = np.random.normal(0, 1, (Nt,)) + 1j*np.random.normal(0, 1, (Nt,))
delta_h3_norm = linalg.norm(delta_h3_entry)
delta_h3 = uncertainty_factor * (delta_h3_entry / delta_h3_norm)
h3 = h3_est + delta_h3
log(h3_est, delta_h3_entry, delta_h3_norm, delta_h3, h3)

