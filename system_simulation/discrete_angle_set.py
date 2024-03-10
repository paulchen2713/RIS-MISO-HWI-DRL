import numpy as np
from sympy import sin, cos, pi

import matplotlib.pyplot as plt
from scipy.stats import vonmises_line

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
        

Ns = 16
bits = 3
n_actions = 2 ** bits
spacing_degree = 360. / n_actions
act = [i for i in range(n_actions)]
deg = [spacing_degree*i - 180. - 15. for i in range(1, n_actions + 1)]
rad = np.radians(deg).tolist()
ans = np.degrees(rad)

angle_set_deg = {
    key:val for (key, val) in zip(act, deg)
}
angle_set_rad = {
    key:val for (key, val) in zip(act, rad)
}

init_action = np.random.randint(low=0, high=n_actions, size=Ns)
Phi = np.eye(Ns, dtype=complex)
np.fill_diagonal(Phi, init_action)

log(act, bits, deg, rad, ans, angle_set_deg, angle_set_rad, np.diag(Phi), init_action)

beta_min=0.2
kappa_PDA=1.5
angle_rad = 1.57079633
beta_PDA = (1 - beta_min) * ((sin(angle_rad - 0.) + 1) / 2) ** kappa_PDA + beta_min
result = beta_PDA * (cos(angle_rad) + 1j*sin(angle_rad))
# log(beta_PDA, result)


arr1 = np.array([[2, -7, 5], [-6, 2, 0]])
arr2 = np.array([[5, 8, -5], [3, 6, 9]])
arr1_add_arr2 = np.add(arr1, arr2)
log(arr1, arr2, arr1_add_arr2)

