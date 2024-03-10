# Source: 
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises.html
#   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises_line.html
# Reference: https://en.wikipedia.org/wiki/Von_Mises_distribution

# vonmises() is a circular distribution which does not restrict the distribution to a fixed interval.
# vonmises_line() is the same distribution, defined on [-\pi, \pi].

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises, vonmises_line

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


loc = 0.43*np.pi  # circular mean
kappa = 1.5       # concentration 

print(f"one random 'vonmises' sample: {vonmises.pdf(loc, kappa, 0)}") 
print(f"one random 'vonmises_line' sample: {vonmises_line.pdf(loc, kappa, 0)}") 

number_of_samples = 1000
samples = vonmises_line(loc=loc, kappa=kappa).rvs(number_of_samples)
log(samples[:10])

fig = plt.figure(figsize=(12, 6))
left  = plt.subplot(121)
right = plt.subplot(122, projection='polar')
x = np.linspace(-np.pi, np.pi, 500)

vonmises_pdf = vonmises_line.pdf(loc, kappa, x)

ticks = [0, 0.15, 0.3]

left.plot(x, vonmises_pdf)
left.set_yticks(ticks)
number_of_bins = int(np.sqrt(number_of_samples))
left.hist(samples, density=True, bins=number_of_bins)
left.set_title("Cartesian plot")
left.set_xlim(-np.pi, np.pi)
left.grid(True)

right.plot(x, vonmises_pdf, label="PDF")
right.set_yticks(ticks)
right.hist(samples, density=True, bins=number_of_bins, label="Histogram")
right.set_title("Polar plot")
right.legend(bbox_to_anchor=(0.15, 1.06))

plt.show()

