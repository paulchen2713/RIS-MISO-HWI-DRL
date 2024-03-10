import torch
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def retrieve_name(var):
    import inspect
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def log(*argv):
    # import torch
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

def array_response(angle1, angle2, num, antenna_array):
    """
    generate ULA and UPA steering vector
    """
    y = np.zeros((num, 1), dtype=complex)
    print(f"original shape: {y.shape}")

    if antenna_array == 'USPA':
        for m in range(int(math.sqrt(num))):
            for n in range(int(math.sqrt(num))):
                y[m * (int(math.sqrt(num))) + n] = np.exp(1j*math.pi*(m*math.sin(angle1)*math.cos(angle2) + n*math.cos(angle2)))
    elif antenna_array == 'ULA':
        for n in range(num):
            y[n] = np.exp(1j*math.pi*(n*math.sin(angle1)))
   
    y = y / math.sqrt(num)
    return y


def array_response_np(angle1, angle2, num, antenna_array):
    """
    generate ULA and UPA steering vector
    """
    y = np.zeros((num, 1), dtype=complex)
    
    if antenna_array == 'USPA':
        for m in range(int(np.sqrt(num))):
            for n in range(int(np.sqrt(num))):
                y[m * (int(np.sqrt(num))) + n] = np.exp(1j*np.pi*(m*math.sin(angle1)*np.cos(angle2) + n*np.cos(angle2)))
    elif antenna_array == 'ULA':
        for n in range(num):
            y[n] = np.exp(1j*np.pi*(n*np.sin(angle1)))
   
    y = y / np.sqrt(num)
    return y


def array_response_torch(angle1, angle2, num, antenna_array):
    """
    generate ULA and UPA steering vector
    """
    y = torch.zeros((num, 1), dtype=torch.complex64).to(device)

    if antenna_array == 'USPA':
        for m in range(int(torch.sqrt(num))):
            for n in range(int(torch.sqrt(num))):
                y[m * (int(torch.sqrt(num))) + n] = torch.exp(1j*np.pi*(m*torch.sin(angle1).to(device)*torch.cos(angle2).to(device) 
                                                                        + n*torch.cos(angle2).to(device))).to(device)
    elif antenna_array == 'ULA':
        for n in range(num):
            y[n] = torch.exp(1j*np.pi*(n*torch.sin(angle1).to(device))).to(device)
   
    y = y / torch.sqrt(num).to(device)
    return y


if __name__ == '__main__':
    y = array_response(2, 3, 4, 'USPA')
    log(y)

    y = array_response_np(2, 3, 4, 'USPA')
    log(y)
    
    y = array_response_torch(torch.tensor(2, dtype=torch.int8), 
                             torch.tensor(3, dtype=torch.int8), 
                             torch.tensor(4, dtype=torch.int8), 
                             'USPA')
    log(y)

    x = np.random.uniform(-1, 1, (1, 5)) + 1j * np.random.uniform(-1, 1, (1, 5)) 
    log(x)
    
