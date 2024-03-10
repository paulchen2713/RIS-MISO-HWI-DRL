import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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



# NOTE The UPA mode only works with perfect square numbers; otherwise, 
#      it is rounded down to the nearest perfect square number.

def np_array_response(angle1, angle2, num, antenna_array):
    """
    generate ULA and UPA steering vectors
    """
    assert num > 0
    
    y = np.zeros((num, 1), dtype=np.complex64)
    
    if antenna_array == 'UPA':
        num_sqrt = int(np.sqrt(num))
        assert num_sqrt * num_sqrt == int(num)

        for m in range(int(np.sqrt(num))):
            for n in range(int(np.sqrt(num))):
                y[m * (int(np.sqrt(num))) + n] = np.exp(1j*np.pi*(m*np.sin(angle1)*np.cos(angle2) + n*np.cos(angle2)))
    elif antenna_array == 'ULA':
        for n in range(num):
            y[n] = np.exp(1j*np.pi*(n*np.sin(angle1)))
   
    y = y / np.sqrt(num)
    return y


def torch_array_response(angle1, angle2, num, antenna_array):
    """
    generate ULA and UPA steering vectors
    """
    assert num > 0

    y = torch.zeros((num, 1), dtype=torch.complex64).to(device)

    if antenna_array == 'UPA':
        num_sqrt = int(torch.sqrt(num))
        assert num_sqrt * num_sqrt == int(num)

        for m in range(num_sqrt):
            for n in range(num_sqrt):
                y[m * num_sqrt + n] = torch.exp(1j*np.pi*(m*torch.sin(angle1).to(device)*torch.cos(angle2).to(device) 
                                                          + n*torch.cos(angle2).to(device))).to(device)
    elif antenna_array == 'ULA':
        for n in range(num):
            y[n] = torch.exp(1j*np.pi*(n*torch.sin(angle1).to(device))).to(device)
   
    y = y / torch.sqrt(num).to(device)
    return y



def np_ULA_response(angle: np.float32, num_antennas: np.uint8) -> np.ndarray:
    """
    Return the ULA steering vector

    Keyword arguments:
    angle:         the angles of arrival(AoA) or angle of departure (AoD) in radian
    num_antennas:  the number of Tx or Rx antennas
    """
    assert num_antennas > 0
    
    array_response = np.zeros((num_antennas, 1), dtype=np.complex64)
    
    for n in range(0, num_antennas):
        array_response[n] = np.exp(1j*np.pi*(n*np.sin(angle)))
    
    array_response = array_response / np.sqrt(num_antennas)
    return array_response


def np_UPA_response(azimuth: np.float32, elevation: np.float32, M_y: np.uint8, M_z: np.uint8) -> np.ndarray:
    """
    Return the UPA steering vector

    Keyword arguments:
    azimuth:    the azimuth AoA or AoD in radian
    elevation:  the elevation AoA or AoD in radian
    M_y:        the number horizontal antennas of Tx or Rx 
    M_z:        the number vertical antennas of Tx or Rx 
    """
    assert M_y > 0 and M_z > 0

    array_response = np.zeros((M_y * M_z, 1), dtype=np.complex64)
    
    for m in range(M_y):
        for n in range(M_z):
            array_response[m * M_z + n] = np.exp(1j*np.pi*(m*np.sin(azimuth)*np.cos(elevation) + n*np.cos(elevation)))

    array_response = array_response / np.sqrt(M_y * M_z)
    return array_response


