import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.device import Device

def list_available_devices():
    available = []
    test_arr = np.ones((1,1), np.float32)
    devices = [d for d in Device._devices if d != "WEBGPU"]
    for dev in devices:
        try:
            _ = Tensor(test_arr, device=dev).realize()  #try to allocate + realize
            available.append(dev)
        except Exception as e:
            # print(f"{dev} failed: {e}")   #
            pass
    # "CPU" is an alias to CLANG 
    if "CLANG" in available and "CPU" not in available:
        available.append("CPU")
    return available
