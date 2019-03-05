from .CPP import afm_transform_cpu
from .CUDA import afm_transform_gpu


def afm_transform(lines, height, width, device = 'cpu'):
    
    if device == 'cpu':
        return afm_transform_cpu(lines, height, width)
    elif isinstance(device, int):
        return afm_transform_gpu(lines, height, width, device)
    else:
        raise IOError('invalid device specification')

