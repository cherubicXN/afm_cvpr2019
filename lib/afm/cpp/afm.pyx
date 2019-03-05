import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "afm.hpp":
    void _AttractionFieldMap(int n_lines, const float* lines,
                         int height, int width, float* afm,int* label);

def afm_transform_cpu(np.ndarray[np.float32_t, ndim=2] lines, int height, int width):
    cdef int n_lines = lines.shape[0]

    cdef np.ndarray[np.float32_t, ndim=3] \
        afmap = np.zeros((2, height, width), dtype=np.float32)
    
    cdef np.ndarray[np.int32_t, ndim=2] \
        aflabel = np.zeros((height, width), dtype=np.int32)

    _AttractionFieldMap(n_lines, <const float*> lines.data,                    height, width, <float*> afmap.data,                    <int*> aflabel.data);

    return afmap, aflabel;
