import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "lsgToAFM.hpp":
    void _lsgToAFM(int , int ,
                     int , const float* ,
                     int , int , float* , int*)

def lsg2afm(np.ndarray[float, ndim=2] lines,
               np.ndarray[np.int32_t, ndim=1] input_size,
               np.ndarray[np.int32_t, ndim=1] output_size):
    cdef int input_H = input_size[0]
    cdef int input_W = input_size[1]
    cdef int output_H = output_size[0]
    cdef int output_W = output_size[1]

    cdef int lsg_num = lines.shape[0]

    cdef np.ndarray[float, ndim=3] \
        afmap = np.zeros((2,output_H, output_W), dtype=np.float32)

    cdef np.ndarray[int, ndim=2] \
        label = np.zeros((output_H, output_W), dtype=np.int32)

    _lsgToAFM(input_H, input_W, lsg_num, <const float*> lines.data,
                    output_H, output_W, <float*> afmap.data, <int*> label.data)

    return afmap, label