import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "squeeze.hpp":
    void _region_grow(int , int ,
                const float* , const float* , const float* ,
                int, float*, int*)

def region_grow(np.ndarray[float,ndim=1] arr_x,
                np.ndarray[float,ndim=1] arr_y,
                np.ndarray[float,ndim=1] arr_ang,
                np.ndarray[np.int32_t, ndim=1] im_size):

    cdef int H = im_size[0]
    cdef int W = im_size[1]

    cdef int cnt = arr_x.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] \
        rectangles = np.zeros((cnt,5),dtype=np.float32)
    cdef int num_out

    _region_grow(H, W, <const float*> arr_x.data,
                 <const float*> arr_y.data,
                 <const float*> arr_ang.data, cnt,
                 <float*> rectangles.data, &num_out)

    rectangles = rectangles[:num_out,:]

    return rectangles


