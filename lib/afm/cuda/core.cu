#include "afm.hpp"
#include <vector>
#include <iostream>
#include <stdio.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x*gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int CUDA_GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


void _set_device(int device_id) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (current_device == device_id)
        return;
    CUDA_CHECK(cudaSetDevice(device_id));
}

__global__ void AfmKernel(const int nthreads, const int n_lines, const int height, 
    const int width,  const float* lines, float* afm, int* label)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    // CUDA_KERNEL_LOOP(index, nthreads)
    if(index < nthreads)
    {
        const int w = index % width;
        const int h = index / width;        
        // printf("%d,", index);
        float min_dis = 1e30;
        float ax_opt = 0;
        float ay_opt = 0;
        float ind_opt = 0;
        float px = (float) w;
        float py = (float) h;
        for(int i = 0; i < n_lines; ++i) {
            float dx = lines[4*i+2] - lines[4*i];
            float dy = lines[4*i+3] - lines[4*i+1];
            float norm2 = dx*dx + dy*dy;

            // float t = (((float)w-lines[4*i])*dx + ((float)h-lines[4*i+1])*dy)/(norm2+1e-6);
            float t = ((px-lines[4*i])*dx + (py-lines[4*i+1])*dy)/(norm2+1e-6);

            t = t<1.0?t:1.0;
            t = t>0.0?t:0.0;


            float ax = lines[4*i]   + t*dx - px;
            float ay = lines[4*i+1] + t*dy - py;
            
            // printf("%f, %f, %f, %f\n",lines[4*i], lines[4*i+1], lines[4*i+2],lines[4*i+3]);
            float dis = ax*ax + ay*ay;

            if (dis < min_dis) {
                min_dis = dis;
                ax_opt = ax;
                ay_opt = ay;
                ind_opt = i;
            }
        }
        afm[h*width + w] = ax_opt;
        afm[h*width + w + height*width] = ay_opt;
        label[h*width+w] = ind_opt;
    }
    // __syncthreads();
    
}

void _AttractionFieldMap(int n_lines, const float* lines, 
                         int height, int width, float* afm, 
                         int* label, int device_id)
{
    _set_device(device_id);
    float* lines_dev = NULL;
    float* afm_dev = NULL;
    int* label_dev = NULL;
    CUDA_CHECK(cudaMalloc(&lines_dev, 4*n_lines*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&afm_dev, 2*height*width*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&label_dev, height*width*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(&lines_dev[0], lines, 4*n_lines*sizeof(float), cudaMemcpyHostToDevice));

    int nthreads = height*width;

    AfmKernel<<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>> (nthreads, n_lines, height, width, lines_dev, 
    afm_dev, label_dev);
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    // cudaDeviceSynchronize();
    
    CUDA_CHECK(cudaMemcpy(&afm[0], afm_dev, 2*sizeof(float)*height*width, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(&label[0], label_dev, sizeof(int)*height*width, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(lines_dev));
    CUDA_CHECK(cudaFree(afm_dev));
    CUDA_CHECK(cudaFree(label_dev));
}