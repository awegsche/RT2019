/**@file	/media/awegsche/HDD/files/RT2019/test4linux/cuda_runtime.h
 * @author	awegsche
 * @version	801
 * @date
 * 	Created:	10th Jan 2019
 * 	Last Update:	10th Jan 2019
 */

#ifndef CUDART
#define CUDART

#include "logger.h"
#include <cstdlib>

#define __host__ /**/
#define __device__ /**/
#define __global__ /**/

struct float2 {
    float x;
    float y;
};

typedef unsigned int cudaError_t;

enum cudaError
{
    cudaSuccess = 0
};

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice
};

template<class T>
cudaError_t cudaMalloc(T** dest, unsigned long size)
{
    DBG("imitating cudaMalloc, allocating " << size << " bytes on host.");
    *dest = malloc(size);
    return cudaError::cudaSuccess;
}

cudaError_t cudaFree(void* ptr);

cudaError_t cudaDeviceSynchronize() { return cudaError::cudaSuccess; }

template<class T>
cudaError_t cudaMemcpy(T*& dst, const T* src, size_t count, cudaMemcpyKind kind)
{
    DBG("imitating cudaMemcpy, dublicating raw pointer, size of data: " << count << " bytes.");
    dst = src;
    return cudaError::cudaSuccess;
}


#endif // CUDART
