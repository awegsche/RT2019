/**@file	/media/awegsche/HDD/files/RT2019/test4linux/cuda_runtime.h
 * @author	awegsche
 * @version	801
 * @date
 * 	Created:	10th Jan 2019
 * 	Last Update:	10th Jan 2019
 */

#ifndef CUDART
#define CUDART

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

cudaError_t cudaMalloc(void** dest, unsigned long size, void* src);


#endif // CUDART
