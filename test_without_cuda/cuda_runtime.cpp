/**@file	/media/awegsche/HDD/files/RT2019/test4linux/cuda_runtime.cpp
 * @author	awegsche
 * @version	801
 * @date
 * 	Created:	11th Jan 2019
 * 	Last Update:	11th Jan 2019
 */

#include "cuda_runtime.h"


cudaError_t cudaMalloc(void** dest, unsigned long size, void* src)
{
    *dest = src;
    return cudaError::cudaSuccess;
}

