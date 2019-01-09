#pragma once
#include <cuda_runtime.h>
#include <logger.h>

#include <helper_cuda.h>

extern cudaError_t __err__;

#define cudaLAUNCH(func_)                               \
	__err__ = func_;                            \
	if (__err__ != cudaError::cudaSuccess)                  \
		ERR(_cudaGetErrorEnum(__err__) << " occured.");
