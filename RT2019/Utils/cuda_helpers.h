#pragma once
#include <cuda_runtime.h>
#include <logger.h>

#include <helper_cuda.h>

extern cudaError_t __err__;

#ifndef __CPU_ONLY__
#define cudaLAUNCH(func_)                               \
	__err__ = func_;                            \
	if (__err__ != cudaError::cudaSuccess)                  \
		ERR(_cudaGetErrorEnum(__err__) << " occured.");
#else
#define cudaLAUNCH(func_) \
    __err__ = func_; \
	if (__err__ != cudaError::cudaSuccess)                  \
		ERR(__err__ << " occured.");
#endif
