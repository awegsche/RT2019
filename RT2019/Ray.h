#pragma once
#include "geometry/commonheader.h"

#include <cuda_runtime.h>


class Ray
{
public:
	vec3 origin;
	vec3 direction;

public:
	__host__ __device__ Ray() : origin(), direction(0, 0, -1.0) {}
	__host__ __device__ Ray(const vec3& o, const vec3& d) : origin(o), direction(d) {}

	__host__ __device__ ~Ray() {}
};

