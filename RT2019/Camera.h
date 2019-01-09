#pragma once

#include "Ray.h"

class SimpleCamera
{
public:
	vec3 up;
	vec3 lookat;
	vec3 position;

	real distance;
	SimpleCamera() : up(0, 1.0074, 0), lookat(), position(0, 0, -100), distance(1) {}
	~SimpleCamera(){}

	__host__ __device__ inline Ray shoot_ray(const vec2& sample) {
		
		vec3 v = (lookat - position).hat();
		vec3 w = up ^ v;
		vec3 u = w ^ v;

		return Ray(position, (v * distance + w * sample.x + u * sample.y).hat());
	}
};

