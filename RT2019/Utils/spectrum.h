#pragma once

#include "cuda_helpers.h"


struct RGBColor {
	float r, g, b;

	__host__ __device__ RGBColor() : r(0.0f), g(0.0f), b(0.0f) {}
	__host__ __device__ RGBColor(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}

	__host__ __device__ __inline__ RGBColor operator+ (const RGBColor& c_) { return RGBColor(r + c_.r, g + c_.g, b + c_.b); }
	__host__ __device__ __inline__ RGBColor operator- (const RGBColor& c_) { return RGBColor(r - c_.r, g - c_.g, b - c_.b); }
	__host__ __device__ __inline__ RGBColor operator* (float a_) { return RGBColor(r * a_, g * a_, b * a_); }
	__host__ __device__ __inline__ RGBColor operator/ (float a_) { return RGBColor(r / a_, g / a_, b / a_); }
};


typedef RGBColor spectrum;
