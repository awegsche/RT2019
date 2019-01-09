#pragma once
#include "../Utils/spectrum.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

struct Sensor {
	spectrum *pixels;
	RGBColor *negative;
	int Width;
	int Height;

	__host__ Sensor(int width, int height) 
	: Width(width), Height(height) {
	}
	__host__ Sensor() 
	: Width(100), Height(100) {
	}

	__host__ void init() {
		cudaMalloc(&pixels, Width * Height * sizeof(spectrum));
		cudaMalloc(&negative, Width * Height * sizeof(spectrum));
//		cudaMalloc(&cuda_ptr, sizeof(Sensor));
//		cudaMemcpy(cuda_ptr, this, sizeof(Sensor), cudaMemcpyHostToDevice);
	}

	__host__ ~Sensor() {
		cudaFree(pixels);
		cudaFree(negative);
	}

	__host__ void map(GLuint vbo) {
		cudaGLMapBufferObject((void**)&negative, vbo);
	}
};
