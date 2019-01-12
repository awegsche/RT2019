
#include "Display/displaywindow.h"
#include "Display/film.h"
// ---- CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// ---- OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_gl_interop.h>
#include "Utils/cuda_helpers.h"

// ---- geometry
#include "geometry/ascii_ply.h"
#include "geometry/bvh.h"
#include "World.h"

// ---- std stuff
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdio.h>

// ---- logging
#include "logger.h"

// ---- testing
#include "tests/load_bunny.h"

using namespace std;
Tracer* tracer_ptr;
const int build = 2;

const int PropKeyLength = 32;

int main()
{
	logging::add_console_logger();
	cout << "\33[1m"
		<< "----------------------------------------------------------------------------" << endl
		<< "------------------------ CUDA Raytracer 2019 -------------------------------" << endl
		<< "----------------------------------------------------------------------------" << endl
		<< "------------------------ build: " << setw(5) << build << " --------------------------------------" << endl
		<< "----------------------------------------------------------------------------" << endl;

	LOG("LOGGING activated");
	DBG("DEBUG output");
	ERR("Test ERROR");
	cudaError_t err = cudaError::cudaSuccess;

	if (cudaError::cudaSuccess != (err = cudaSetDevice(0))) {
		ERR("Setting CUDA device for computation failed.");
		return 1;
	}

	DBG("size of RGBColor = " << sizeof(RGBColor));
	cudaDeviceProp dev_props;
	
	if (cudaError::cudaSuccess != (err = cudaGetDeviceProperties(&dev_props, 0))) {
		ERR("Coudn't retrieve CUDA Device Properties");
	}
	else {
		LOG("CUDA device selected. Properties:");
		LOG(left << setw(PropKeyLength) << "  Name: " << right << dev_props.name);
		LOG(left << setw(PropKeyLength) << "  Global Memory: " << right << dev_props.totalGlobalMem / 1048576 << " MB");
		LOG(left << setw(PropKeyLength) << "  Shared Memory per Block: " << right << dev_props.sharedMemPerBlock / 1024 << " KB");
		LOG(left << setw(PropKeyLength) << "  Compute Capability: " << right
			<< "\33[1m" << dev_props.major << "." << dev_props.minor << "\33[22m");
		LOG(left << setw(PropKeyLength) << "  Multiprocessor Count:" << right << dev_props.multiProcessorCount);
		LOG(left << setw(PropKeyLength) << "  max Threads per Block: " << right << dev_props.maxThreadsPerBlock);
		LOG(left << setw(PropKeyLength) << "  max Threads Dimension: " << right
			<< dev_props.maxThreadsDim[0] << "/" << dev_props.maxThreadsDim[1] << "/" << dev_props.maxThreadsDim[2]);
		LOG(left << setw(PropKeyLength) << "  max Threads per Multiprocessor: " << right << dev_props.maxThreadsPerMultiProcessor);
		LOG(left << setw(PropKeyLength) << "  Single/Double PerfRatio: " << right << dev_props.singleToDoublePrecisionPerfRatio);
		LOG(left << setw(PropKeyLength) << "  Clock Rate: " << right << dev_props.clockRate);

	}

	World w;
	w.single_bvh = &load_and_test_simple();
	w.set_cam(new SimpleCamera());

	//if (cudaError::cudaSuccess != (err = cudaGLSetGLDevice(0))) {
	//	ERR("Setting CUDA device for GL failed.");
	//	return 1;
	//}
	if (0 != init_window())
	{
		ERR("initialisation failed.");
		return 1;
	}
	tracer_ptr = new Tracer();
	tracer_ptr->set_world(&w);
	init_tracer(tracer_ptr);

	glutMainLoop();

    return 0;
}

