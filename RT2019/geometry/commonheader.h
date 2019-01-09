#pragma once

//#define USE_GLM



//#ifndef __GPU_RT__ // hide cuda functionality
//    #define __host__ {}
//    #define __device__ {}
//    #define __global__ {}
//#endif

typedef float real;

typedef unsigned int uint;
typedef unsigned long long ull;

#ifdef USE_GLM

#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
using glm::vec3;
using glm::vec2;

#else

#include "space.h"
#include <string>

typedef Vector3D<real> vec3;
typedef Vector2D<real> vec2;

//template std::string vtos<real>(const Vector3D<real>& v);

#endif

#ifdef _MSC_VER
#include "../../logger/include/logger.h"
#else
#include "logger.h"
#endif // _MSC_VER

