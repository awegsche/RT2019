#pragma once
#include "commonheader.h"
#include "morton.h"

#include <iostream>
#include <vector>
#include <string>

#include <cuda_runtime.h>

using namespace std;

namespace geometry
{

struct Triangle 
{
    uint v1, v2, v3;
    __host__ __device__ Triangle(int a, int b, int c) : v1(a), v2(b), v3(c) {}

    string to_string(const vector<vec3>& vertices);
};

unsigned long long morton(const Triangle& tri, const vector<vec3>& vertices);

ostream& operator<<(ostream& os, const Triangle& tri) ;
}
