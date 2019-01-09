#ifndef AABB_H
#define AABB_H

#include "commonheader.h"

#include "morton.h"

#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

using std::min;
using std::max;


namespace geometry {

class AABB
{
private:
   vec3 p0, p1;
public:
	__host__ __device__ AABB()
		: p0(), p1() {}
    __host__ __device__ AABB(const vec3& p0_, const vec3& p1_)
		: p0(p0_), p1(p1_) {}
    __host__ __device__ AABB(real p0x, real p0y, real p0z, real p1x, real p1y, real p1z)
		: p0(p0x, p0y, p0z), p1(p1x, p1y, p1z) {}

	unsigned int morton_code() const {
		float fact = 0.5f;
		return morton3D(fact * (p0.x + p0.y), fact * (p0.y + p1.y), fact * (p1.z + p1.z));
	}

    ///
    /// \brief grows the bounding box to include the point
    /// \param point
    ///
	void grow(const vec3& point)
	{
		p0.x = min(p0.x, point.x);
		p0.y = min(p0.y, point.y);
		p0.z = min(p0.z, point.z);
		p1.x = max(p1.x, point.x);
		p1.y = max(p1.y, point.y);
		p1.z = max(p1.z, point.z);
	}

	//AABB unite(const AABB& box);

	__host__ __device__ const vec3& operator[](int i) const { return (i == 0) ? p0 : p1; }
    __host__ __device__ vec3& operator[](int i) { return (i == 0) ? p0 : p1; }

	__host__ __device__ vec3 sides() const {

		return p1 - p0;
	}


    friend std::ostream& operator<<(std::ostream& os, const AABB& box);

	static __host__ __device__ AABB unite(const AABB& a, const AABB& b);

};


std::ostream& operator<<(std::ostream& os, const AABB& box);

}
#endif // AABB_H
