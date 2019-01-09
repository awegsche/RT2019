#include <cuda_runtime.h>
#include "bvh.h"
#include "triangle.h"
#include "commonheader.h"
#include "../Utils/cuda_helpers.h" 

#include <thrust/device_vector.h>

//#if __CUDA_ARCH__ < 600
//__device__ double atomicAdd(double* address, double val)
//{
//    unsigned long long int* address_as_ull =
//                              (unsigned long long int*)address;
//    unsigned long long int old = *address_as_ull, assumed;
//
//    do {
//        assumed = old;
//        old = atomicCAS(address_as_ull, assumed,
//                        __double_as_longlong(val +
//                               __longlong_as_double(assumed)));
//
//    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//    } while (assumed != old);
//
//    return __longlong_as_double(old);
//}
//#endif

namespace geometry {

	const int BlockSize = 256;

	__host__ __device__ AABB tri_bounding_box(const Triangle &tri, const vec3* vertices) {
		return AABB(
			min(min(vertices[tri.v1].x, vertices[tri.v2].x), vertices[tri.v3].x),
			min(min(vertices[tri.v1].y, vertices[tri.v2].y), vertices[tri.v3].y),
			min(min(vertices[tri.v1].z, vertices[tri.v2].z), vertices[tri.v3].z),

			max(max(vertices[tri.v1].x, vertices[tri.v2].x), vertices[tri.v3].x),
			max(max(vertices[tri.v1].y, vertices[tri.v2].y), vertices[tri.v3].y),
			max(max(vertices[tri.v1].z, vertices[tri.v2].z), vertices[tri.v3].z)
		);
	}

	AABB AABB::unite(const AABB & a, const AABB & b)
	{
		return AABB(
			vec3(min(a.p0.x, b.p0.x), min(a.p0.y, b.p0.y), min(a.p0.z, b.p0.z)),
			vec3(max(a.p1.x, b.p1.x), max(a.p1.y, b.p1.y), max(a.p1.z, b.p1.z))
		);
	}


	__global__ void construct_bvh(BVHNode *nodes, BVHNode* leaves,
		Triangle* tris, vec3* vertices, int numtriangles, int* nodeCounter)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i < numtriangles) {
			BVHNode* leaf = leaves + i;

			// Handle leaf first
			leaf->minId = i;
			//printf("%d, %d\n", leaf->minId, (leaves + i)->minId);
			leaf->boundin_box = tri_bounding_box(tris[i], vertices);

			uint current = leaf->parent;

			int res = atomicAdd(nodeCounter + current, 1);
		printf("%d\n", i);

			// Go up and handle internal nodes
			while (true) {
				if (res == 0) {
					return;
				}
				BVHNode* currentNode = nodes + current;
				AABB leftBoundingBox = nodes[currentNode->left].boundin_box;
				AABB rightBoundingBox = nodes[currentNode->right].boundin_box;

				// Compute current bounding box
				currentNode->boundin_box = AABB::unite(leftBoundingBox,
											  rightBoundingBox);
				vec3 sides = currentNode->boundin_box.sides();
				printf("%f, %f, %f\n", sides.x, sides.y, sides.z);

				// If current is root, return
				if (current == 0) {
					return;
				}
				current = currentNode->parent;
				res = atomicAdd(nodeCounter + current, 1);
			}
		}
	}

	void cuda_construct_bvh(const BVH& bvh)
	{
		int blockSize = BlockSize;
		int gridSize = (bvh.numTriangles + blockSize - 1) / blockSize;
		thrust::device_vector<int> nodeCounter(bvh.numTriangles);
		construct_bvh << < 1, bvh.numTriangles >> > (bvh.device_nodes, bvh.device_leaves,
			bvh.device_tris, bvh.device_vertices, bvh.numTriangles, nodeCounter.data().get());

		cudaLAUNCH(cudaDeviceSynchronize())
	}


}
