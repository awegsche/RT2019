#include "../Utils/spectrum.h"
#include "../Display/film.h"
#include "../geometry/bvh.h"
#include "../geometry/aabb.h"
#include "../World.h"

#include <cuda.h>
#include <cuda_runtime.h>

using namespace geometry;

const int CUDA_BLOCKW = 16;
const int STACK_SIZE = 64;
const int EntrypointSentinel = 0x76543210;

bool first = true;

texture<float4, 1, cudaReadModeElementType> bvh_texture;

union fcolour {
	uchar4 components;
	float colour;
};

__device__ fcolour make_colour(const RGBColor& colour) {
	fcolour c;
	c.components = make_uchar4(
		(unsigned char)(colour.r * 255),
		(unsigned char)(colour.g * 255),
		(unsigned char)(colour.b * 255),
		255
	);
	return c;
}

__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) {
	return max(max(min(a0, a1), min(b0, b1)), max(min(c0, c1), d));
}
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) {
	return min(min(max(a0, a1), max(b0, b1)), min(max(c0, c1), d));
}

__device__ void traverseBVH(const Ray& ray, int &nodes_crossed) {
	// BVH layout Compact2 for Kepler
	int traversalStack[STACK_SIZE];

	// Live state during traversal, stored in registers.

	int		rayidx;		// not used, can be removed
	float   origx, origy, origz;    // Ray origin.
	float   dirx, diry, dirz;       // Ray direction.
	float   tmin;                   // t-value from which the ray starts. Usually 0.
	float   idirx, idiry, idirz;    // 1 / ray direction
	float   oodx, oody, oodz;       // ray origin / ray direction

	char*   stackPtr;               // Current position in traversal stack.
	int     leafAddr;               // If negative, then first postponed leaf, non-negative if no leaf (innernode).
	int     nodeAddr;
	int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
	float   hitT;                   // t-value of the closest intersection.

	int threadId1; // ipv rayidx

	// Initialize (stores local variables in registers)
	{
		// Pick ray index.

		threadId1 = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));


		// Fetch ray.

		// required when tracing ray batches
		// float4 o = rays[rayidx * 2 + 0];  
		// float4 d = rays[rayidx * 2 + 1];
		//__shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.

		origx = ray.origin.x;
		origy = ray.origin.y;
		origz = ray.origin.z;
		dirx = ray.direction.x;
		diry = ray.direction.y;
		dirz = ray.direction.z;
		tmin = 1.0e12;

		// ooeps is very small number, used instead of raydir xyz component when that component is near zero
		float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number
		idirx = 1.0f / (fabsf(dirx) > ooeps ? dirx : copysignf(ooeps, dirx)); // inverse ray direction
		idiry = 1.0f / (fabsf(diry) > ooeps ? diry : copysignf(ooeps, diry)); // inverse ray direction
		idirz = 1.0f / (fabsf(dirz) > ooeps ? dirz : copysignf(ooeps, dirz)); // inverse ray direction
		oodx = origx * idirx;  // ray origin / ray direction
		oody = origy * idiry;  // ray origin / ray direction
		oodz = origz * idirz;  // ray origin / ray direction

		// Setup traversal + initialisation

		traversalStack[0] = EntrypointSentinel; // Bottom-most entry. 0x76543210 (1985229328 in decimal)
		stackPtr = (char*)&traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
		leafAddr = 0;   // No postponed leaf.
		nodeAddr = 0;   // Start from the root.
		hitIndex = -1;  // No triangle intersected so far.
		hitT = 1.0e12;
	}

	// Traversal loop.

	while (nodeAddr != EntrypointSentinel)
	{
		// Traverse internal nodes until all SIMD lanes have found a leaf.

		bool searchingLeaf = true; // required for warp efficiency
		while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
		{
			// Fetch AABBs of the two child nodes.

			// nodeAddr is an offset in number of bytes (char) in gpuNodes array

			float4 bbx0y0z0x1 = tex1Dfetch(bvh_texture, nodeAddr); // (bb.x0, bb.y0, bb.y0, bb.x1)
			float4 bby1z1leftright = tex1Dfetch(bvh_texture, nodeAddr + 1); // (bb.y1, bb.z1, left, right)
			float4 minIdparent = tex1Dfetch(bvh_texture, nodeAddr + 2); // (minId, parent, --, --)
			// (childindex = size of array during building, see CudaBVH.cpp)

			// compute ray intersections with BVH node bounding box

			/// RAY BOX INTERSECTION
			// Intersect the ray against the child nodes.

			float c0lox = bbx0y0z0x1.x * idirx - oodx; // n0xy.x = c0.lo.x, child 0 minbound x
			float c0hix = bbx0y0z0x1.w * idirx - oodx; // n0xy.y = c0.hi.x, child 0 maxbound x
			float c0loy = bbx0y0z0x1.y * idiry - oody; // n0xy.z = c0.lo.y, child 0 minbound y
			float c0hiy = bby1z1leftright.x * idiry - oody; // n0xy.w = c0.hi.y, child 0 maxbound y
			float c0loz = bbx0y0z0x1.z * idirz - oodz; // nz.x   = c0.lo.z, child 0 minbound z
			float c0hiz = bby1z1leftright.y * idirz - oodz; // nz.y   = c0.hi.z, child 0 maxbound z
			float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin); // Tesla does max4(min, min, min, tmin)
			float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT); // Tesla does min4(max, max, max, tmax)

			// ray box intersection boundary tests:

			float ray_tmax = 1e20;
			bool traverseChild0 = (c0min <= c0max); // && (c0min >= tmin) && (c0min <= ray_tmax);

			// didn't hit the node

			if (!traverseChild0)
			{
				nodeAddr = *(int*)stackPtr; // fetch next node by popping the stack 
				stackPtr -= 4; // popping decrements stackPtr by 4 bytes (because stackPtr is a pointer to char)   
			}

			// else hit, check if leaf
			else if (bby1z1leftright.z == 0)
			{
				nodes_crossed++;
				searchingLeaf = false; // required for warp efficiency
				leafAddr = nodeAddr;
				nodeAddr = *(int*)stackPtr;  // pops next node from stack
				stackPtr -= 4;  // decrements stackptr by 4 bytes (because stackPtr is a pointer to char)
			}

			// Otherwise, has child nodes, select the left and push right to stack
			else {
				nodeAddr = *(int*)&bby1z1leftright.z;
				stackPtr += 4;  // pushing increments stack by 4 bytes (stackPtr is a pointer to char)
				*(int*)stackPtr = *(int*)&bby1z1leftright.w;
			}

			// First leaf => postpone and continue traversal.
			// leafnodes have a negative index to distinguish them from inner nodes
			// if nodeAddr less than 0 -> nodeAddr is a leaf

			// All SIMD lanes have found a leaf => process them.

			// to increase efficiency, check if all the threads in a warp have found a leaf before proceeding to the
			// ray/triangle intersection routine
			// this bit of code requires PTX (CUDA assembly) code to work properly

			// if (!__any(searchingLeaf)) -> "__any" keyword: if none of the threads is searching a leaf, in other words
			// if all threads in the warp found a leafnode, then break from while loop and go to triangle intersection

			//if(!__any(leafAddr >= 0))     
			//    break;

			// if (!__any(searchingLeaf))
			//	break;    /// break from while loop and go to code below, processing leaf nodes

			// NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;".
			// tried everything with CUDA 4.2 but always got several redundant instructions.

			unsigned int mask; // replaces searchingLeaf

			asm("{\n"
				"   .reg .pred p;               \n"
				"setp.ge.s32        p, %1, 0;   \n"
				"vote.ballot.b32    %0,p;       \n"
				"}"
				: "=r"(mask)
				: "r"(leafAddr));

			if (!mask)
				break;
		}
	}
}

__global__ void tracingKernel(
	RGBColor *filmneg, int Height, int Width,
	BVHNode* nodes, SimpleCamera* cam
	)
{
	// assign a CUDA thread to every pixel by using the threadIndex
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	//int pixelx = threadId % scrwidth; // pixel x-coordinate on screen
	//int pixely = threadId / scrwidth; // pixel y-coordintate on screen
	int i = (Height - y - 1) * Width + x; // pixel index in buffer	
	i = y * Width + x; // pixel index in buffer	
	int pixelx = x; // pixel x-coordinate on screen
	int pixely = Height - y - 1; // pixel y-coordintate on screen


	Ray ray = cam->shoot_ray(vec2((real)x / (real)Width, (real)y / (real)Height));


	int nodes_crossed = 0;

	//traverseBVH(ray, nodes_crossed);

	float blue = min((float)nodes_crossed * 1.0e-2f, 1.0f);
	float green = 0;
	float red = 0;
	if (nodes_crossed > 100) green = min((float)nodes_crossed * 1.0e-4f, 1.0f);
	if (nodes_crossed > 10000) red = min((float)nodes_crossed * 1.0e-6f, 1.0f);

	/*
	fcolour c = make_colour(RGBColor(
		fabs(ray.direction.x),
		fabs(ray.direction.y),	
		fabs(ray.direction.z)
	));
*/
	fcolour c = make_colour(RGBColor(
		red, green, blue
	));

	filmneg[i] = RGBColor(x, y, c.colour);
//	film.pixels[i] = RGBColor(x, y, (float)x / (float)film.Height);
}

void bound_bvh_to_texture(BVHNode *nodes, int nodeSize) {
	cudaChannelFormatDesc channel3desc = cudaCreateChannelDesc<float4>();
	DBG("binding BVH to CUDA texture. Nodes: " << nodeSize << ".");
	cudaLAUNCH(cudaBindTexture(NULL, &bvh_texture, nodes, &channel3desc, nodeSize * sizeof(BVHNode)))
}

void Render(Sensor* film, World* world) {
	dim3 block(CUDA_BLOCKW, CUDA_BLOCKW, 1);   // dim3 CUDA specific syntax, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 grid(film->Width / block.x, film->Height / block.y, 1);
	BVH* world_bvh = world->single_bvh;

	if (first) {
		LOG("binding textures");
		bound_bvh_to_texture(world_bvh->nodes, world_bvh->numTriangles - 1);
		cudaLAUNCH(cudaDeviceSynchronize())
		first = false;
	}
	else return;

	tracingKernel << <grid, block >> > (
		film->negative, film->Height, film->Width,
		world_bvh->nodes, world->cam
		);
}