#include "bvh.h"

#include <vector>
#include <algorithm>
#include <boost/compute/algorithm/sort_by_key.hpp>
//#include <thrust/sort.h>

#include <cuda_runtime.h>
#include "../Utils/cuda_helpers.h"

using namespace std;

#ifdef _MSC_VER
#define __clz(x) __lzcnt(x)
#else
#define __clz(x) __builtin_clz(x)
#endif


geometry::BVH::BVH()
	: numTriangles(0), nodes(nullptr), leaves(nullptr),
	device_leaves(nullptr), device_nodes(nullptr), device_vertices(nullptr), device_tris(nullptr)
{

}

geometry::BVH::~BVH()
{
	cudaFree(device_leaves);
	cudaFree(device_nodes);
	cudaFree(device_tris);
	cudaFree(device_vertices);
}

void geometry::BVH::host_construct(const mesh & original_mesh)
{
	host_construct_radix(original_mesh);
	copy_to_device(original_mesh);

	cuda_construct_bvh(*this);
}

void geometry::BVH::host_construct_radix(const mesh &original_mesh)
{
	numTriangles = original_mesh.faces.size();
    leaves = new BVHNode[original_mesh.faces.size()];
    nodes = new BVHNode[original_mesh.faces.size() - 1];
	morton_codes.reserve(numTriangles);
	copy(original_mesh.faces.begin(), original_mesh.faces.end(), back_inserter(triangles));


    for (auto tri : triangles) {
		morton_codes.push_back(morton(tri, original_mesh.vertices));
    }
    boost::compute::sort_by_key(morton_codes.begin(), morton_codes.end(), triangles.begin());

	int n = morton_codes.size();
	leaves[n - 1].minId = n - 1;

    for (int i = 0; i < n - 1; i++) {
		leaves[i].minId = i;
        int d = (delta(i, i+ 1, morton_codes) - delta(i, i-1, morton_codes)) > 0 ? 1 : -1;

        int deltamin = delta(i, i - d, morton_codes);
        int lmax = 2;
        while (delta(i, i+lmax*d, morton_codes) < deltamin) {
            lmax = lmax * 2;
        }
        int l = 0;
        for(int t = lmax / 2; t > 0; t /= 2) {
            if (delta(i, i+(l+t)*d, morton_codes) > deltamin)
                l += t;
            t /= 2;
        }
        int j = i + l * d;
        int deltanode = delta(i, j, morton_codes);
        int s = 0;

        for(int t = lmax / 2; t > 0; t /= 2) {
            if (delta(i, i + (s + t) * d, morton_codes) > deltanode) {
                s += t;
            }
        }
        int gamma = i + s * d + min(d, 0);

        BVHNode* current = nodes + i;

        if(min(i,j) == gamma) {
            current->left = gamma;
            (leaves + gamma)->parent = current - nodes;
        }
        else {
            current->left = gamma;
            (nodes + gamma)->parent = current - nodes;
        }

        if(min(i,j) == gamma + 1) {
            current->left = gamma + 1;
            (leaves + gamma + 1)->parent = current - nodes;
        }
        else {
            current->left = gamma + 1;
            (nodes + gamma + 1)->parent = current - nodes;
        }

		current->minId = min(i, j);
    }

}

void geometry::BVH::calculate_bounding_boxes_device(const mesh & original_mesh)
{
}

//void geometry::BVH::calculate_bounding_boxes(const mesh & original_mesh)
//{
//	calculate_bb_of_node(nodes, original_mesh);
//}

void geometry::BVH::copy_to_device(const mesh& original_mesh)
{
	cudaLAUNCH(cudaDeviceSynchronize())

	cudaLAUNCH(cudaMalloc(&device_nodes, sizeof(BVHNode) * (numTriangles - 1)));
	cudaLAUNCH(cudaMalloc(&device_leaves, sizeof(BVHNode) * numTriangles));
	cudaLAUNCH(cudaMalloc(&device_tris, sizeof(Triangle) * (numTriangles)));
	cudaLAUNCH(cudaMalloc(&device_vertices, sizeof(vec3) * original_mesh.vertices.size()));

	cudaLAUNCH(cudaMemcpy(device_nodes, nodes, sizeof(BVHNode) * (numTriangles - 1), cudaMemcpyHostToDevice))
	cudaLAUNCH(cudaMemcpy(device_leaves, leaves, sizeof(BVHNode) * numTriangles, cudaMemcpyHostToDevice))
	cudaLAUNCH(cudaMemcpy(device_tris, triangles.data(), sizeof(Triangle) * numTriangles, cudaMemcpyHostToDevice))
	cudaLAUNCH(cudaMemcpy(device_vertices, original_mesh.vertices.data(), sizeof(vec3) * original_mesh.vertices.size(), cudaMemcpyHostToDevice))
	
}


int geometry::delta(int a, int b, const vector<morton_int> &morton_table)
{

   return (b < 0 || b >= morton_table.size()) ? -1 :
                               __clz(morton_table[a] ^ morton_table[b]);
}

/*a
geometry::AABB geometry::calculate_bb_of_node(BVHNode * node, const mesh& original_mesh)
{
	if (node->IsLeaf()) {
		geometry::AABB box(
			original_mesh.vertices[original_mesh.faces[node->minId].v1],
			original_mesh.vertices[original_mesh.faces[node->minId].v2]
		);
		box.grow(
			original_mesh.vertices[original_mesh.faces[node->minId].v3]
		);
		node->boundin_box = box;
	}
	else {
		auto boxA = calculate_bb_of_node(node->left, original_mesh);
		auto boxB = calculate_bb_of_node(node->right, original_mesh);
		node->boundin_box = AABB::unite(boxA, boxB);
	}
	return node->boundin_box;
}

*/
//bool geometry::compareTriRefs(const TriRef& a, const TriRef& b) {
//    return a.morton_code < b.morton_code;
//}
