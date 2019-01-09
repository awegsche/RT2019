#ifndef BVH_H
#define BVH_H

#include "commonheader.h"
#include "triangle.h"
#include "mesh.h"
#include "aabb.h"



namespace geometry {

	class BVH;

void cuda_construct_bvh(const BVH& bvh);  // cuda function

//bool compareTriRefs(const TriRef& a, const TriRef& b);
struct BVHNode {
    AABB boundin_box; // 6 float
	uint left;		// 1 float
	uint right;		// 1 float
	uint minId;	// 1 float
	//int maxId;
    uint parent; // 1 float
	float2 padding;

    inline bool IsLeaf() { return left == 0; }
	BVHNode() :left(0), right(0), minId(0xFFFFFFFF), /*maxId(-1),*/ parent(0), boundin_box() {}
};

class BVH
{
public:
    BVH();
	~BVH();
    BVHNode* nodes;
    BVHNode* leaves;

    uint numTriangles;
	vector<morton_int> morton_codes;
	vector<Triangle> triangles;


    void host_construct(const mesh& original_mesh);

private:
    void host_construct_radix(const mesh& original_mesh);
	void calculate_bounding_boxes_device(const mesh& original_mesh);
	void copy_to_device(const mesh& original_mesh);
	

public:  // TODO: implement getters
	BVHNode *device_nodes;
	BVHNode *device_leaves;
	Triangle* device_tris;
	vec3* device_vertices;
};

int delta(int a, int b, const vector<morton_int>& morton_table);

//AABB calculate_bb_of_node(BVHNode* node, const mesh& original_mesh);

}
#endif // BVH_H
