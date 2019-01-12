#pragma once
#include "../geometry/ascii_ply.h"
#include "../geometry/bvh.h"
#include "../geometry/mesh.h"
#include <memory>


using namespace geometry;

BVH load_and_test_bunny() {
	LOG("loading bunny");
	mesh bunny = load_ply("G:\\Raytracing\\RT2019\\RT2019\\models\\bun_zipper.ply");
	LOG("loading bunny finished.");
	fit_mesh_bb(bunny);
	BVH bvh_bunny;

	bvh_bunny.host_construct(bunny);
	//bvh_bunny.calculate_bounding_boxes(bunny);

	LOG("bounding box of root = " << bvh_bunny.nodes[0].boundin_box);
	return bvh_bunny;
}

std::unique_ptr<BVH> load_and_test_simple() {
	LOG("creating simple primitive object (just a few triangles).");
	mesh m;
	m.insert_vertex(vec3(0, 1, 0));
	m.insert_vertex(vec3(0, 0, 0));
	m.insert_vertex(vec3(1, 0, 0));

	m.faces.push_back(Triangle(0, 1, 2));
	
	fit_mesh_bb(m);
    std::unique_ptr<BVH> b = std::make_unique<BVH>();
	b->host_construct(m);
	return b;
	
}
