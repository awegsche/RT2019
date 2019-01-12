#pragma once
#include "geometry/bvh.h"
#include "Camera.h"
#include <memory>

class World
{
public:
	unique_ptr<geometry::BVH> single_bvh;  // ideally we have here something bigger, a BVH whose leaves are BVHs for example
	//unique_otr<SimpleCamera> cam;

	World() : single_bvh(nullptr)/*, cam(nullptr)*/ {}

	void set_cam(SimpleCamera* camera) {
		cudaMalloc(&cam, sizeof(SimpleCamera));
		cudaMemcpy(cam, camera, sizeof(SimpleCamera), cudaMemcpyHostToDevice);
	}

	~World() {
		cudaFree(cam);
	}
};

