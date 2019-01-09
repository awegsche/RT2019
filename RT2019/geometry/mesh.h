#pragma once

#include "commonheader.h"
#include "triangle.h"
#include "aabb.h"
#include <vector>
//#include <map>
//#include <boost/sort/sort.hpp>

using std::vector;

namespace geometry 
{

struct mesh
{
    vector<vec3> vertices;
    vector<Triangle> faces;

    AABB bounding_box;

    mesh() : vertices(), faces(), bounding_box() {}

    void insert_vertex(const vec3& vertex);
};

void fit_mesh_bb(mesh& m);

}
