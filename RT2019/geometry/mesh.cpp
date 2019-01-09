#include "mesh.h"


void geometry::mesh::insert_vertex(const vec3 &vertex)
{
    vertices.push_back(vertex);
    bounding_box.grow(vertex);
}

void geometry::fit_mesh_bb(geometry::mesh &m)
{
    vec3 sides = m.bounding_box.sides();


    if (sides.squared_length() == 0.0f)
        throw underflow_error("the bounding box of the mesh is infinitely small. did you forget to construct it?");
    float max_side = 1.0f / max(max(sides.x, sides.y), sides.z);


    for(auto& vertex : m.vertices) {
        vertex = (vertex - m.bounding_box[0]) * max_side;
    }
    m.bounding_box = AABB(vec3(), vec3(sides * max_side));
}
