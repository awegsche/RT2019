#include "triangle.h"


#include <sstream>



unsigned long long geometry::morton(const geometry::Triangle& tri, const vector<vec3>& vertices)
{
    return morton3D(
        .33f * (vertices[tri.v1].x + vertices[tri.v2].x + vertices[tri.v3].x),
        .33f * (vertices[tri.v1].y + vertices[tri.v2].y + vertices[tri.v3].y),
        .33f * (vertices[tri.v1].z + vertices[tri.v2].z + vertices[tri.v3].z));
}

string geometry::Triangle::to_string(const vector<vec3> &vertices)
{
   return static_cast<stringstream&>(ostringstream() << "(" << vtos(vertices[v1]) << ", " << vtos(vertices[v2]) << ", " << vtos(vertices[v3]) << ")").str();
}

ostream &geometry::operator<<(ostream &os, const geometry::Triangle &tri)
{
    return (os << "(" << tri.v1 << ", " << tri.v2 << ", " << tri.v3 << ")");
}
