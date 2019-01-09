#include "aabb.h"

using std::min;
using std::max;


std::ostream &geometry::operator<<(std::ostream &os, const AABB &box)
{
    return (os << "[" << vtos(box.p0) << " -- " << vtos(box.p1) << "]");
}
