#include "morton.h"
#include "commonheader.h"

#include <algorithm>

using std::max;
using std::min;

morton_int expandBits(morton_int v)
{
    v = (v * 0x00010001ull) & 0xFF0000FFull;
    v = (v * 0x00000101ull) & 0x0F00F00Full;
    v = (v * 0x00000011ull) & 0xC30C30C3ull;
    v = (v * 0x00000005ull) & 0x49249249ull;
    return v;
}

morton_int morton3D(float x, float y, float z)
{
    x = min(max(x * 1048576.0f, 0.0f), 1048575.0f);
    y = min(max(y * 1048576.0f, 0.0f), 1048575.0f);
    z = min(max(z * 1048576.0f, 0.0f), 1048575.0f);
    morton_int xx = expandBits((morton_int)x);
    morton_int yy = expandBits((morton_int)y);
    morton_int zz = expandBits((morton_int)z);
    return xx * 4 + yy * 2 + zz;
}


