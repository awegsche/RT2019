#ifndef MORTON_H
#define MORTON_H

#include "commonheader.h"

// from Tero Karras' article

typedef unsigned int morton_int;

morton_int expandBits(morton_int v);

///
/// \brief morton3D
/// \param x
/// \param y
/// \param z
/// \param mesh_bb
/// \return
/// Calculates the morton code of the coordinate (x,y,z) inside the bounding box mesh_bb
morton_int morton3D(float x, float y, float z);

#endif // MORTON_H
