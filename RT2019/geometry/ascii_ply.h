#pragma once

#include "mesh.h"
#include <string>


geometry::mesh load_ply(const std::string& filename, bool fit_bounding_box = false);
