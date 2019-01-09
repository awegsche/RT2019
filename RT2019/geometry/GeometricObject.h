#pragma once

namespace geometry
{

/**
 * @brief  Geometric Object (C++) base class.
 * This will take care of the triangle mesh, assigned materials and animations on the host side
 * @note   
 * @retval None
 */
class GeometricObject 
{
public:
    GeometricObject() {}

    /**
     * @brief  Copies the data to the device.
     * @note   
     * @param  dest: 
     * @retval None
     */
    void copy_to_cuda(void** dest) = 0;
};

}