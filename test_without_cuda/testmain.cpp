/**@file	/media/awegsche/HDD/files/RT2019/test4linux/testmain.cpp
 * @author	awegsche
 * @version	801
 * @date
 * 	Created:	10th Jan 2019
 * 	Last Update:	10th Jan 2019
 */

// { Includes

#include "geometry/mesh.h"
#include "geometry/bvh.h"
#include "inspector.h"
#include <iostream>

// }

/*===========================================================================*/
/*==============================[ main ]==============================*/
/*===========================================================================*/

using namespace std;

int main () {
    LOG("hello test");

    geometry::mesh m;
    m.insert_vertex(vec3(0,0,0));
    m.insert_vertex(vec3(1,0,0));
    m.insert_vertex(vec3(0,1,0));
    m.insert_vertex(vec3(1,1,0));

    m.insert_vertex(vec3(0,0,1));
    m.insert_vertex(vec3(1,0,1));
    m.insert_vertex(vec3(0,1,1));
    m.insert_vertex(vec3(1,1,1));

    m.faces.push_back(Triangle(0,1,2));
    m.faces.push_back(Triangle(1,3,2));
    
    m.faces.push_back(Triangle(4,5,6));
    m.faces.push_back(Triangle(5,7,6));

    geometry::BVH b;
    b.host_construct(m);

    inspect(m);
    inspect(b);
    
    return 0;
}
