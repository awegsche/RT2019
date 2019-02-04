/**@file	/media/awegsche/HDD/files/RT2019/test4linux/inspector.h
 * @author	awegsche
 * @version	801
 * @date
 * 	Created:	11th Jan 2019
 * 	Last Update:	11th Jan 2019
 */

#ifndef INSPECTOR_H
#define INSPECTOR_H

#include "geometry/mesh.h"
#include "geometry/bvh.h"

#include <iostream>
#include <string>
#include <sstream>

const string CMD_quit("q");
const string FMT_in("\33[38;2;128;128;128m");
const string FMT_0("\33[0m");

using namespace std;

template<class T>
void inspect(const T& object) {
    cout << object;
}

string commandline() {
    cout << FMT_in << "[in: ] " << FMT_0;
    string input = "";
    getline(cin, input);
    return input;
}

// this specializatoin should probably go elsewhere

using namespace geometry;
const string CMD_faces("faces");

template<>
void inspect<mesh>(const mesh& m) {
    cout << "MESH\n"
        << "vertices: " << m.vertices.size()
        << "\nfaces: " << m.faces.size()
        << "\nbounding box: " << m.bounding_box << "\n";

    bool cont = true;

    while(cont) 
    {    
        string line = commandline();
        if (CMD_quit.compare(line) == 0) {
           cont = false; 
        }
        else if (CMD_faces.compare(line) == 0) {
            for (auto face : m.faces) {
                cout << face << "\n";
            }
        }
    }
}

const string CMD_display_nodes("nodes");
const string CMD_display_triangles("tris");
template<>
void inspect<BVH>(const BVH& b)
{
    cout << "Terro BVH\n";
    cout << "numTriangles: " << b.numTriangles << "\n";

    bool cont = true;

    while(cont)
    {    
        string line = commandline();

        if (CMD_quit.compare(line) == 0) {
           cont = false; 
        }
        else if (CMD_display_nodes.compare(line) == 0) {
            cout << "--- not yet implemented ----"; 
        }
        else if (CMD_display_triangles.compare(line) == 0) {
            cout << "--- not yet implemented ----"; 
        }
    }
    
}

#endif // INSPECTOR_H
