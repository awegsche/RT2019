#include "ascii_ply.h"
#include "commonheader.h"

#include "triangle.h"
#include "mesh.h"

#include "string"
#include <vector>
#include "fstream"
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace boost;
using namespace geometry;

const size_t BufLength = 4096;
const string PlyFormat("ply");
const string FormatAscii("ascii");
const string TagFormat("format");
const string TagElement("element");
const string ElementVertex("vertex");
const string ElementFace("face");
const string EndHeader("end_header");


mesh load_ply(const string& filename, bool fit_bounding_box)
{
    FILE *f = fopen(filename.c_str(), "r");

    if (!f) {
        ERR("could not open '" << filename << "'.");
        return mesh();
    }

    mesh ply;
    char* buf = new char[BufLength];

    fgets(buf, BufLength, f);
    string filet(buf);
    algorithm::trim(filet);
    if (filet.compare(PlyFormat) != 0)
    {
        ERR("'" << filename << "' is not of type PLY. First line reads '" << filet << "'");
        return mesh();
    }
    fgets(buf, BufLength, f);
    string format(buf);
    algorithm::trim(format);

    vector<string> formatline;
    algorithm::split(formatline, format, algorithm::is_any_of(" \t"), algorithm::token_compress_on);

    if(formatline[0].compare(TagFormat) != 0)
    {
        ERR("expected 'format ascii VERSION' but got " << format);
        return mesh();
    }
    if (formatline[1].compare(FormatAscii) != 0)
    {
        ERR("unknown format '" << formatline[1] << "'");
    }

    DBG("loading PLY file. " << format << ".");

    int vertexcount;
    int facecount;

    while(!feof(f)) {
        fgets(buf, BufLength, f);
        string line(buf);
        algorithm::trim(line);
        vector<string> words;
        algorithm::split(words, line, algorithm::is_any_of(" \t"), algorithm::token_compress_on);
        
//        DBG("reading header line '" << line << "'");

        if(words[0].compare(TagElement) == 0)
        {
            if(words[1].compare(ElementVertex) == 0)
            {
                vertexcount = stoi(words[2]);
            }
            else if (words[1].compare(ElementFace) == 0)
            {
                facecount = stoi(words[2]);
            }
        }
        else if (words[0].compare(EndHeader) == 0)
        {
            break;
        }
        else
            DBG("unknown header field " << words[0] << ". Whole line: " << line);

    }

    DBG("vertex count = " << vertexcount);
    DBG("face count = " << facecount);

    for (uint i = 0 ; i < vertexcount; i++) {

        fgets(buf, BufLength, f);
        string line(buf);
        algorithm::trim(line);
        vector<string> words;
        algorithm::split(words, line, algorithm::is_any_of(" \t"), algorithm::token_compress_on);

        ply.insert_vertex(
            vec3(atof(words[0].c_str()),
                     atof(words[1].c_str()),
                     atof(words[2].c_str()))
        );
    }
    for (uint i = 0 ; i < facecount; i++) {

        fgets(buf, BufLength, f);
        string line(buf);
        algorithm::trim(line);
        vector<string> words;
        algorithm::split(words, line, algorithm::is_any_of(" \t"), algorithm::token_compress_on);

        ply.faces.push_back(
            Triangle(atoi(words[1].c_str()),
                     atoi(words[2].c_str()),
                     atoi(words[3].c_str()))
        );
    }
    fclose(f);

    DBG("Done loading PLY file [" << vertexcount << "|" << facecount << "], bb: " << ply.bounding_box);
    return ply;
}
