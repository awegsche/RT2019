#pragma once
#include "film.h"
#include "../Tracer/tracer.h"
//#include <GL/glew.h>
#include <chrono>

extern int ScreenWidth;
extern int ScreenHeight;
extern GLuint _vbo;
extern Sensor film;
namespace Display {
	extern Tracer* tracer_ptr;
}


// initialiye the openGL window and setup everything that we might need
int init_window();
int init_tracer(Tracer* tracer);

void disp();

void createVBO(GLuint* vbo);
void keyboard(unsigned char key, int x, int y);
void Timer(int obsolete);