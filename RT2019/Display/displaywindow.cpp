#include "displaywindow.h"

#include "../Utils/spectrum.h"
#include "logger.h"

#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <iomanip>

using namespace std;
using namespace std::chrono;


int ScreenWidth = 16 * 16 * 3;
int ScreenHeight = 9 * 16 * 3;

GLuint _vbo;
int glutWindowId;

int frames = 0;
int fps_frames = 0;
namespace Display {
	Tracer* tracer_ptr = nullptr;
}
Sensor film(ScreenWidth, ScreenHeight);



time_point<steady_clock> last_time;

int init_window()
{
	LOG("Initializing OpenGL window and setup everything that we might need to display stuff.");

	cudaError_t err = cudaError::cudaSuccess;
	int argc = 0;
	glutInit(&argc, nullptr);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowPosition(100, 100);
	LOG("Initial Screen dimensions: " << ScreenWidth << "x" << ScreenHeight);
	glutInitWindowSize(ScreenWidth, ScreenHeight);
	glutWindowId = glutCreateWindow("RT2019");
	DBG("Window ID: " << glutWindowId);


	glClearColor(.0f, .0f, .5f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(.0f, ScreenWidth, 0.0f, ScreenHeight);
	LOG("OpenGL Initialised.");

	glutDisplayFunc(disp);
	glutKeyboardFunc(keyboard);
	
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		ERR("Suppor for necessary OpenGL extensions (V2.0) missing.");
		return 1;
	}
	LOG("GLEW Initialised.");

	createVBO(&_vbo);
	LOG("VBO created");



	// ---------------------------------------
	LOG("Display successfully initialised.");

	last_time = steady_clock::now();
	//Timer(0);
	return 0;
}

int init_tracer(Tracer * tracer)
{
	Display::tracer_ptr = tracer;
	film.init();
	return 0;
}

void createVBO(GLuint* vbo)
{
	//Create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//Initialize VBO
	unsigned int size = ScreenWidth * ScreenHeight * sizeof(spectrum);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//Register VBO with CUDA
	if (cudaError::cudaSuccess != cudaGLRegisterBufferObject(*vbo))
		ERR("Couldn't register VBO to CUDA.");
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key) {
	case('q'): glutDestroyWindow(glutWindowId); break;
	}
}

// display function called by glutMainLoop(), gets executed every frame 
void disp(void)
{
	auto duration = duration_cast<milliseconds>
		(steady_clock::now() - last_time);
	float ms = (float)duration.count();

	if (ms > 1000.0f) {
		LOG(std::setprecision(5) << " fps. "
			<< (float)fps_frames / ms * 1000.0f
			<< " total frames: " << frames);
		last_time = steady_clock::now();
		fps_frames = 0;
	}
	cudaThreadSynchronize();
	film.map(_vbo);

	glClear(GL_COLOR_BUFFER_BIT);
	glClearColor(.5f, .5f, .5f, 1.0f);

	Display::tracer_ptr->render(film);

	cudaError_t err;

	if (cudaError::cudaSuccess != (err = cudaThreadSynchronize()))
	{
		ERR("Synchronising CUDA thread failed. " << err);
	}

//	RGBColor* buffer = new RGBColor[100];
//	cudaMemcpy(buffer, film.negative, 100 * sizeof(RGBColor), cudaMemcpyDeviceToHost);
//
//	DBG(buffer[frames].r << ", " << buffer[frames].b << ", " << buffer[frames].g);

	if (cudaError::cudaSuccess != (err = cudaGLUnmapBufferObject(_vbo)))
		ERR("Unmapping VBO failed" << err);

	glFlush();
	glFinish();
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, ScreenWidth * ScreenHeight);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
	fps_frames++;
	frames++;
	glutPostRedisplay();
}

void Timer(int obsolete) {
	glutPostRedisplay();
	glutTimerFunc(1, Timer, 0);
}
