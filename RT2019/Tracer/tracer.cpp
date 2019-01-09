#include "tracer.h"
#include "cudarenderkernel.h"



void Tracer::render(Sensor & film)
{
	Render(&film, world_ptr);
}
