#pragma once
#include "../Utils/spectrum.h"
#include "../Display/film.h"
#include "../World.h"

class Tracer
{
private:
	// World etc.
	World* world_ptr;

public:
	Tracer() {}

	/// <summary>
	/// Does the rendering (hopefully in CUDA)
	/// </summary>
	/// <param name="film">The final </param>	
	void render(Sensor& film);

	void set_world(World* w) {
		world_ptr = w;
	}
};
