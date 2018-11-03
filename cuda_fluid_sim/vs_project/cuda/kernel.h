#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "cutil_math.h"

void initSimulation(int width, int height);
void resetSimulationHost();
void resetPresssure();
void simulateAdvection();
void simulateVorticity();
void simulateDiffusion();
void projection();
void simulateDensityAdvection();
void addForce(int x, int y, float2 force, float4 densityColor, int width, int height);
void simulationStep();
void visualizationStep(int visualizationMethod, int width, int height);


#endif