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

static float4 densityColor;

static float2 force;

// visualization
static int width = 512;
static int height = 512;

static int visualizationMethod = 0;

void initSimulation();
void resetSimulationHost();
void resetPresssure();
void simulateAdvection();
void simulateVorticity();
void simulateDiffusion();
void projection();
void simulateDensityAdvection();
void addForce(int x, int y, float2 force);
void simulationStep();
void visualizationStep();


#endif