
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "cutil_math.h"

__device__
float4 cross(float4 a, float4 b) {
	float3 res =cross(make_float3(a.x, a.y, a.z), make_float3(b.x, b.y, b.z));
	return(make_float4(res.x, res.y, res.z, 0.0f));
}

__device__
float fract(float x, float* iptr) {
	*iptr =  x - floorf(x);
	return x - floorf(x);
}

__device__
float2 mix(float2 x, float2 y, float a) {
	return x + (y - x) * a;
}

__device__
float4 mix(float4 x, float4 y, float a) {
	return x + (y - x) * a;
}

/* -*- mode: c++ -*- */

__constant__ float dt = 0.1f;

__global__
void resetSimulation(const int gridResolution,
	 float2* velocityBuffer,
	 float* pressureBuffer,
	 float4* densityBuffer) {
	
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x < gridResolution && id.y < gridResolution) {
		velocityBuffer[id.x + id.y * gridResolution] = make_float2(0.0f, 0.0f);
		pressureBuffer[id.x + id.y * gridResolution] = 0.0f;
		densityBuffer[id.x + id.y * gridResolution] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	}
}

// bilinear interpolation
__device__
float2 getBil(float2 p, int gridResolution,  float2* buffer) {
	p = clamp(p, make_float2(0.0f), make_float2(gridResolution));

	float2 p00 = buffer[(int)(p.x) + (int)(p.y) * gridResolution];
	float2 p10 = buffer[(int)(p.x) + 1 + (int)(p.y) * gridResolution];
	float2 p11 = buffer[(int)(p.x) + 1 + (int)(p.y + 1.0f) * gridResolution];
	float2 p01 = buffer[(int)(p.x) + (int)(p.y + 1.0f) * gridResolution];

	float flr;
	float t0 = fract(p.x, &flr);
	float t1 = fract(p.y, &flr);

	float2 v0 = mix(p00, p10, t0);
	float2 v1 = mix(p01, p11, t0);

	return mix(v0, v1, t1);
}

__device__
float4 getBil4(float2 p, int gridResolution,  float4* buffer) {
	p = clamp(p, make_float2(0.0f), make_float2(gridResolution));

	float4 p00 = buffer[(int)(p.x) + (int)(p.y) * gridResolution];
	float4 p10 = buffer[(int)(p.x) + 1 + (int)(p.y) * gridResolution];
	float4 p11 = buffer[(int)(p.x) + 1 + (int)(p.y + 1.0f) * gridResolution];
	float4 p01 = buffer[(int)(p.x) + (int)(p.y + 1.0f) * gridResolution];

	float flr;
	float t0 = fract(p.x, &flr);
	float t1 = fract(p.y, &flr);

	float4 v0 = mix(p00, p10, t0);
	float4 v1 = mix(p01, p11, t0);

	return mix(v0, v1, t1);
}

__global__
void advection(const int gridResolution,
	 float2* inputVelocityBuffer,
	 float2* outputVelocityBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x > 0 && id.x < gridResolution - 1 &&
		id.y > 0 && id.y < gridResolution - 1) {
		float2 velocity = inputVelocityBuffer[id.x + id.y * gridResolution];

		float2 p = make_float2((float)id.x - dt * velocity.x, (float)id.y - dt * velocity.y);

		outputVelocityBuffer[id.x + id.y * gridResolution] = getBil(p, gridResolution, inputVelocityBuffer);
	}
	else {
		if (id.x == 0) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + id.y * gridResolution];
		if (id.x == gridResolution - 1) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x - 1 + id.y * gridResolution];
		if (id.y == 0) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + (id.y + 1) * gridResolution];
		if (id.y == gridResolution - 1) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + (id.y - 1) * gridResolution];
	}
}

__global__
void advectionDensity(const int gridResolution,
	 float2* velocityBuffer,
	 float4* inputDensityBuffer,
	 float4* outputDensityBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x > 0 && id.x < gridResolution - 1 &&
		id.y > 0 && id.y < gridResolution - 1) {
		float2 velocity = velocityBuffer[id.x + id.y * gridResolution];

		float2 p = make_float2((float)id.x - dt * velocity.x, (float)id.y - dt * velocity.y);

		outputDensityBuffer[id.x + id.y * gridResolution] = getBil4(p, gridResolution, inputDensityBuffer);
	}
	else {
		outputDensityBuffer[id.x + id.y * gridResolution] = make_float4(0.0f);
	}
}

__global__
void diffusion(const int gridResolution,
	 float2* inputVelocityBuffer,
	 float2* outputVelocityBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	float viscousity = 0.01f;
	float alpha = 1.0f / (viscousity * dt);
	float beta = 1.0f / (4.0f + alpha);

	if (id.x > 0 && id.x < gridResolution - 1 &&
		id.y > 0 && id.y < gridResolution - 1) {
		float2 vL = inputVelocityBuffer[id.x - 1 + id.y * gridResolution];
		float2 vR = inputVelocityBuffer[id.x + 1 + id.y * gridResolution];
		float2 vB = inputVelocityBuffer[id.x + (id.y - 1) * gridResolution];
		float2 vT = inputVelocityBuffer[id.x + (id.y + 1) * gridResolution];

		float2 velocity = inputVelocityBuffer[id.x + id.y * gridResolution];

		outputVelocityBuffer[id.x + id.y * gridResolution] = (vL + vR + vB + vT + alpha * velocity) * beta;
	}
	else {
		outputVelocityBuffer[id.x + id.y * gridResolution] = inputVelocityBuffer[id.x + id.y * gridResolution];
	}
}

__global__
void vorticity(const int gridResolution,  float2* velocityBuffer,
	 float* vorticityBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x > 0 && id.x < gridResolution - 1 &&
		id.y > 0 && id.y < gridResolution - 1) {
		float2 vL = velocityBuffer[id.x - 1 + id.y * gridResolution];
		float2 vR = velocityBuffer[id.x + 1 + id.y * gridResolution];
		float2 vB = velocityBuffer[id.x + (id.y - 1) * gridResolution];
		float2 vT = velocityBuffer[id.x + (id.y + 1) * gridResolution];

		vorticityBuffer[id.x + id.y * gridResolution] = (vR.y - vL.y) - (vT.x - vB.x);
	}
	else {
		vorticityBuffer[id.x + id.y * gridResolution] = 0.0f;
	}
}

__global__
void addVorticity(const int gridResolution,  float* vorticityBuffer,
	 float2* velocityBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const float scale = 0.2f;

	if (id.x > 0 && id.x < gridResolution - 1 &&
		id.y > 0 && id.y < gridResolution - 1) {
		float vL = vorticityBuffer[id.x - 1 + id.y * gridResolution];
		float vR = vorticityBuffer[id.x + 1 + id.y * gridResolution];
		float vB = vorticityBuffer[id.x + (id.y - 1) * gridResolution];
		float vT = vorticityBuffer[id.x + (id.y + 1) * gridResolution];

		float4 gradV = make_float4(vR - vL, vT - vB, 0.0f, 0.0f);
		float4 z = make_float4(0.0f, 0.0f, 1.0f, 0.0f);

		if (dot(gradV, gradV)) {
			float4 vorticityForce = scale * cross(gradV, z);
			velocityBuffer[id.x + id.y * gridResolution] += make_float2(vorticityForce.x, vorticityForce.y) * dt;
		}
	}
}

__global__
void divergence(const int gridResolution,  float2* velocityBuffer,
	 float* divergenceBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x > 0 && id.x < gridResolution - 1 &&
		id.y > 0 && id.y < gridResolution - 1) {
		float2 vL = velocityBuffer[id.x - 1 + id.y * gridResolution];
		float2 vR = velocityBuffer[id.x + 1 + id.y * gridResolution];
		float2 vB = velocityBuffer[id.x + (id.y - 1) * gridResolution];
		float2 vT = velocityBuffer[id.x + (id.y + 1) * gridResolution];

		divergenceBuffer[id.x + id.y * gridResolution] = 0.5f * ((vR.x - vL.x) + (vT.y - vB.y));
	}
	else {
		divergenceBuffer[id.x + id.y * gridResolution] = 0.0f;
	}
}

__global__
void pressureJacobi(const int gridResolution,
	 float* inputPressureBuffer,
	 float* outputPressureBuffer,
	 float* divergenceBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x > 0 && id.x < gridResolution - 1 &&
		id.y > 0 && id.y < gridResolution - 1) {

		float alpha = -1.0f;
		float beta = 0.25f;

		float vL = inputPressureBuffer[id.x - 1 + id.y * gridResolution];
		float vR = inputPressureBuffer[id.x + 1 + id.y * gridResolution];
		float vB = inputPressureBuffer[id.x + (id.y - 1) * gridResolution];
		float vT = inputPressureBuffer[id.x + (id.y + 1) * gridResolution];

		float divergence = divergenceBuffer[id.x + id.y * gridResolution];

		outputPressureBuffer[id.x + id.y * gridResolution] = (vL + vR + vB + vT + alpha * divergence) * beta;
	}
	else {
		if (id.x == 0) outputPressureBuffer[id.x + id.y * gridResolution] = inputPressureBuffer[id.x + 1 + id.y * gridResolution];
		if (id.x == gridResolution - 1) outputPressureBuffer[id.x + id.y * gridResolution] = inputPressureBuffer[id.x - 1 + id.y * gridResolution];
		if (id.y == 0) outputPressureBuffer[id.x + id.y * gridResolution] = inputPressureBuffer[id.x + (id.y + 1) * gridResolution];
		if (id.y == gridResolution - 1) outputPressureBuffer[id.x + id.y * gridResolution] = inputPressureBuffer[id.x + (id.y - 1) * gridResolution];
	}
}

__global__
void projection(const int gridResolution,
	 float2* inputVelocityBuffer,
	 float* pressureBuffer,
	 float2* outputVelocityBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x > 0 && id.x < gridResolution - 1 &&
		id.y > 0 && id.y < gridResolution - 1) {
		float pL = pressureBuffer[id.x - 1 + id.y * gridResolution];
		float pR = pressureBuffer[id.x + 1 + id.y * gridResolution];
		float pB = pressureBuffer[id.x + (id.y - 1) * gridResolution];
		float pT = pressureBuffer[id.x + (id.y + 1) * gridResolution];

		float2 velocity = inputVelocityBuffer[id.x + id.y * gridResolution];

		outputVelocityBuffer[id.x + id.y * gridResolution] = velocity -  /* 0.5f **//* (1.0f / 256.0f) **/ make_float2(pR - pL, pT - pB);
	}
	else {
		if (id.x == 0) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + id.y * gridResolution];
		if (id.x == gridResolution - 1) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x - 1 + id.y * gridResolution];
		if (id.y == 0) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + (id.y + 1) * gridResolution];
		if (id.y == gridResolution - 1) outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + 1 + (id.y - 1) * gridResolution];
	}
}

__global__
void addForce(const float x, const float y, const float2 force,
	const int gridResolution,  float2* velocityBuffer,
	const float4 density,  float4* densityBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	float dx = ((float)id.x / (float)gridResolution) - x;
	float dy = ((float)id.y / (float)gridResolution) - y;

	float radius = 0.001f;

	float c = exp(-(dx * dx + dy * dy) / radius) * dt;

	velocityBuffer[id.x + id.y * gridResolution] += c * force;
	densityBuffer[id.x + id.y * gridResolution] += c * density;
}

// *************
// Visualization
// *************

__global__
void visualizationDensity(const int width, const int height,  float4* visualizationBuffer,
	const int gridResolution,  float4* densityBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x < width && id.y < height) {
		float4 density = densityBuffer[id.x + id.y * width];
		visualizationBuffer[id.x + id.y * width] = density;
	}
}

__global__
void visualizationVelocity(const int width, const int height,  float4* visualizationBuffer,
	const int gridResolution,  float2* velocityBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x < width && id.y < height) {
		float2 velocity = velocityBuffer[id.x + id.y * width];
		visualizationBuffer[id.x + id.y * width] = make_float4(((1.0f + velocity) / 2.0f).x, ((1.0f + velocity) / 2.0f).y, 0.0f, 0.0f);
	}
}

__global__
void visualizationPressure(const int width, const int height,  float4* visualizationBuffer,
	const int gridResolution,  float* pressureBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x < width && id.y < height) {
		float pressure = pressureBuffer[id.x + id.y * width];
		visualizationBuffer[id.x + id.y * width] = make_float4((1.0f + pressure) / 2.0f);
	}
}

int gridResolution = 512;
int* d_gridResolution;

int inputVelocityBuffer = 0;
float2* d_velocityBuffer[2];

int inputDensityBuffer = 0;
float4* d_densityBuffer[2];
float4 densityColor;

int inputPressureBuffer = 0;
float* d_pressureBuffer[2];
float* d_divergenceBuffer;

float* d_vorticityBuffer;

size_t problemSize[2];

float2 force;

// visualization
int width = 512;
int height = 512;

float4* d_visualizationBufferGPU;
float4* visualizationBufferCPU;

int visualizationMethod = 0;

size_t visualizationSize[2];

void initSimulation() {
	// simulation
	problemSize[0] = gridResolution;
	problemSize[1] = gridResolution;

//Allocating device memory
	//velocityBuffer
	cudaMalloc((void**)&d_velocityBuffer[0], sizeof(float2)*gridResolution*gridResolution);
	cudaMalloc((void**)&d_velocityBuffer[1], sizeof(float2)*gridResolution*gridResolution);
	
	//densityBuffer
	cudaMalloc((void**)&d_densityBuffer[0], sizeof(float4)*gridResolution*gridResolution);
	cudaMalloc((void**)&d_densityBuffer[1], sizeof(float4)*gridResolution*gridResolution);

	//pressureBuffer
	cudaMalloc((void**)&d_pressureBuffer[0], sizeof(float)*gridResolution*gridResolution);
	cudaMalloc((void**)&d_pressureBuffer[1], sizeof(float)*gridResolution*gridResolution);

	//divergenceBuffer
	cudaMalloc((void**)&d_divergenceBuffer, sizeof(float)*gridResolution*gridResolution);

	//vorticityBuffer
	cudaMalloc((void**)&d_vorticityBuffer, sizeof(float)*gridResolution*gridResolution);

	densityColor.x = densityColor.y = densityColor.z = densityColor.w = 1.0f;


	// visualization
	visualizationSize[0] = width;
	visualizationSize[1] = height;

	//CPU
	visualizationBufferCPU = new float4[width * height];

	//GPU visualizationBuffer
	cudaMalloc((void**)&d_visualizationBufferGPU, sizeof(float4)*width*height);
}

void resetSimulationHost() {
	resetSimulation <<<512, 512>>>(
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		d_pressureBuffer[inputPressureBuffer],
		d_densityBuffer[inputDensityBuffer]
		);
}

void resetPresssure() {
	resetSimulation <<<512, 512>>>(
		gridResolution,
		d_velocityBuffer[(inputVelocityBuffer + 1) % 2],
		d_pressureBuffer[inputPressureBuffer],
		d_densityBuffer[(inputVelocityBuffer + 1) % 2]
		);
}

void simulateAdvection() {
	advection<<<512, 512>>>(
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		d_velocityBuffer[(inputVelocityBuffer + 1) % 2]
		);
}


void simulateVorticity() {
	vorticity<<<512, 512>>>(
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		d_vorticityBuffer
		);

	addVorticity<<<512, 512>>>(
		gridResolution,
		d_vorticityBuffer,
		d_velocityBuffer[inputVelocityBuffer]
		);
}

void simulateDiffusion() {
	for (int i = 0; i < 10; ++i) {
		diffusion<<<512,512>>>(
			gridResolution,
			d_velocityBuffer[inputVelocityBuffer],
			d_velocityBuffer[(inputVelocityBuffer + 1) % 2]
			);

		inputVelocityBuffer = (inputVelocityBuffer + 1) % 2;
	}
}

void projection() {
	divergence<<<512,512>>>(
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		d_divergenceBuffer
		);

	resetPresssure();

	for (int i = 0; i < 10; ++i) {
		pressureJacobi<<<512,512>>>(
			gridResolution,
			d_pressureBuffer[inputPressureBuffer],
			d_pressureBuffer[(inputPressureBuffer + 1) % 2],
			d_divergenceBuffer
			);


		inputPressureBuffer = (inputPressureBuffer + 1) % 2;
	}

	projection<<<512,512>>>(
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		d_pressureBuffer[inputPressureBuffer],
		d_velocityBuffer[(inputVelocityBuffer + 1) % 2]
		);

	inputVelocityBuffer = (inputVelocityBuffer + 1) % 2;
}

void simulateDensityAdvection() {
	advectionDensity<<<512,512>>>(
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		d_densityBuffer[inputDensityBuffer],
		d_densityBuffer[(inputDensityBuffer + 1) % 2]
		);

	inputDensityBuffer = (inputDensityBuffer + 1) % 2;
}

void addForce(int x, int y, float2 force) {
	float fx = (float)x / width;
	float fy = (float)y / height;

	addForce<<<512,512>>>(
		fx,
		fy,
		force,
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		densityColor,
		d_densityBuffer[inputDensityBuffer]
		);
}

void simulationStep() {
	simulateAdvection();
	simulateDiffusion();
	simulateVorticity();
	projection();
	simulateDensityAdvection();
}

void visualizationStep() {
	switch (visualizationMethod) {
	case 0:
		visualizationDensity<<<512,512>>>(
			width,
			height,
			d_visualizationBufferGPU,
			gridResolution,
			d_densityBuffer[inputDensityBuffer]
			);
		break;
	case 1:
		visualizationVelocity<<<512,512 >>>(
			width,
			height,
			d_visualizationBufferGPU,
			gridResolution,
			d_velocityBuffer[inputVelocityBuffer]
			);
		break;
	case 2:
		visualizationPressure<<<512,512>>>(
			width,
			height,
			d_visualizationBufferGPU,
			gridResolution,
			d_pressureBuffer[inputPressureBuffer]
			);
		break;
	
	}

	cudaMemcpy(visualizationBufferCPU, d_visualizationBufferGPU, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
	glDrawPixels(width, height, GL_RGBA, GL_FLOAT, visualizationBufferCPU);
}

// OpenGL
int method = 1;
bool keysPressed[256];

void initOpenGL() {
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
	}
	else {
		if (GLEW_VERSION_3_0)
		{
			std::cout << "Driver supports OpenGL 3.0\nDetails:" << std::endl;
			std::cout << "  Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
			std::cout << "  Vendor: " << glGetString(GL_VENDOR) << std::endl;
			std::cout << "  Renderer: " << glGetString(GL_RENDERER) << std::endl;
			std::cout << "  Version: " << glGetString(GL_VERSION) << std::endl;
			std::cout << "  GLSL: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
		}
	}

	glClearColor(0.17f, 0.4f, 0.6f, 1.0f);
}

void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	simulationStep();
	visualizationStep();

	glEnable(GL_DEPTH_TEST);
	glutSwapBuffers();
}

void idle() {
	glutPostRedisplay();
}

void keyDown(unsigned char key, int x, int y) {
	keysPressed[key] = true;
}

void keyUp(unsigned char key, int x, int y) {
	keysPressed[key] = false;
	switch (key) {
	case 'r':
		resetSimulationHost();
		break;

	case 'd':
		visualizationMethod = 0;
		break;
	case 'v':
		visualizationMethod = 1;
		break;
	case 'p':
		visualizationMethod = 2;
		break;

	case '1':
		densityColor.x = densityColor.y = densityColor.z = densityColor.w = 1.0f;
		break;

	case '2':
		densityColor.x = 1.0f;
		densityColor.y = densityColor.z = densityColor.w = 0.0f;
		break;

	case '3':
		densityColor.y = 1.0f;
		densityColor.x = densityColor.z = densityColor.w = 0.0f;
		break;

	case '4':
		densityColor.z = 1.0f;
		densityColor.x = densityColor.y = densityColor.w = 0.0f;
		break;

	case 27:
		exit(0);
		break;
	}
}

int mX, mY;

void mouseClick(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON)
		if (state == GLUT_DOWN) {
			mX = x;
			mY = y;
		}
}

void mouseMove(int x, int y) {
	force.x = (float)(x - mX);
	force.y = -(float)(y - mY);
	//addForce(mX, height - mY, force);
	addForce(256, 256, force);
	mX = x;
	mY = y;
}

void reshape(int newWidth, int newHeight) {
	width = newWidth;
	height = newHeight;
	glViewport(0, 0, width, height);
}

int main(int argc, char* argv[]) {
	glutInit(&argc, argv);
	glutInitContextVersion(3, 0);
	glutInitContextFlags(GLUT_CORE_PROFILE | GLUT_DEBUG);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("GPGPU 13. labor: Incompressible fluid simulation");

	initOpenGL();

	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyDown);
	glutKeyboardUpFunc(keyUp);
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMove);

	// OpenCL processing
	initSimulation();

	glutMainLoop();
	return(0);
}
