#include "kernel.h"

#pragma region Kernel_code

////Defining required device functions for the calculations
__device__
float4 cross(float4 a, float4 b) {
	float3 res = cross(make_float3(a.x, a.y, a.z), make_float3(b.x, b.y, b.z));
	return(make_float4(res.x, res.y, res.z, 0.0f));
}

__device__
float fract(float x, float* iptr) {
	*iptr = fmin(x - floor(x), 0.999999999999f);
	return fmin(x - floor(x), 0.999999999999f);
}

__device__
float2 mix(float2 x, float2 y, float a) {
	return x + (y - x) * a;
}

__device__
float4 mix(float4 x, float4 y, float a) {
	return x + (y - x) * a;
}

// bilinear interpolation
__device__
float2 getBil(float2 p, int gridResolution, float2* buffer) {
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
float4 getBil4(float2 p, int gridResolution, float4* buffer) {
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

__device__
bool isInside(int gridResolution, int2 id,  int* boundaryBuffer) {
	return (boundaryBuffer[id.x + id.y * gridResolution] == 0);
}

__device__
void initBoundaryBuffer(int gridResolution,int2 id, int* boundaryBuffer) {
	if ((id.x > 0 && id.x < gridResolution - 1 && id.y > 0 && id.y < gridResolution-1) &&
		((id.x < 100 || id.x > 200) || (id.y < 300 || id.y >400))) {
		boundaryBuffer[id.x + id.y * gridResolution] = 0;
	}
	else {
		boundaryBuffer[id.x + id.y * gridResolution] = 1;
	}
}
__device__
int2 calculateBoundaryNeighbour(int gridResolution,int2 id, int* boundaryBuffer) {
	int2 tempid = id;
	tempid.x++;
	if (isInside(gridResolution, tempid, boundaryBuffer)) { return tempid; }
	else {
		tempid.x-=2;
		if (isInside(gridResolution, tempid, boundaryBuffer)) { return tempid; }
		else {
			tempid.x++;
			tempid.y++;
			if (isInside(gridResolution, tempid, boundaryBuffer)) { return tempid; }
			else {
				tempid.y-=2;
				if (isInside(gridResolution, tempid, boundaryBuffer)) { return tempid; }
			}
		}
	}
	tempid.y++;
	return tempid;
}

////ENDOF device functions

////Kernel functions

__constant__ float dt = 0.1f;

__global__
void resetSimulation(const int gridResolution,
	float2* velocityBuffer,
	float* pressureBuffer,
	float4* densityBuffer,
	int* boundaryBuffer) {

	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x < gridResolution && id.y < gridResolution) {
		velocityBuffer[id.x + id.y * gridResolution] = make_float2(5.0f, 0.0f);
		pressureBuffer[id.x + id.y * gridResolution] = 0.0f;
		densityBuffer[id.x + id.y * gridResolution] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		initBoundaryBuffer(gridResolution,id,boundaryBuffer);
	}
}

__global__
void advection(const int gridResolution,
	float2* inputVelocityBuffer,
	float2* outputVelocityBuffer,
	int* boundaryBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (isInside(gridResolution,id, boundaryBuffer)) {
		float2 velocity = inputVelocityBuffer[id.x + id.y * gridResolution];

		float2 p = make_float2((float)id.x - dt * velocity.x, (float)id.y - dt * velocity.y);

		outputVelocityBuffer[id.x + id.y * gridResolution] = getBil(p, gridResolution, inputVelocityBuffer);// +make_float2(0.05f, 0.0f);
	}
	else {
		float2 tempvelocity = make_float2(0.0f);
		int n = 0;
		if (boundaryBuffer[id.x + 1 + id.y * gridResolution] == 0) {
			tempvelocity += inputVelocityBuffer[id.x + 1 + id.y * gridResolution];
			n++;
		}
		if (boundaryBuffer[id.x - 1 + id.y * gridResolution] == 0) {
			tempvelocity += inputVelocityBuffer[id.x - 1 + id.y * gridResolution];
			n++;
		}
		if (boundaryBuffer[id.x + (id.y + 1) * gridResolution] == 0) {
			tempvelocity += inputVelocityBuffer[id.x + (id.y + 1) * gridResolution];
			n++;
		}
		if (boundaryBuffer[id.x + (id.y - 1) * gridResolution] == 0) {
			tempvelocity += inputVelocityBuffer[id.x + (id.y - 1) * gridResolution];
			n++;
		}
		if(n!=0)
		outputVelocityBuffer[id.x + id.y * gridResolution] = -(tempvelocity / n);
		else
		outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[id.x + id.y * gridResolution];
	}
}

__global__
void advectionDensity(const int gridResolution,
	float2* velocityBuffer,
	float4* inputDensityBuffer,
	float4* outputDensityBuffer,
	int* boundaryBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (isInside(gridResolution,id, boundaryBuffer)) {
		float2 velocity = velocityBuffer[id.x + id.y * gridResolution];

		float2 p = make_float2((float)id.x - dt * velocity.x, (float)id.y - dt * velocity.y);

		outputDensityBuffer[id.x + id.y * gridResolution] = getBil4(p, gridResolution, inputDensityBuffer);
		if (id.x == 1 && id.y % 2 == 0)
			outputDensityBuffer[id.x + id.y * gridResolution] += make_float4(1.0f);
	}
	else {
		outputDensityBuffer[id.x + id.y * gridResolution] = make_float4(0.0f,0.0f,0.0f,0.0f);
	}
}

__global__
void diffusion(const int gridResolution,
	float2* inputVelocityBuffer,
	float2* outputVelocityBuffer,
	int* boundaryBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	float viscousity = 0.01f;
	float alpha = 1.0f / (viscousity * dt);
	float beta = 1.0f / (4.0f + alpha);

	if (isInside(gridResolution,id, boundaryBuffer)) {
		float2 vL = inputVelocityBuffer[id.x - 1 + id.y * gridResolution];
		float2 vR = inputVelocityBuffer[id.x + 1 + id.y * gridResolution];
		float2 vB = inputVelocityBuffer[id.x + (id.y - 1) * gridResolution];
		float2 vT = inputVelocityBuffer[id.x + (id.y + 1) * gridResolution];

		float2 velocity = inputVelocityBuffer[id.x + id.y * gridResolution];

		outputVelocityBuffer[id.x + id.y * gridResolution] = (vL + vR + vB + vT +alpha * velocity) * beta;
		if (id.x == 1 && id.y % 3 == 0)
			outputVelocityBuffer[id.x + id.y * gridResolution] += make_float2(20.0f,0.0f);


	}
	else {
		outputVelocityBuffer[id.x + id.y * gridResolution] = inputVelocityBuffer[id.x + id.y * gridResolution];
	}
}

__global__
void vorticity(const int gridResolution, float2* velocityBuffer,
	float* vorticityBuffer,
	int* boundaryBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (isInside(gridResolution, id,boundaryBuffer)) {
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
void addVorticity(const int gridResolution, float* vorticityBuffer,
	float2* velocityBuffer, int* boundaryBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const float scale = 0.2f;

	if (isInside(gridResolution, id, boundaryBuffer)) {
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
void divergence(const int gridResolution, float2* velocityBuffer,
	float* divergenceBuffer,
	int* boundaryBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (isInside(gridResolution, id, boundaryBuffer)) {
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
	float* divergenceBuffer,
	int* boundaryBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (isInside(gridResolution, id,boundaryBuffer)) {

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
		int2 tempid = calculateBoundaryNeighbour(gridResolution, id, boundaryBuffer);
		outputPressureBuffer[id.x + id.y * gridResolution] = -inputPressureBuffer[tempid.x + tempid.y * gridResolution];
	}
}

__global__
void projection(const int gridResolution,
	float2* inputVelocityBuffer,
	float* pressureBuffer,
	float2* outputVelocityBuffer,
	int* boundaryBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (isInside(gridResolution, id,boundaryBuffer)) {
		float pL = pressureBuffer[id.x - 1 + id.y * gridResolution];
		float pR = pressureBuffer[id.x + 1 + id.y * gridResolution];
		float pB = pressureBuffer[id.x + (id.y - 1) * gridResolution];
		float pT = pressureBuffer[id.x + (id.y + 1) * gridResolution];

		float2 velocity = inputVelocityBuffer[id.x + id.y * gridResolution];

		outputVelocityBuffer[id.x + id.y * gridResolution] = velocity -  /* 0.5f **//* (1.0f / 256.0f) **/ make_float2(pR - pL, pT - pB);
	}
	else {
		int2 tempid = calculateBoundaryNeighbour(gridResolution, id, boundaryBuffer);
		outputVelocityBuffer[id.x + id.y * gridResolution] = -inputVelocityBuffer[tempid.x + tempid.y * gridResolution];
	}
}

__global__
void addForce(const float x, const float y, const float2 force,
	const int gridResolution, float2* velocityBuffer,
	const float4 density, float4* densityBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	float dx = ((float)id.x / (float)gridResolution) - x;
	float dy = ((float)id.y / (float)gridResolution) - y;

	float radius = 0.001f;

	float c = exp(-(dx * dx + dy * dy) / radius) * dt;

	velocityBuffer[id.x + id.y * gridResolution] += c * force;
	densityBuffer[id.x + id.y * gridResolution] += c * density;
}
/*__global__
void addForcetoSide(const float2 force,
	const int gridResolution, float2* velocityBuffer,
	const float4 density, float4* densityBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);
	float x = 1;
	//float y = gridResolution / 2;
	float dx = ((float)id.x / (float)gridResolution) - x;
	float dy = ((float)id.y / (float)gridResolution);

	float radius = 0.001f;

	float c = exp(-(dx * dx + dy * dy) / radius) * dt;

	velocityBuffer[id.x + id.y * gridResolution] += c * force;
	densityBuffer[id.x + id.y * gridResolution] += c * density;
}
*/
// *************
// Visualization
// *************

__global__
void visualizationDensity(const int width, const int height, float4* visualizationBuffer,
	const int gridResolution, float4* densityBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x < width && id.y < height) {
		float4 density = densityBuffer[id.x + id.y * width];
		visualizationBuffer[id.x + id.y * width] = density;
	}
}

__global__
void visualizationVelocity(const int width, const int height, float4* visualizationBuffer,
	const int gridResolution, float2* velocityBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x < width && id.y < height) {
		float2 velocity = velocityBuffer[id.x + id.y * width];
		visualizationBuffer[id.x + id.y * width] = make_float4(((1.0f + velocity) / 2.0f).x, ((1.0f + velocity) / 2.0f).y, 0.0f, 0.0f);
	}
}

__global__
void visualizationPressure(const int width, const int height, float4* visualizationBuffer,
	const int gridResolution, float* pressureBuffer) {
	int2 id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	if (id.x < width && id.y < height) {
		float pressure = pressureBuffer[id.x + id.y * width];
		visualizationBuffer[id.x + id.y * width] = make_float4((1.0f + pressure) / 2.0f);
	}
}

////ENDOF Kernel functions

#pragma endregion

////Variables for the simulation
//d_ indicates that the variable is a pointer to the device memory

int gridResolution = 512;

int inputVelocityBuffer = 0;
float2* d_velocityBuffer[2];

int inputDensityBuffer = 0;
float4* d_densityBuffer[2];

int inputPressureBuffer = 0;
float* d_pressureBuffer[2];
float* d_divergenceBuffer;

float* d_vorticityBuffer;

size_t problemSize[2];

float4* d_visualizationBufferGPU;
float4* visualizationBufferCPU;

int* d_boundaryBuffer;

size_t visualizationSize[2];

////Blocks, threads
dim3 threadsPerBlock(16, 16);
dim3 numBlocks(gridResolution / threadsPerBlock.x, gridResolution / threadsPerBlock.y);

void initSimulation(int width, int height) {
	// simulation
	problemSize[0] = gridResolution;
	problemSize[1] = gridResolution;

	////Allocating device memory
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

	//boundaryBuffer
	cudaMalloc((void**)&d_boundaryBuffer, sizeof(int)*gridResolution*gridResolution);

	// visualization
	visualizationSize[0] = width;
	visualizationSize[1] = height;

	//CPU - this is where we copy back the data from the GPU
	visualizationBufferCPU = new float4[width * height];

	//GPU visualizationBuffer
	cudaMalloc((void**)&d_visualizationBufferGPU, sizeof(float4)*width*height);
}

void resetSimulationHost() {
	resetSimulation<<<numBlocks, threadsPerBlock>>>(
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		d_pressureBuffer[inputPressureBuffer],
		d_densityBuffer[inputDensityBuffer],
		d_boundaryBuffer
		);
}

void resetPresssure() {
	resetSimulation<<<numBlocks, threadsPerBlock>>>(
		gridResolution,
		d_velocityBuffer[(inputVelocityBuffer + 1) % 2],
		d_pressureBuffer[inputPressureBuffer],
		d_densityBuffer[(inputDensityBuffer + 1) % 2],
		d_boundaryBuffer
		);
}

void simulateAdvection() {
	advection<<<numBlocks, threadsPerBlock>>>(
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		d_velocityBuffer[(inputVelocityBuffer + 1) % 2],
		d_boundaryBuffer
		);
	inputVelocityBuffer = (inputVelocityBuffer + 1) % 2;
}


void simulateVorticity() {
	vorticity<<<numBlocks, threadsPerBlock>>>(
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		d_vorticityBuffer,
		d_boundaryBuffer
		);

	addVorticity<<<numBlocks, threadsPerBlock>>>(
		gridResolution,
		d_vorticityBuffer,
		d_velocityBuffer[inputVelocityBuffer],
		d_boundaryBuffer
		);
}

void simulateDiffusion() {
	for (int i = 0; i < 10; ++i) {
		diffusion<<<numBlocks, threadsPerBlock>>>(
			gridResolution,
			d_velocityBuffer[inputVelocityBuffer],
			d_velocityBuffer[(inputVelocityBuffer + 1) % 2],
			d_boundaryBuffer
			);

		inputVelocityBuffer = (inputVelocityBuffer + 1) % 2;
	}
}

void projection() {
	divergence<<<numBlocks, threadsPerBlock>>>(
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		d_divergenceBuffer,
		d_boundaryBuffer
		);

	resetPresssure();

	for (int i = 0; i < 10; ++i) {
		pressureJacobi<<<numBlocks, threadsPerBlock>>>(
			gridResolution,
			d_pressureBuffer[inputPressureBuffer],
			d_pressureBuffer[(inputPressureBuffer + 1) % 2],
			d_divergenceBuffer,
			d_boundaryBuffer
			);


		inputPressureBuffer = (inputPressureBuffer + 1) % 2;
	}

	projection<<<numBlocks, threadsPerBlock>>>(
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		d_pressureBuffer[inputPressureBuffer],
		d_velocityBuffer[(inputVelocityBuffer + 1) % 2],
		d_boundaryBuffer
		);

	inputVelocityBuffer = (inputVelocityBuffer + 1) % 2;
}

void simulateDensityAdvection() {
	advectionDensity<<<numBlocks, threadsPerBlock>>>(
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		d_densityBuffer[inputDensityBuffer],
		d_densityBuffer[(inputDensityBuffer + 1) % 2],
		d_boundaryBuffer
		);

	inputDensityBuffer = (inputDensityBuffer + 1) % 2;
}

void addForce(int x, int y, float2 force, float4 densityColor, int width, int height) {
	float fx = (float)x / width;
	float fy = (float)y / height;

	addForce<<<numBlocks, threadsPerBlock>>>(
		fx,
		fy,
		force,
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		densityColor,
		d_densityBuffer[inputDensityBuffer]
		);
}
/*void addForcetoSide(float2 force, float4 densityColor, int width, int height) {

	addForcetoSide << <numBlocks, threadsPerBlock >> >(
		force,
		gridResolution,
		d_velocityBuffer[inputVelocityBuffer],
		densityColor,
		d_densityBuffer[inputDensityBuffer]
		);
}*/

void simulationStep() {
	simulateAdvection();
	simulateDiffusion();
	simulateVorticity();
	projection();
	simulateDensityAdvection();
}

void visualizationStep(int visualizationMethod, int width, int height) {
	switch (visualizationMethod) {
	case 0:
		visualizationDensity<<<numBlocks, threadsPerBlock>>>(
			width,
			height,
			d_visualizationBufferGPU,
			gridResolution,
			d_densityBuffer[inputDensityBuffer]
			);
		break;
	case 1:
		visualizationVelocity<<<numBlocks, threadsPerBlock>>>(
			width,
			height,
			d_visualizationBufferGPU,
			gridResolution,
			d_velocityBuffer[inputVelocityBuffer]
			);

		break;
	case 2:
		visualizationPressure<<<numBlocks, threadsPerBlock>>>(
			width,
			height,
			d_visualizationBufferGPU,
			gridResolution,
			d_pressureBuffer[inputPressureBuffer]
			);
		break;

	}

	//Copying the visualization data back form the device, than drawing it
	cudaMemcpy(visualizationBufferCPU, d_visualizationBufferGPU, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
	glDrawPixels(width, height, GL_RGBA, GL_FLOAT, visualizationBufferCPU);
}