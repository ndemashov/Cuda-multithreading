
#include "device_launch_parameters.h"

#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include<cmath>
#include<fstream>
#include<iostream>


#define PI 3.14159265358979323846
#define threads_number 5
#define t 1e-1
#define Ax -0.353
#define Ay 0.3
#define Bx 0.353
#define By 0.3
#define C 3 * PI / 8
#define accuracy 1e-10
#define steps 4282

//		x1 + y * np.cos(3 * np.pi / 2 - f1)  Ax,
//		x2 + y * np.cos(3 * np.pi / 2 + f2) - Bx,
//		y + y * np.sin(3 * np.pi / 2 - f1) - Ay,
//		(f1 + f2) * y + (x2 - x1) - C,
//		y + y * np.sin(3 * np.pi / 2 + f2) - By

__device__ void calcF(const double X[threads_number], double F[threads_number], const int equation_number) {
	switch (equation_number) {
	case 0:
		F[0] = X[0] + X[2] * cos(3 * PI / 2 - X[3]) - Ax;
		break;
	case 1:
		F[1] = X[1] + X[2] * cos(3 * PI / 2 + X[4]) - Bx;
		break;
	case 2:
		F[2] = X[2] + X[2] * sin(3 * PI / 2 - X[3]) - Ay;
		break;
	case 3:
		F[3] = (X[3] + X[4]) * X[2] + (X[1] - X[0]) - C;
		break;
	case 4:
		F[4] = X[2] + X[2] * sin(3 * PI / 2 + X[4]) - By;
		break;
	}
}

__device__ bool isSolution(const double F[threads_number]) {
	for (int i = 0; i < threads_number; ++i) {
		if (fabs(F[i]) > accuracy) {
			return false;
		}
	}
	return true;
}

__global__ void calc(double* XbySteps, unsigned* real_steps) {
	__shared__ double X[threads_number], F[threads_number];
	__shared__ bool solution_found;
	solution_found = false;
	X[threadIdx.x] = 0;
	F[threadIdx.x] = 0;
	while (!solution_found) {
		X[threadIdx.x] = X[threadIdx.x] - t * F[threadIdx.x];
		__syncthreads(); 
		calcF(X, F, threadIdx.x);
		__syncthreads();
		if (0 == threadIdx.x) {
			for (int i = 0; i < threads_number; ++i) {
				XbySteps[threads_number * *real_steps + i] = X[i];
			}
			solution_found = isSolution(F);
		}
		__syncthreads();
		(*real_steps)++;
		__syncthreads();
	}
}

void random_ints(double* a, int M)
{
	int i;
	for (i = 0; i < M; ++i)
		a[i] = rand() % 100;
}

int main(void) {
	double* XbySteps;
	unsigned* real_steps;
	double* dev_XbySteps;
	unsigned* dev_real_steps;
	int size = steps * threads_number * sizeof(double);

	cudaMalloc((void**)&dev_XbySteps, size);
	cudaMalloc((void**)&dev_real_steps, sizeof(unsigned));

	XbySteps = (double*)malloc(size);
	real_steps = (unsigned*)malloc(sizeof(unsigned));

	*real_steps = 0;

	cudaMemcpy(dev_XbySteps, XbySteps, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_real_steps, real_steps, sizeof(unsigned), cudaMemcpyHostToDevice);

	calc << < 1, threads_number >> > (dev_XbySteps, dev_real_steps);

	cudaMemcpy(XbySteps, dev_XbySteps, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(real_steps, dev_real_steps, sizeof(unsigned), cudaMemcpyDeviceToHost);

	std::ofstream fXbySteps("output.txt");


	for (unsigned i = 0; i < *real_steps; i++) {
		for (unsigned j = 0; j < threads_number; ++j) {
			fXbySteps << XbySteps[threads_number * i + j] << ' ';
		}
		fXbySteps << std::endl;
	}

	fXbySteps.close();

	free(XbySteps);
	free(real_steps);
	cudaFree(dev_XbySteps);
	cudaFree(dev_real_steps);

	return 0;

}