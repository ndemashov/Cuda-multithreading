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
#define C 3 * PI / 8
#define accuracy 1e-10
#define steps 5000
#define variable_number 9
#define m 100.0
#define p 2000.0
#define real_time 0.01
#define number_real_time 250
#define image_time 0.005
#define number_image_time 10000

#define Ax -0.353
#define Ay 0.3
#define Bx 0.353
#define By 0.3
#define C 3 * PI / 8

//		x1 + y * np.cos(3 * np.pi / 2 - f1)  Ax,
//		x2 + y * np.cos(3 * np.pi / 2 + f2) - Bx,
//		y + y * np.sin(3 * np.pi / 2 - f1) - Ay,
//		(f1 + f2) * y + (x2 - x1) - C,
//		y + y * np.sin(3 * np.pi / 2 + f2) - By

__device__ void calcF(const double X[variable_number], double F[threads_number], const int equation_number) {
	//printf("%f %f\n", X[6], X[8]);
	switch (equation_number) {
	case 0:
		F[0] = X[0] + X[2] * cos(3 * PI / 2 - X[3]) - X[5];
		break;
	case 1:
		F[1] = X[1] + X[2] * cos(3 * PI / 2 + X[4]) - X[7];
		break;
	case 2:
		F[2] = X[2] + X[2] * sin(3 * PI / 2 - X[3]) - X[6];
		break;
	case 3:
		F[3] = (X[3] + X[4]) * X[2] + (X[1] - X[0]) - C;
		break;
	case 4:
		F[4] = X[2] + X[2] * sin(3 * PI / 2 + X[4]) - X[8];
		break;
	}
}

//case 0:
//	F[0] = X[0] + X[2] * cos(3 * PI / 2 - X[3]) - X[5];
//	break;
//case 1:
//	F[1] = X[1] + X[2] * cos(3 * PI / 2 + X[4]) - X[7];
//	break;
//case 2:
//	F[2] = X[2] + X[2] * sin(3 * PI / 2 - X[3]) - X[6];
//	break;
//case 3:
//	F[3] = (X[3] + X[4]) * X[2] + (X[1] - X[0]) - C;
//	break;
//case 4:
//	F[4] = X[2] + X[2] * sin(3 * PI / 2 + X[4]) - X[8];
//	break;
//	}

__device__ bool isSolution(const double F[threads_number]) {
	for (int i = 0; i < threads_number; ++i) {
		if (fabs(F[i]) > accuracy) {
			return false;
		}
	}
	return true;
}

__global__ void calc(double* XS, double* XbySteps = nullptr, unsigned* real_steps = nullptr) {
	__shared__ double X[threads_number], F[threads_number];
	__shared__ bool solution_found;
	__shared__ unsigned step;
	solution_found = false;
	X[threadIdx.x] = 0;
	F[threadIdx.x] = 0;
	while (!solution_found) {
		XS[threadIdx.x] = XS[threadIdx.x] - t * F[threadIdx.x];
		__syncthreads();
		calcF(XS, F, threadIdx.x);
		//printf(solution_found ? "true" : "false");
		__syncthreads();
		if (0 == threadIdx.x) {
			for (int i = 0; i < threads_number; ++i) {
				//printf("%f ", F[i]);
			}
			//printf("\n");
			if (XbySteps) {
				for (int i = 0; i < variable_number; ++i) {
					XbySteps[variable_number * *real_steps + i] = XS[i];
					//printf("%f ", XbySteps[variable_number * *real_steps + i]);
				}
				(*real_steps)++;
				//printf("%d \n", *real_steps);
				//printf("\n");
			}
			solution_found = isSolution(F);
		}
		__syncthreads();
	}
}


int main(void) {
	double* XS, * dev_XS;
	double* XbySteps;
	unsigned* real_steps;
	double* dev_XbySteps;
	unsigned* dev_real_steps;
	int size = steps * variable_number * sizeof(double);
	XS = (double*)malloc(size);
	XbySteps = (double*)malloc(size);
	real_steps = (unsigned*)malloc(sizeof(unsigned));
	XS[0] = 0; XS[1] = 0; XS[2] = 0.25; XS[3] = PI; XS[4] = PI; XS[5] = -0.353; XS[6] = 0.3; XS[7] = 0.353; XS[8] = 0.3;
	cudaMalloc((void**)&dev_XS, variable_number * sizeof(double));
	cudaMalloc((void**)&dev_XbySteps, size);
	cudaMalloc((void**)&dev_real_steps, sizeof(unsigned));
	*real_steps = 0;
	cudaMemcpy(dev_XbySteps, XbySteps, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_real_steps, real_steps, sizeof(unsigned), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_XS, XS, variable_number * sizeof(double), cudaMemcpyHostToDevice);
	calc << < 1, threads_number >> > (dev_XS, dev_XbySteps, dev_real_steps);
	cudaMemcpy(XbySteps, dev_XbySteps, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(real_steps, dev_real_steps, sizeof(unsigned), cudaMemcpyDeviceToHost);
	std::ofstream fXbySteps("output.txt");

	for (unsigned i = 0; i < *real_steps; i++) {
		for (unsigned j = 0; j < variable_number; ++j) {
			fXbySteps << XbySteps[variable_number * i + j] << ' ';
		}
		fXbySteps << std::endl;
	}

	fXbySteps.close();

	free(XbySteps);
	free(real_steps);
	cudaFree(dev_XbySteps);
	cudaFree(dev_real_steps);

	//double *XS, *dev_XS;
	//// XS = x1, x2, y, f1, f2, Ax, Ay, Bx, By
	//cudaMalloc((void**)&dev_XS, variable_number * sizeof(double));
	//XS = (double*)malloc(variable_number * sizeof(double));
	//XS[0] = 0; XS[1] = 0; XS[2] = 0.25; XS[3] = PI; XS[4] = PI; XS[5] = -0.353; XS[6] = 0.3; XS[7] = 0.353; XS[8] = 0.3;
	//	/*X0 = (0, 0, 0.25, np.pi, np.pi)
	//	P0 = 0.3
	//	V0 = 0
	//	steps = 0
	//	t = 1e-1

	//	Ay = 0.3
	//	By = 0.3
	//	Ax = -0.353
	//	Bx = 0.353
	//	C = 3 * np.pi / 8*/
	//std::ofstream fXbySteps("output.txt");
	//double V0 = 0, P0 = 0.3;

	//for (int n = 0; n < number_real_time + 1; ++n) {
	//	cudaMemcpy(dev_XS, XS, variable_number * sizeof(double), cudaMemcpyHostToDevice);

	//	calc << < 1, threads_number >> > (dev_XS);

	//	cudaMemcpy(XS, dev_XS, variable_number * sizeof(double), cudaMemcpyDeviceToHost);

	//	double length = XS[1] - XS[0];
	//	double V1 = V0 + (1.0 / m) * (p * length - m * 10.0) * real_time;
	//	double V0 = V1;
	//	double P1 = P0 + V0 * real_time;
	//	P0 = P1;
	//	XS[6] = P0;
	//	XS[8] = P0;
	//	for (unsigned i = 0; i < variable_number; ++i) {
	//		fXbySteps << XS[i] << ' ';
	//	}
	//	fXbySteps << std::endl;
	//	std::cout << n << std::endl;
	//}
	//fXbySteps.close();

	//free(XS);
	//cudaFree(dev_XS);

	return 0;

}