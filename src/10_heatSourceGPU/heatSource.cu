#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include "cuda-samples-master/Common/helper_cuda.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "device_functions.h"
#include "../common_book/book.h"

#define REAL double
#define MAX_NUMBER 100000
#define ITER_TIME 1000
#define SPEED 0.33f

extern std::vector<std::vector<int>> gAdjInfo; // 存储的是每个三角形的邻接三角形（邻接指的是两个三角形的包围盒相交）
std::vector<std::vector<int>> adjoinTriangles; // 三角形的邻接三角形
extern std::vector<REAL> gIntensity[2];
extern int currentPass; // 用这个来表示当前 gIntensity 是第几个，在 0 和 1 之间交替
extern std::vector<int> gSources; // 热源点

struct TriangleGPU {
	REAL intensity = 0;
	REAL out = 0;
	int adjInfoSize;
	int adj[20];
};

struct TriangleAdjoin {
	bool isConstant = false;
	REAL intensity = 0;
	int adj[3];
};

//struct TEST {
//	float intensity = 0;
//	// REAL out = 0;
//	int adjInfoSize;
//	//int adj[20];
//};

//__constant__ TEST test[1];

//__constant__ TriangleGPU triangle[40000];
//__device__ TriangleGPU *triangle;

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void heatAdd(TriangleGPU* d, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		d[i].out = d[i].intensity;
		for (int j = 0; j < d[i].adjInfoSize; j++) {
			int tj = d[i].adj[j];
			d[i].out += d[tj].intensity;
		}

		d[i].out /= REAL(d[i].adjInfoSize + 1);
	}
};

__global__ void heatCalculate(TriangleAdjoin* d, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		for (int m = 0; m < ITER_TIME; m++) {
			REAL out = d[i].intensity;
			for (int j = 0; j < 3; j++) {
				int index = d[i].adj[j];
				out += SPEED * (d[index].intensity - d[i].intensity);
			}

			__syncthreads();

			d[i].intensity = out;
			
			__syncthreads();

			if (d[i].isConstant) {
				d[i].intensity = 1;
			}
		}
	}
};

__global__ void heatCalculate2(TriangleAdjoin* d, TriangleAdjoin* o, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		o[i].intensity = d[i].intensity;
		for (int j = 0; j < 3; j++) {
			int index = d[i].adj[j];
			o[i].intensity += SPEED * (d[index].intensity - d[i].intensity);
		}
	}
};

//__global__ void heatAddConstant(int numElements)
//{
//	int i = blockDim.x * blockIdx.x + threadIdx.x;
//
//	if (i < numElements)
//	{
//		REAL out = triangle[i].intensity;
//		for (int j = 0; j < triangle[i].adjInfoSize; j++) {
//			int tj = triangle[i].adj[j];
//			out += triangle[tj].intensity;
//		}
//
//		out /= REAL(triangle[i].adjInfoSize + 1);
//
//		__syncthreads();
//
//		triangle[i].intensity = out;
//	}
//};
//
//extern "C" int readyGPUdata()
//{
//	cudaError_t err = cudaSuccess;
//	int numElements = gIntensity[currentPass].size();
//
//	size_t size = numElements * sizeof(TriangleGPU);
//	
//	// 准备GPU数据
//	err = cudaMalloc((void**)&triangle, size);
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to allocate device vector d_0 (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	// 准备 CPU 数据
//	TriangleGPU* h = (TriangleGPU*)malloc(size);
//
//
//	// Verify that allocations succeeded
//	if (h == NULL)
//	{
//		fprintf(stderr, "Failed to allocate host vectors!\n");
//		exit(EXIT_FAILURE);
//	}
//
//	for (int i = 0; i < numElements; i++) {
//		h[i].intensity = gIntensity[currentPass][i];
//		h[i].adjInfoSize = 0;
//		for (int j = 0; j < 20; j++) {
//			if (j >= gAdjInfo[i].size()) {
//				break;
//			}
//			h[i].adj[j] = gAdjInfo[i][j];
//			h[i].adjInfoSize++;
//		}
//	}
//	
//	//err = cudaMemcpyToSymbol(triangle, h, size, cudaMemcpyHostToDevice);
//	err = cudaMemcpy(triangle, h, size, cudaMemcpyHostToDevice);
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to copy vector h from host to device (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	return 0;
//}

//extern "C" int doPropogateGPUConstant()
//{
//	// 启动定时器
//	cudaEvent_t start, stop;
//	float elapsedTime;
//	HANDLE_ERROR(cudaEventCreate(&start));
//	HANDLE_ERROR(cudaEventCreate(&stop));
//	HANDLE_ERROR(cudaEventRecord(start, 0));
//
//	// Error code to check return values for CUDA calls
//	cudaError_t err = cudaSuccess;
//
//	// Print the vector length to be used, and compute its size
//	int numElements = gIntensity[currentPass].size();
//	int prevPass = currentPass;
//	currentPass = 1 - currentPass;
//
//	size_t size = numElements * sizeof(TriangleGPU);
//
//	// 进行 GPU 计算
//	// Launch the Vector Add CUDA Kernel
//	int threadsPerBlock = 1024;
//	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
//
//	//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
//	heatAddConstant << <blocksPerGrid, threadsPerBlock >> > (numElements);
//
//	err = cudaGetLastError();
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to launch heatAdd kernel (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	// 将 GPU 的结果拷贝回 CPU 中
//	// Copy the device result vector in device memory to the host result vector in host memory.
//
//	TriangleGPU* h = (TriangleGPU*)malloc(size);
//	err = cudaMemcpy(h, triangle, size, cudaMemcpyDeviceToHost);
//	//err = cudaMemcpy(h, triangle, size, cudaMemcpyDeviceToHost);
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to copy vector triangle from device to host (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	for (int i = 0; i < numElements; i++) {
//		gIntensity[currentPass][i] = h[i].intensity;
//		//printf("the result of %d, %f\n", i, gIntensity[currentPass][i]);
//	}
//	for (int i = 0; i < gSources.size(); i++) {
//		gIntensity[currentPass][gSources[i]] = 1.0;
//	}
//
//	//printf("###################\n");
//
//	// 释放空间
//	// Free device global memory
//	//err = cudaFree(d);
//
//	//if (err != cudaSuccess)
//	//{
//	//	fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
//	//	exit(EXIT_FAILURE);
//	//}
//
//	//err = cudaFree(d_B);
//
//	//if (err != cudaSuccess)
//	//{
//	//	fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
//	//	exit(EXIT_FAILURE);
//	//}
//
//	//err = cudaFree(d_C);
//
//	//if (err != cudaSuccess)
//	//{
//	//	fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
//	//	exit(EXIT_FAILURE);
//	//}
//
//	// Free host memory
//	free(h);
//	//free(h_B);
//	//free(h_C);
//
//	//printf("Done\n");
//
//	// 计算时间
//	HANDLE_ERROR(cudaEventRecord(stop, 0));
//	HANDLE_ERROR(cudaEventSynchronize(stop));
//	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
//	printf("Time taken: %3.1f ms\n", elapsedTime);
//
//	return 0;
//}

extern "C" int doPropogateGPU()
{
	// 启动定时器
	cudaEvent_t start, stop;
	float elapsedTime;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = gIntensity[currentPass].size();
	int prevPass = currentPass;
	currentPass = 1 - currentPass;

	size_t size = numElements * sizeof(TriangleGPU);

	// 准备 CPU 数据
	//// Allocate the host input vector h
	//REAL* h = (REAL*)malloc(size);
	TriangleGPU* h = (TriangleGPU *)malloc(size);

	// Verify that allocations succeeded
	if (h == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < numElements; i++){
		//h[i] = gIntensity[prevPass][i];
		h[i].intensity = gIntensity[prevPass][i];
		h[i].adjInfoSize = 0;
		if (gAdjInfo[i].size() > 20) {
			//printf("right\n");
		}
		for (int j = 0; j < 20; j++) {
			if (j >= gAdjInfo[i].size()) {
				break;
			}
			h[i].adj[j] = gAdjInfo[i][j];
			h[i].adjInfoSize++;
		}
	}

	// 在GPU上创建内存
	// Allocate the device input vector d

	TriangleGPU* d = NULL;
	err = cudaMalloc((void**)&d, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_0 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector d_0
	//REAL* d_1 = NULL;
	//err = cudaMalloc((void**)&d_1, size);

	//if (err != cudaSuccess)
	//{
	//	fprintf(stderr, "Failed to allocate device vector d_1 (error code %s)!\n", cudaGetErrorString(err));
	//	exit(EXIT_FAILURE);
	//}

	// 将 CPU 内的值拷贝到 GPU 中
	// Copy the host input vectors A and B in host memory to the device input vectors in device memory
	//printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 进行 GPU 计算
	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 1024;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	heatAdd << <blocksPerGrid, threadsPerBlock >> > (d, numElements);

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch heatAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 将 GPU 的结果拷贝回 CPU 中
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	//printf("Copy output data from the CUDA device to the host memory\n");
	//err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	free(h);
	h = (TriangleGPU*)malloc(size);
	err = cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < numElements; i++) {
		gIntensity[currentPass][i] = h[i].out;
		//printf("the result of %d, %f\n", i, gIntensity[currentPass][i]);
	}
	for (int i = 0; i < gSources.size(); i++) {
		gIntensity[currentPass][gSources[i]] = 1.0;
	}

	//printf("###################\n");

	// 释放空间
	// Free device global memory
	err = cudaFree(d);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//err = cudaFree(d_B);

	//if (err != cudaSuccess)
	//{
	//	fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
	//	exit(EXIT_FAILURE);
	//}

	//err = cudaFree(d_C);

	//if (err != cudaSuccess)
	//{
	//	fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
	//	exit(EXIT_FAILURE);
	//}

	// Free host memory
	free(h);
	//free(h_B);
	//free(h_C);

	//printf("Done\n");

	// 计算时间
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time taken: %3.1f ms\n", elapsedTime);

	return 0;
}

extern "C" int doPropogateGPU_New()
{
	// 启动定时器
	cudaEvent_t start, stop;
	float elapsedTime;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = gIntensity[currentPass].size();
	int prevPass = currentPass;
	currentPass = 1 - currentPass;

	size_t size = numElements * sizeof(TriangleAdjoin);

	// 准备 CPU 数据
	//// Allocate the host input vector h
	//REAL* h = (REAL*)malloc(size);
	TriangleAdjoin* h = (TriangleAdjoin*)malloc(size);

	// Verify that allocations succeeded
	if (h == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < numElements; i++) {
		//h[i] = gIntensity[prevPass][i];
		h[i].intensity = gIntensity[prevPass][i];
		h[i].isConstant = false;
		for (int j = 0; j < 3; j++) {
			h[i].adj[j] = adjoinTriangles[i][j];
		}
	}
	for (int i = 0; i < gSources.size(); i++) {
		h[gSources[i]].isConstant = true;
	}

	// 在GPU上创建内存
	// Allocate the device input vector d

	TriangleAdjoin* d = NULL;
	err = cudaMalloc((void**)&d, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector d_0 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//TriangleAdjoin* d_o = NULL;
	//err = cudaMalloc((void**)&d_o, size);

	//if (err != cudaSuccess)
	//{
	//	fprintf(stderr, "Failed to allocate device vector d_0 (error code %s)!\n", cudaGetErrorString(err));
	//	exit(EXIT_FAILURE);
	//}

	// Allocate the device input vector d_0
	//REAL* d_1 = NULL;
	//err = cudaMalloc((void**)&d_1, size);

	//if (err != cudaSuccess)
	//{
	//	fprintf(stderr, "Failed to allocate device vector d_1 (error code %s)!\n", cudaGetErrorString(err));
	//	exit(EXIT_FAILURE);
	//}

	// 将 CPU 内的值拷贝到 GPU 中
	// Copy the host input vectors A and B in host memory to the device input vectors in device memory
	//printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//err = cudaMemcpy(d_o, h, size, cudaMemcpyHostToDevice);

	//if (err != cudaSuccess)
	//{
	//	fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
	//	exit(EXIT_FAILURE);
	//}

	// 进行 GPU 计算
	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 1024;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	heatCalculate << <blocksPerGrid, threadsPerBlock >> > (d, numElements);
	//for (int i = 0; i < ITER_TIME; i++) {
		//heatCalculate2 << <blocksPerGrid, threadsPerBlock >> > (d, d_o, numElements);
		//swap(d, d_o);

		//free(h);
		//h = (TriangleAdjoin*)malloc(size);
		//err = cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);
		//for (int i = 0; i < numElements; i++) {
		//	printf("the result of %d, %f\n", i, h[i].intensity);
		//}
	//}

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch heatAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 将 GPU 的结果拷贝回 CPU 中
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	//printf("Copy output data from the CUDA device to the host memory\n");
	//err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	free(h);
	h = (TriangleAdjoin*)malloc(size);
	err = cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < numElements; i++) {
		gIntensity[currentPass][i] = h[i].intensity;
		//printf("the result of %d, %f\n", i, gIntensity[currentPass][i]);
	}
	for (int i = 0; i < gSources.size(); i++) {
		gIntensity[currentPass][gSources[i]] = 1.0;
	}

	//printf("###################\n");

	// 释放空间
	// Free device global memory
	err = cudaFree(d);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//err = cudaFree(d_B);

	//if (err != cudaSuccess)
	//{
	//	fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
	//	exit(EXIT_FAILURE);
	//}

	//err = cudaFree(d_C);

	//if (err != cudaSuccess)
	//{
	//	fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
	//	exit(EXIT_FAILURE);
	//}

	// Free host memory
	free(h);
	//free(h_B);
	//free(h_C);

	//printf("Done\n");

	// 计算时间
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Do %d Times and Time taken: %3.1f ms\n", ITER_TIME, elapsedTime);

	return 0;
}