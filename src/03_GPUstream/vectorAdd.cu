/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /**
  * Vector addition: C = A + B.
  *
  * This sample is a very basic sample that implements element by element
  * vector addition. It is the same as the sample illustrating Chapter 2
  * of the programming guide with some additions like error checking.
  */

#include <stdio.h>

  // For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
// #include <helper_cuda.h>
#include "cuda-samples-master/Common/helper_cuda.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "device_functions.h"
#include "../common_book/book.h"

#define REAL double

#define START_GPU {\
cudaEvent_t     start, stop;\
float   elapsedTime;\
checkCudaErrors(cudaEventCreate(&start)); \
checkCudaErrors(cudaEventCreate(&stop));\
checkCudaErrors(cudaEventRecord(start, 0));\

#define END_GPU \
checkCudaErrors(cudaEventRecord(stop, 0));\
checkCudaErrors(cudaEventSynchronize(stop));\
checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop)); \
printf("GPU Time used:  %3.1f ms\n", elapsedTime);\
checkCudaErrors(cudaEventDestroy(start));\
checkCudaErrors(cudaEventDestroy(stop));}

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const REAL *A, const REAL *B, REAL *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		//C[i] = A[i] + B[i];
		C[i] = atan(A[i]) / (fabs(sin(fabs(B[i]) + 0.0001)) + 0.1);

	}
}

extern double *gA, *gB, *gC;
int main_cpu();

// 老师最初的计算函数
//int origin()
//{
//	//main_cpu();
//
//	// Error code to check return values for CUDA calls
//	cudaError_t err = cudaSuccess;
//
//	// Print the vector length to be used, and compute its size
//	//int numElements = 50000;
//	int numElements = 50000000;
//
//	size_t size = numElements * sizeof(REAL);
//	printf("[Vector addition of %d elements]\n", numElements);
//
//#if 1
//	// Allocate the host input vector A
//	REAL *h_A = (REAL *)malloc(size);
//
//	// Allocate the host input vector B
//	REAL *h_B = (REAL *)malloc(size);
//
//	// Allocate the host output vector C
//	REAL *h_C = (REAL *)malloc(size);
//
//	// Verify that allocations succeeded
//	if (h_A == NULL || h_B == NULL || h_C == NULL)
//	{
//		fprintf(stderr, "Failed to allocate host vectors!\n");
//		exit(EXIT_FAILURE);
//	}
//
//	// Initialize the host input vectors
//	for (int i = 0; i < numElements; ++i)
//	{
//		h_A[i] = rand() / (REAL)RAND_MAX;
//		h_B[i] = rand() / (REAL)RAND_MAX;
//	}
//#else
//	REAL *h_A = gA;
//	REAL *h_B = gB;
//	REAL *h_C = gC;
//#endif
//
//	// Allocate the device input vector A
//	REAL *d_A = NULL;
//	err = cudaMalloc((void **)&d_A, size);
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	// Allocate the device input vector B
//	REAL *d_B = NULL;
//	err = cudaMalloc((void **)&d_B, size);
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	// Allocate the device output vector C
//	REAL *d_C = NULL;
//	err = cudaMalloc((void **)&d_C, size);
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	// Copy the host input vectors A and B in host memory to the device input vectors in
//	// device memory
//	printf("Copy input data from the host memory to the CUDA device\n");
//	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	// Launch the Vector Add CUDA Kernel
//	int threadsPerBlock = 1024;
//	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
//
//	START_GPU
//	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
//	vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, numElements);
//	END_GPU
//
//	err = cudaGetLastError();
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	// Copy the device result vector in device memory to the host result vector
//	// in host memory.
//	printf("Copy output data from the CUDA device to the host memory\n");
//	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	printf("###################\n");
//
//	// Verify that the result vector is correct
//	for (int i = 0; i < 20; ++i)
//	{
//		printf("gA=%lf, gB=%lf, gC=%lf\n", h_A[i], h_B[i], h_C[i]);
//	}
//
//#if 0
//	// Verify that the result vector is correct
//	for (int i = 0; i < numElements; ++i)
//	{
//		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
//		{
//			fprintf(stderr, "Result verification failed at element %d!\n", i);
//			exit(EXIT_FAILURE);
//		}
//	}
//#endif
//
//	printf("Test PASSED\n");
//
//	// Free device global memory
//	err = cudaFree(d_A);
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	err = cudaFree(d_B);
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	err = cudaFree(d_C);
//
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//
//	// Free host memory
//	free(h_A);
//	free(h_B);
//	free(h_C);
//
//	printf("Done\n");
//	return 0;
//}

int noStream()
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = 50000000;
	//int numElements = 20000000;
	//int numElements = 10000000;
	//int numElements = 5000000;
	//int numElements = 100000;

	size_t size = numElements * sizeof(REAL);
	//printf("\n", numElements);
	//printf("###################\n");

	printf("[Use NO stream: Vector addition of %d elements]\n", numElements);

	// 启动定时器
	cudaEvent_t start, stop;
	float elapsedTime;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	
	// 在 CPU 上创建内存
	// Allocate the host input vector A
	REAL* h_A = (REAL*)malloc(size);

	// Allocate the host input vector B
	REAL* h_B = (REAL*)malloc(size);

	// Allocate the host output vector C
	REAL* h_C = (REAL*)malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// 在 CPU 上随机数进行计算
	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (REAL)RAND_MAX;
		h_B[i] = rand() / (REAL)RAND_MAX;
	}

	// 在GPU上创建内存
	// Allocate the device input vector A
	REAL* d_A = NULL;
	err = cudaMalloc((void**)&d_A, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	REAL* d_B = NULL;
	err = cudaMalloc((void**)&d_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	REAL* d_C = NULL;
	err = cudaMalloc((void**)&d_C, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 将 CPU 内的值拷贝到 GPU 中
	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 进行 GPU 计算
	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 1024;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, numElements);


	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 将 GPU 的结果拷贝回 CPU 中
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("###################\n");


	// 计算时间
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time taken: %3.1f ms\n", elapsedTime);

	// 释放空间
	// Free device global memory
	err = cudaFree(d_A);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_B);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_C);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Done\n");
	return 0;
}

int singleStream()
{
	cudaDeviceProp prop;
	int whichDevice;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
	if (!prop.deviceOverlap) {
		printf("Device will not handle overlaps, so no speed up from streams\n");
		return 0;
	}

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = 50000000;
	//int numElements = 20000000;
	//int numElements = 10000000;
	//int numElements = 5000000;
	//int numElements = 100000;

	size_t size = numElements * sizeof(REAL);
	//printf("\n", numElements);
	//printf("###################\n");

	printf("[Use NO stream: Vector addition of %d elements]\n", numElements);

	// 启动定时器
	cudaEvent_t start, stop;
	float elapsedTime;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// 创建流
	cudaStream_t stream;
	HANDLE_ERROR(cudaStreamCreate(&stream));

	// 创建页锁定内存
	// Allocate the host input vector A
	REAL *h_A, *h_B, *h_C;

	HANDLE_ERROR(cudaHostAlloc((void**)&h_A, size, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_B, size, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_C, size, cudaHostAllocDefault));

	// 在 CPU 上随机数进行计算
	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (REAL)RAND_MAX;
		h_B[i] = rand() / (REAL)RAND_MAX;
	}

	// 在GPU上创建内存
	// Allocate the device input vector A
	REAL* d_A = NULL;
	err = cudaMalloc((void**)&d_A, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	REAL* d_B = NULL;
	err = cudaMalloc((void**)&d_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	REAL* d_C = NULL;
	err = cudaMalloc((void**)&d_C, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 将 CPU 内的值拷贝到 GPU 中
	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 进行 GPU 计算
	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 1024;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, numElements);


	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 将 GPU 的结果拷贝回 CPU 中
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("###################\n");


	// 计算时间
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time taken: %3.1f ms\n", elapsedTime);

	// 释放空间
	// Free device global memory
	err = cudaFree(d_A);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_B);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_C);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);

	HANDLE_ERROR(cudaStreamDestroy(stream));

	printf("Done\n");
	return 0;
}

int twoStreamDepth()
{
	cudaDeviceProp prop;
	int whichDevice;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
	if (!prop.deviceOverlap) {
		printf("Device will not handle overlaps, so no speed up from streams\n");
		return 0;
	}

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = 50000000;
	//int numElements = 20000000;
	//int numElements = 10000000;
	//int numElements = 5000000;
	//int numElements = 100000;

	size_t size = numElements * sizeof(REAL);
	//printf("\n", numElements);
	//printf("###################\n");

	printf("[Use TWO DEPTH stream: Vector addition of %d elements]\n", numElements);

	// 启动定时器
	cudaEvent_t start, stop;
	float elapsedTime;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// 创建流
	cudaStream_t stream1;
	HANDLE_ERROR(cudaStreamCreate(&stream1));
	cudaStream_t stream2;
	HANDLE_ERROR(cudaStreamCreate(&stream2));

	// 创建页锁定内存
	// Allocate the host input vector A
	REAL* h_A, * h_B, * h_C;

	HANDLE_ERROR(cudaHostAlloc((void**)&h_A, size, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_B, size, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_C, size, cudaHostAllocDefault));

	// 在 CPU 上随机数进行计算
	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (REAL)RAND_MAX;
		h_B[i] = rand() / (REAL)RAND_MAX;
	}

	// 在GPU上创建内存
	// Allocate the device input vector A
	REAL* d_A0 = NULL;
	err = cudaMalloc((void**)&d_A0, size / 2.0);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	REAL* d_B0 = NULL;
	err = cudaMalloc((void**)&d_B0, size / 2.0);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	REAL* d_C0 = NULL;
	err = cudaMalloc((void**)&d_C0, size / 2.0);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector A
	REAL* d_A1 = NULL;
	err = cudaMalloc((void**)&d_A1, size / 2.0);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	REAL* d_B1 = NULL;
	err = cudaMalloc((void**)&d_B1, size / 2.0);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	REAL* d_C1 = NULL;
	err = cudaMalloc((void**)&d_C1, size / 2.0);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}



	printf("Copy input data from the host memory of stream1 to the CUDA device\n");
	// stream1
	// 将 CPU 内的值拷贝到 GPU 中
	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	err = cudaMemcpyAsync(d_A0, h_A, size / 2.0, cudaMemcpyHostToDevice, stream1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyAsync(d_B0, h_B, size / 2.0, cudaMemcpyHostToDevice, stream1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 进行 GPU 计算
	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 1024;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd << <blocksPerGrid, threadsPerBlock, 0 ,stream1 >> > (d_A0, d_B0, d_C0, numElements/2);

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 将 GPU 的结果拷贝回 CPU 中
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpyAsync(h_C, d_C0, size / 2.0, cudaMemcpyDeviceToHost, stream1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}



	// stream2
	// 将 CPU 内的值拷贝到 GPU 中
	err = cudaMemcpyAsync(d_A1, h_A + numElements / 2, size / 2.0, cudaMemcpyHostToDevice, stream2);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyAsync(d_B1, h_B + numElements / 2, size / 2.0, cudaMemcpyHostToDevice, stream2);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 进行 GPU 计算
	// Launch the Vector Add CUDA Kernel

	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd << <blocksPerGrid, threadsPerBlock, 0, stream2 >> > (d_A1, d_B1, d_C1, numElements/2);

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 将 GPU 的结果拷贝回 CPU 中
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpyAsync(h_C + numElements / 2, d_C1, size / 2.0, cudaMemcpyDeviceToHost, stream2);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("###################\n");


	// 计算时间
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time taken: %3.1f ms\n", elapsedTime);

	// 释放空间
	// Free device global memory
	err = cudaFree(d_A0);
	err = cudaFree(d_A1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_B0);
	err = cudaFree(d_B1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_C0);
	err = cudaFree(d_C1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);

	HANDLE_ERROR(cudaStreamDestroy(stream1));
	HANDLE_ERROR(cudaStreamDestroy(stream2));

	printf("Done\n");
	return 0;
}

int twoStreamWidth()
{
	cudaDeviceProp prop;
	int whichDevice;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
	if (!prop.deviceOverlap) {
		printf("Device will not handle overlaps, so no speed up from streams\n");
		return 0;
	}

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = 50000000;
	//int numElements = 20000000;
	//int numElements = 10000000;
	//int numElements = 5000000;
	//int numElements = 100000;

	size_t size = numElements * sizeof(REAL);
	//printf("\n", numElements);
	//printf("###################\n");

	printf("[Use TWO DEPTH stream: Vector addition of %d elements]\n", numElements);

	// 启动定时器
	cudaEvent_t start, stop;
	float elapsedTime;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// 创建流
	cudaStream_t stream1;
	HANDLE_ERROR(cudaStreamCreate(&stream1));
	cudaStream_t stream2;
	HANDLE_ERROR(cudaStreamCreate(&stream2));

	// 创建页锁定内存
	// Allocate the host input vector A
	REAL* h_A, * h_B, * h_C;

	HANDLE_ERROR(cudaHostAlloc((void**)&h_A, size, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_B, size, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_C, size, cudaHostAllocDefault));

	// 在 CPU 上随机数进行计算
	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (REAL)RAND_MAX;
		h_B[i] = rand() / (REAL)RAND_MAX;
	}

	// 在GPU上创建内存
	// Allocate the device input vector A
	REAL* d_A0 = NULL;
	err = cudaMalloc((void**)&d_A0, size / 2.0);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	REAL* d_B0 = NULL;
	err = cudaMalloc((void**)&d_B0, size / 2.0);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	REAL* d_C0 = NULL;
	err = cudaMalloc((void**)&d_C0, size / 2.0);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector A
	REAL* d_A1 = NULL;
	err = cudaMalloc((void**)&d_A1, size / 2.0);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	REAL* d_B1 = NULL;
	err = cudaMalloc((void**)&d_B1, size / 2.0);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	REAL* d_C1 = NULL;
	err = cudaMalloc((void**)&d_C1, size / 2.0);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}



	printf("Copy input data from the host memory of stream1 to the CUDA device\n");
	// stream1
	// 将 CPU 内的值拷贝到 GPU 中
	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	err = cudaMemcpyAsync(d_A0, h_A, size / 2.0, cudaMemcpyHostToDevice, stream1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyAsync(d_B0, h_B, size / 2.0, cudaMemcpyHostToDevice, stream1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// stream2
	// 将 CPU 内的值拷贝到 GPU 中
	printf("Copy input data from the host memory of stream2 to the CUDA device\n");
	err = cudaMemcpyAsync(d_A1, h_A + numElements / 2, size / 2.0, cudaMemcpyHostToDevice, stream2);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyAsync(d_B1, h_B + numElements / 2, size / 2.0, cudaMemcpyHostToDevice, stream2);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	// 进行 GPU 计算
	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 1024;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd << <blocksPerGrid, threadsPerBlock, 0, stream1 >> > (d_A0, d_B0, d_C0, numElements / 2);

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 进行 GPU 计算
	// Launch the Vector Add CUDA Kernel

	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd << <blocksPerGrid, threadsPerBlock, 0, stream2 >> > (d_A1, d_B1, d_C1, numElements / 2);

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 将 GPU 的结果拷贝回 CPU 中
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpyAsync(h_C, d_C0, size / 2.0, cudaMemcpyDeviceToHost, stream1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// 将 GPU 的结果拷贝回 CPU 中
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpyAsync(h_C + numElements / 2, d_C1, size / 2.0, cudaMemcpyDeviceToHost, stream2);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("###################\n");


	// 计算时间
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time taken: %3.1f ms\n", elapsedTime);

	// 释放空间
	// Free device global memory
	err = cudaFree(d_A0);
	err = cudaFree(d_A1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_B0);
	err = cudaFree(d_B1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_C0);
	err = cudaFree(d_C1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);

	HANDLE_ERROR(cudaStreamDestroy(stream1));
	HANDLE_ERROR(cudaStreamDestroy(stream2));

	printf("Done\n");
	return 0;
}

int twoStreamDepthNew()
{
	cudaDeviceProp prop;
	int whichDevice;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
	if (!prop.deviceOverlap) {
		printf("Device will not handle overlaps, so no speed up from streams\n");
		return 0;
	}

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = 50000000;
	//int numElements = 20000000;
	//int numElements = 10000000;
	//int numElements = 5000000;
	//int numElements = 100000;

	int myBlock = numElements / 20;

	size_t size = numElements * sizeof(REAL);
	//printf("\n", numElements);
	//printf("###################\n");

	printf("[Use TWO DEPTH stream: Vector addition of %d elements]\n", numElements);

	// 启动定时器
	cudaEvent_t start, stop;
	float elapsedTime;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// 创建流
	cudaStream_t stream1;
	HANDLE_ERROR(cudaStreamCreate(&stream1));
	cudaStream_t stream2;
	HANDLE_ERROR(cudaStreamCreate(&stream2));

	// 创建页锁定内存
	// Allocate the host input vector A
	REAL* h_A, * h_B, * h_C;

	HANDLE_ERROR(cudaHostAlloc((void**)&h_A, size, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_B, size, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_C, size, cudaHostAllocDefault));

	// 在 CPU 上随机数进行计算
	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (REAL)RAND_MAX;
		h_B[i] = rand() / (REAL)RAND_MAX;
	}

	// 在GPU上创建内存
	// Allocate the device input vector A
	REAL* d_A0 = NULL;
	err = cudaMalloc((void**)&d_A0, myBlock * sizeof(REAL));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	REAL* d_B0 = NULL;
	err = cudaMalloc((void**)&d_B0, myBlock * sizeof(REAL));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	REAL* d_C0 = NULL;
	err = cudaMalloc((void**)&d_C0, myBlock * sizeof(REAL));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector A
	REAL* d_A1 = NULL;
	err = cudaMalloc((void**)&d_A1, myBlock * sizeof(REAL));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	REAL* d_B1 = NULL;
	err = cudaMalloc((void**)&d_B1, myBlock * sizeof(REAL));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	REAL* d_C1 = NULL;
	err = cudaMalloc((void**)&d_C1, myBlock * sizeof(REAL));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for (int o = 0; o < numElements; o += myBlock * 2) {

		// stream1
		printf("Copy input data from the host memory of stream1 to the CUDA device\n");
		// 将 CPU 内的值拷贝到 GPU 中
		// Copy the host input vectors A and B in host memory to the device input vectors in
		// device memory
		err = cudaMemcpyAsync(d_A0, h_A + o, myBlock * sizeof(REAL), cudaMemcpyHostToDevice, stream1);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaMemcpyAsync(d_B0, h_B + o, myBlock * sizeof(REAL), cudaMemcpyHostToDevice, stream1);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// 进行 GPU 计算
		// Launch the Vector Add CUDA Kernel
		int threadsPerBlock = 1024;
		int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

		printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
		vectorAdd << <blocksPerGrid, threadsPerBlock, 0, stream1 >> > (d_A0, d_B0, d_C0, myBlock);

		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// 将 GPU 的结果拷贝回 CPU 中
		// Copy the device result vector in device memory to the host result vector
		// in host memory.
		printf("Copy output data from the CUDA device to the host memory\n");
		err = cudaMemcpyAsync(h_C + o, d_C0, myBlock * sizeof(REAL), cudaMemcpyDeviceToHost, stream1);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}



		// stream2
		// 将 CPU 内的值拷贝到 GPU 中
		err = cudaMemcpyAsync(d_A1, h_A + o + myBlock, myBlock * sizeof(REAL), cudaMemcpyHostToDevice, stream2);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaMemcpyAsync(d_B1, h_B + o + myBlock, myBlock * sizeof(REAL), cudaMemcpyHostToDevice, stream2);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// 进行 GPU 计算
		// Launch the Vector Add CUDA Kernel

		printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
		vectorAdd << <blocksPerGrid, threadsPerBlock, 0, stream2 >> > (d_A1, d_B1, d_C1, myBlock);

		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// 将 GPU 的结果拷贝回 CPU 中
		// Copy the device result vector in device memory to the host result vector
		// in host memory.
		printf("Copy output data from the CUDA device to the host memory\n");
		err = cudaMemcpyAsync(h_C + o + myBlock, d_C1, myBlock * sizeof(REAL), cudaMemcpyDeviceToHost, stream2);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

	}

	printf("###################\n");


	// 计算时间
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time taken: %3.1f ms\n", elapsedTime);

	// 释放空间
	// Free device global memory
	err = cudaFree(d_A0);
	err = cudaFree(d_A1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_B0);
	err = cudaFree(d_B1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_C0);
	err = cudaFree(d_C1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);

	HANDLE_ERROR(cudaStreamDestroy(stream1));
	HANDLE_ERROR(cudaStreamDestroy(stream2));

	printf("Done\n");
	return 0;
}

int twoStreamWidthNew()
{
	cudaDeviceProp prop;
	int whichDevice;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
	if (!prop.deviceOverlap) {
		printf("Device will not handle overlaps, so no speed up from streams\n");
		return 0;
	}

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = 50000000;
	//int numElements = 20000000;
	//int numElements = 10000000;
	//int numElements = 5000000;
	//int numElements = 100000;

	int myBlock = numElements / 20;

	size_t size = numElements * sizeof(REAL);
	//printf("\n", numElements);
	//printf("###################\n");

	printf("[Use TWO DEPTH stream: Vector addition of %d elements]\n", numElements);

	// 启动定时器
	cudaEvent_t start, stop;
	float elapsedTime;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// 创建流
	cudaStream_t stream1;
	HANDLE_ERROR(cudaStreamCreate(&stream1));
	cudaStream_t stream2;
	HANDLE_ERROR(cudaStreamCreate(&stream2));

	// 创建页锁定内存
	// Allocate the host input vector A
	REAL* h_A, * h_B, * h_C;

	HANDLE_ERROR(cudaHostAlloc((void**)&h_A, size, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_B, size, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_C, size, cudaHostAllocDefault));

	// 在 CPU 上随机数进行计算
	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (REAL)RAND_MAX;
		h_B[i] = rand() / (REAL)RAND_MAX;
	}

	// 在GPU上创建内存
	// Allocate the device input vector A
	REAL* d_A0 = NULL;
	err = cudaMalloc((void**)&d_A0, myBlock * sizeof(REAL));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	REAL* d_B0 = NULL;
	err = cudaMalloc((void**)&d_B0, myBlock * sizeof(REAL));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	REAL* d_C0 = NULL;
	err = cudaMalloc((void**)&d_C0, myBlock * sizeof(REAL));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector A
	REAL* d_A1 = NULL;
	err = cudaMalloc((void**)&d_A1, myBlock * sizeof(REAL));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	REAL* d_B1 = NULL;
	err = cudaMalloc((void**)&d_B1, myBlock * sizeof(REAL));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	REAL* d_C1 = NULL;
	err = cudaMalloc((void**)&d_C1, myBlock * sizeof(REAL));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for (int o = 0; o < numElements; o += myBlock * 2) {

		// stream1
		printf("Copy input data from the host memory of stream1 to the CUDA device\n");
		// 将 CPU 内的值拷贝到 GPU 中
		// Copy the host input vectors A and B in host memory to the device input vectors in
		// device memory
		err = cudaMemcpyAsync(d_A0, h_A + o, myBlock * sizeof(REAL), cudaMemcpyHostToDevice, stream1);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaMemcpyAsync(d_B0, h_B + o, myBlock * sizeof(REAL), cudaMemcpyHostToDevice, stream1);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// stream2
		// 将 CPU 内的值拷贝到 GPU 中
		err = cudaMemcpyAsync(d_A1, h_A + o + myBlock, myBlock * sizeof(REAL), cudaMemcpyHostToDevice, stream2);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaMemcpyAsync(d_B1, h_B + o + myBlock, myBlock * sizeof(REAL), cudaMemcpyHostToDevice, stream2);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// 进行 GPU 计算
		// Launch the Vector Add CUDA Kernel
		int threadsPerBlock = 1024;
		int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

		printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
		vectorAdd << <blocksPerGrid, threadsPerBlock, 0, stream1 >> > (d_A0, d_B0, d_C0, myBlock);

		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// 进行 GPU 计算
		// Launch the Vector Add CUDA Kernel

		printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
		vectorAdd << <blocksPerGrid, threadsPerBlock, 0, stream2 >> > (d_A1, d_B1, d_C1, myBlock);

		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// 将 GPU 的结果拷贝回 CPU 中
		// Copy the device result vector in device memory to the host result vector
		// in host memory.
		printf("Copy output data from the CUDA device to the host memory\n");
		err = cudaMemcpyAsync(h_C + o, d_C0, myBlock * sizeof(REAL), cudaMemcpyDeviceToHost, stream1);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// 将 GPU 的结果拷贝回 CPU 中
		// Copy the device result vector in device memory to the host result vector
		// in host memory.
		printf("Copy output data from the CUDA device to the host memory\n");
		err = cudaMemcpyAsync(h_C + o + myBlock, d_C1, myBlock * sizeof(REAL), cudaMemcpyDeviceToHost, stream2);

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

	}

	printf("###################\n");


	// 计算时间
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time taken: %3.1f ms\n", elapsedTime);

	// 释放空间
	// Free device global memory
	err = cudaFree(d_A0);
	err = cudaFree(d_A1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_B0);
	err = cudaFree(d_B1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_C0);
	err = cudaFree(d_C1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);

	HANDLE_ERROR(cudaStreamDestroy(stream1));
	HANDLE_ERROR(cudaStreamDestroy(stream2));

	printf("Done\n");
	return 0;
}

int main() {

	//noStream();
	//singleStream();
	//twoStreamDepth();
	//twoStreamWidth();
	//twoStreamDepthNew();
	twoStreamWidthNew();
	return 0;
}
