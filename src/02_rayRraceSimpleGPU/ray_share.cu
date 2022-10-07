/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include<iostream>
#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "device_functions.h"
#include "../common_book/book.h"
#include "../common_book/cpu_bitmap.h"

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

#define SPHERES 1

struct Sphere {
    float   r,b,g;
    float   radius;
    float   x,y,z;
    __device__ float hit( float ox, float oy, float *n ) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        }
        return -INF;
    }
};


__global__ void kernel(Sphere* s, unsigned char* ptr) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ Sphere shared_s[SPHERES];

    int in_offset = threadIdx.x + threadIdx.y * blockDim.x;
    if (in_offset < SPHERES) {
        shared_s[in_offset].x = s[in_offset].x;
        shared_s[in_offset].y = s[in_offset].y;
        shared_s[in_offset].z = s[in_offset].z;
        shared_s[in_offset].r = s[in_offset].r;
        shared_s[in_offset].g = s[in_offset].g;
        shared_s[in_offset].b = s[in_offset].b;
        shared_s[in_offset].radius = s[in_offset].radius;
    }

    __syncthreads();

    float   ox = (x - DIM/2);
    float   oy = (y - DIM/2);

    float   r=0, g=0, b=0;
    float   maxz = -INF;


    for(int i=0; i<SPHERES; i++) {
        float   n;
        float   t = shared_s[i].hit( ox, oy, &n );
        if (t > maxz) {
            float fscale = n;
            r = shared_s[i].r * fscale;
            g = shared_s[i].g * fscale;
            b = shared_s[i].b * fscale;
            maxz = t;
        }
    } 

    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}


// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
    Sphere          *s;
};

extern "C" int drawRayShare() {

    std::cout << "Share SPHERES: " << SPHERES << std::endl;

    int iterTime = 100;
    double sumTime = 0;
    for (int i = 0; i < iterTime; i++) {

        DataBlock   data;
        // capture the start time
        cudaEvent_t     start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, 0));

        CPUBitmap bitmap(DIM, DIM, &data);
        unsigned char* dev_bitmap;
        Sphere* s;


        // allocate memory on the GPU for the output bitmap
        HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap,
            bitmap.image_size()));
        // allocate memory for the Sphere dataset
        HANDLE_ERROR(cudaMalloc((void**)&s,
            sizeof(Sphere) * SPHERES));

        // allocate temp memory, initialize it, copy to
        // memory on the GPU, then free our temp memory
        Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
        for (int i = 0; i < SPHERES; i++) {
            temp_s[i].r = rnd(1.0f);
            temp_s[i].g = rnd(1.0f);
            temp_s[i].b = rnd(1.0f);
            temp_s[i].x = rnd(1000.0f) - 500;
            temp_s[i].y = rnd(1000.0f) - 500;
            temp_s[i].z = rnd(1000.0f) - 500;
            temp_s[i].radius = rnd(100.0f) + 20;
        }
        HANDLE_ERROR(cudaMemcpy(s, temp_s,
            sizeof(Sphere) * SPHERES,
            cudaMemcpyHostToDevice));
        free(temp_s);

        // generate a bitmap from our sphere data
        dim3    grids(DIM / 16, DIM / 16);
        dim3    threads(16, 16);
        kernel << <grids, threads >> > (s, dev_bitmap);

        // copy our bitmap back from the GPU for display
        HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
            bitmap.image_size(),
            cudaMemcpyDeviceToHost));

        // get stop time, and display the timing results
        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float   elapsedTime;
        HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
            start, stop));
        printf("Time to generate:  %3.1f ms\n", elapsedTime);

        sumTime += elapsedTime;

        HANDLE_ERROR(cudaEventDestroy(start));
        HANDLE_ERROR(cudaEventDestroy(stop));

        HANDLE_ERROR(cudaFree(dev_bitmap));
        HANDLE_ERROR(cudaFree(s));

        // display
        //bitmap.display_and_exit();

    }

    printf("Time to generate average:  %3.1f ms\n", sumTime / iterTime);

} 