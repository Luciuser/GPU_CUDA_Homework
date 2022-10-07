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

//#include<time.h>
//#include "cuda_runtime.h" 
//#include "device_launch_parameters.h"
//#include "cublas_v2.h"
//#include "device_functions.h"
//#include "../common_book/book.h"
//#include "../common_book/cpu_bitmap.h"
//#include "../common_book/cpu_anim.h"
//
//#define DIM 1024
//
//#define rnd( x ) (x * rand() / RAND_MAX)
//#define INF     2e10f
//
//#define SPHERES 20
//
//struct Sphere {
//    float   r,b,g;
//    float   radius;
//    float   x,y,z;
//    __device__ float hit( float ox, float oy, float *n ) {
//        float dx = ox - x;
//        float dy = oy - y;
//        if (dx*dx + dy*dy < radius*radius) {
//            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
//            *n = dz / sqrtf( radius * radius );
//            return dz + z;
//        }
//        return -INF;
//    }
//};
//
////__constant__ Sphere s[SPHERES];
//
//__global__ void AddInGPU(Sphere* s, int number) {
//    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;
//    int offset = x + y * blockDim.x * gridDim.x;
//
//    if (offset < 3) {
//        s[offset].x += number * offset * 0.3;
//    }
//    else if (offset < 6) {
//        s[offset].y += number * offset * 0.15;
//    }
//    else if (offset < 12) {
//        s[offset].radius += number * offset * 0.01;
//    }
//    else if (offset < SPHERES) {
//        s[offset].b += number * offset * 0.0001;
//        s[offset].g += number * offset * 0.00005;
//    }
//
//    __syncthreads();
//}
//
//__global__ void kernel( Sphere *s, unsigned char *ptr ) {
//    // map from threadIdx/BlockIdx to pixel position
//    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;
//    int offset = x + y * blockDim.x * gridDim.x;
//    float   ox = (x - DIM/2);
//    float   oy = (y - DIM/2);
//
//    float   r=0, g=0, b=0;
//    float   maxz = -INF;
//    for(int i=0; i<SPHERES; i++) {
//        float   n;
//        float   t = s[i].hit( ox, oy, &n );
//        if (t > maxz) {
//            float fscale = n;
//            r = s[i].r * fscale;
//            g = s[i].g * fscale;
//            b = s[i].b * fscale;
//            maxz = t;
//        }
//    } 
//
//    ptr[offset*4 + 0] = (int)(r * 255);
//    ptr[offset*4 + 1] = (int)(g * 255);
//    ptr[offset*4 + 2] = (int)(b * 255);
//    ptr[offset*4 + 3] = 255;
//}
//
//// globals needed by the update routine
//struct DataBlock {
//    unsigned char* output_bitmap;
//    Sphere* s;
//    CPUAnimBitmap* bitmap;
//
//    cudaEvent_t     start, stop;
//    float           totalTime;
//    float           frames;
//};
//
//void anim_gpu(DataBlock* d, int ticks) {
//    HANDLE_ERROR(cudaEventRecord(d->start, 0));
//    dim3    blocks(DIM / 16, DIM / 16);
//    dim3    threads(16, 16);
//    CPUAnimBitmap* bitmap = d->bitmap;
//
//    if ((ticks / 400) % 2 == 0) {
//        AddInGPU << <blocks, threads >> > (d->s, 1);
//    }
//    else {
//        AddInGPU << <blocks, threads >> > (d->s, -1);
//    }
//    kernel << <blocks, threads >> > (d->s, d->output_bitmap);
//
//    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(),
//        d->output_bitmap,
//        bitmap->image_size(),
//        cudaMemcpyDeviceToHost));
//
//    HANDLE_ERROR(cudaEventRecord(d->stop, 0));
//    HANDLE_ERROR(cudaEventSynchronize(d->stop));
//    float   elapsedTime;
//    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
//    d->totalTime += elapsedTime;
//    ++d->frames;
//    printf("Average Time per frame:  %3.1f ms\n", d->totalTime / d->frames);
//}
//
//// clean up memory allocated on the GPU
//void anim_exit(DataBlock* d) {
//    HANDLE_ERROR(cudaEventDestroy(d->start));
//    HANDLE_ERROR(cudaEventDestroy(d->stop));
//}
//
//extern "C" int drawRayDynamic() {
//    DataBlock   data;
//    CPUAnimBitmap bitmap( DIM, DIM, &data );
//    data.bitmap = &bitmap;
//    data.totalTime = 0;
//    data.frames = 0;
//    HANDLE_ERROR( cudaEventCreate( &data.start ) );
//    HANDLE_ERROR( cudaEventCreate( &data.stop ) );
//   
//    int imageSize = bitmap.image_size();
//    
//    HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap, imageSize ) );
//    
//    HANDLE_ERROR( cudaMalloc( (void**)&data.s, sizeof(Sphere) * SPHERES ) );
//
//    // allocate temp memory, initialize it, copy to
//    // memory on the GPU, then free our temp memory
//    srand((unsigned)time(NULL));
//    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
//    for (int i=0; i<SPHERES; i++) {
//        temp_s[i].r = rnd( 1.0f );
//        temp_s[i].g = rnd( 1.0f );
//        temp_s[i].b = rnd( 1.0f );
//        temp_s[i].x = rnd( 1000.0f ) - 500;
//        temp_s[i].y = rnd( 1000.0f ) - 500;
//        temp_s[i].z = rnd( 1000.0f ) - 500;
//        temp_s[i].radius = rnd( 100.0f ) + 20;
//    }
//    HANDLE_ERROR( cudaMemcpy( data.s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice ) );
//
//    free( temp_s );
//
//    bitmap.anim_and_exit((void (*)(void*, int))anim_gpu, (void (*)(void*))anim_exit);
//
//    return 0;
//}