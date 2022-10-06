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


#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "device_functions.h"
#include "../common_book/book.h"
#include "../common_book/cpu_bitmap.h"
#include "../common_book/cpu_anim.h"

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

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
#define SPHERES 20

//__constant__ Sphere s[SPHERES];

__global__ void AddInGPU(Sphere* s, int number) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (offset < SPHERES) {
        s[offset].x += number;
    }

    __syncthreads();
}

__global__ void kernel( Sphere *s, unsigned char *ptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - DIM/2);
    float   oy = (y - DIM/2);

    float   r=0, g=0, b=0;
    float   maxz = -INF;
    for(int i=0; i<SPHERES; i++) {
        float   n;
        float   t = s[i].hit( ox, oy, &n );
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
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
    unsigned char* output_bitmap;
    Sphere* s;
    //float* dev_inSrc;
    //float* dev_outSrc;
    //float* dev_constSrc;
    CPUAnimBitmap* bitmap;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;
};

void anim_gpu(DataBlock* d, int ticks) {
    HANDLE_ERROR(cudaEventRecord(d->start, 0));
    dim3    blocks(DIM / 16, DIM / 16);
    dim3    threads(16, 16);
    CPUAnimBitmap* bitmap = d->bitmap;

    // since tex is global and bound, we have to use a flag to
    // select which is in/out per iteration
    //volatile bool dstOut = true;
    //for (int i = 0; i < 90; i++) {
    //    for (int j = 0; j < SPHERES; j++) {
    //        d->s[j].x += 0.01;
    //    }
    //}

    AddInGPU << <blocks, threads >> > (d->s, 1);
    kernel << <blocks, threads >> > (d->s, d->output_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(),
        d->output_bitmap,
        bitmap->image_size(),
        cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(d->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(d->stop));
    float   elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
    d->totalTime += elapsedTime;
    ++d->frames;
    printf("Average Time per frame:  %3.1f ms\n", d->totalTime / d->frames);
}

// clean up memory allocated on the GPU
void anim_exit(DataBlock* d) {
    //HANDLE_ERROR(cudaFree(d->dev_inSrc));
    //HANDLE_ERROR(cudaFree(d->dev_outSrc));
    //HANDLE_ERROR(cudaFree(d->dev_constSrc));

    HANDLE_ERROR(cudaEventDestroy(d->start));
    HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main( void ) {
    DataBlock   data;
    CPUAnimBitmap bitmap( DIM, DIM, &data );
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    HANDLE_ERROR( cudaEventCreate( &data.start ) );
    HANDLE_ERROR( cudaEventCreate( &data.stop ) );
   
    int imageSize = bitmap.image_size();
    
    HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap, imageSize ) );
    
    // assume float == 4 chars in size (ie rgba)
    //HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc, imageSize ) );
    //HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc, imageSize ) );
    //HANDLE_ERROR( cudaMalloc( (void**)&data.dev_constSrc, image  Size ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.s, sizeof(Sphere) * SPHERES ) );

    // allocate temp memory, initialize it, copy to
    // memory on the GPU, then free our temp memory
    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
    for (int i=0; i<SPHERES; i++) {
        temp_s[i].r = rnd( 1.0f );
        temp_s[i].g = rnd( 1.0f );
        temp_s[i].b = rnd( 1.0f );
        temp_s[i].x = rnd( 1000.0f ) - 500;
        temp_s[i].y = rnd( 1000.0f ) - 500;
        temp_s[i].z = rnd( 1000.0f ) - 500;
        temp_s[i].radius = rnd( 100.0f ) + 20;
    }
    HANDLE_ERROR( cudaMemcpy( data.s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice ) );

    free( temp_s );

    bitmap.anim_and_exit((void (*)(void*, int))anim_gpu, (void (*)(void*))anim_exit);

    //// generate a bitmap from our sphere data
    //dim3    grids(DIM/16,DIM/16);
    //dim3    threads(16,16);
    //kernel<<<grids,threads>>>( dev_bitmap );

    //// copy our bitmap back from the GPU for display
    //HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
    //                          bitmap.image_size(),
    //                          cudaMemcpyDeviceToHost ) );

    //// get stop time, and display the timing results
    //HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    //HANDLE_ERROR( cudaEventSynchronize( stop ) );
    //float   elapsedTime;
    //HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
    //                                    start, stop ) );
    //printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    //HANDLE_ERROR( cudaEventDestroy( start ) );
    //HANDLE_ERROR( cudaEventDestroy( stop ) );

    //HANDLE_ERROR( cudaFree( dev_bitmap ) );


    //for (int i=0; i<DIM*DIM; i++) {
    //    temp[i] = 0;
    //    int x = i % DIM;
    //    int y = i / DIM;
    //    if ((x>300) && (x<600) && (y>310) && (y<601))
    //        temp[i] = MAX_TEMP;
    //}
    //temp[DIM*100+100] = (MAX_TEMP + MIN_TEMP)/2;
    //temp[DIM*700+100] = MIN_TEMP;
    //temp[DIM*300+300] = MIN_TEMP;
    //temp[DIM*200+700] = MIN_TEMP;
    //for (int y=800; y<900; y++) {
    //    for (int x=400; x<500; x++) {
    //        temp[x+y*DIM] = MIN_TEMP;
    //    }
    //}
    //HANDLE_ERROR( cudaMemcpy( data.dev_constSrc, temp, imageSize, cudaMemcpyHostToDevice ) );    
    //
    //// initialize the input data
    //for (int y=800; y<DIM; y++) {
    //    for (int x=0; x<200; x++) {
    //        temp[x+y*DIM] = MAX_TEMP;
    //    }
    //}
    //HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, temp, imageSize, cudaMemcpyHostToDevice ) );
    //free( temp );
    

}

