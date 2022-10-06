/*
 * @Author: Bingyang Jin
 * @Date: 2022-10-06 15:23:51
 * @Editor: Bingyang Jin
 * @FilePath: /src/01_drawJuliaSetGPU/drawJuliaSetGPU.cuh
 * @Description: create the file
 */

#include<iostream>
#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include"..\Common\complexNumber.h"
#include"..\Common\PNGManager.h"

__device__ double julia(int x, int y) {

	//printf("x=%d, y=%d\n", x, y);
	
	const double scale = 1.5;
	const double dim = 1000;
	const double iteration = 150;
	const double end = 5000;
	COMMON_LFK::ComplexNumberGPU<double> c(-0.8, 0.156);

	double jx = scale * (dim / 2 - x) / (dim / 2);
	double jy = scale * (dim / 2 - y) / (dim / 2);

	COMMON_LFK::ComplexNumberGPU<double> a(jx, jy);
	double max = 0;
	for (int i = 0; i < iteration; i++) {
		a = a * a + c;
		double aMag = a.magnitude2();
		if (aMag > end) {
			return 0;
		}
		if (aMag > max) {
			max = aMag;
		}
	}

	return 1.0 * (end - max) / end;
}

__global__ void kernel(unsigned char* rgb) {
	const double intensity = 6;
	const double red = 0.6;
	const double green = 0.15;
	const double blue = 1.0;

	int x = blockIdx.x;
	int y = blockIdx.y;
	int pixelindex = x + y * gridDim.x;

	double temp = julia(x, y);
	// map
	double cut = 0.9993;
	double mapCut = 0.005;
	if (temp < cut) {
		temp = temp / cut * mapCut;
	}
	else {
		temp = (temp - cut) / (1 - cut) * (1 - mapCut) + mapCut;
	}

	double tempR = temp * temp * temp * temp * temp;
	double tempG = temp * temp * temp;
	double tempB = temp * (1 - temp) * (1 - temp);

	int R = tempR * red * intensity * 255;
	int G = tempG * green * intensity * 255;
	int B = tempB * blue * intensity * 255;

	if (R > 255) { R = 255; }
	else if (R < 0) { R = 0; }
	if (G > 255) { G = 255; }
	else if (G < 0) { G = 0; }
	if (B > 255) { B = 255; }
	else if (B < 0) { B = 0; }

	//printf("%d, %d, %d\n", &R, &G, &B);

	rgb[pixelindex * 3] = R;
	rgb[pixelindex * 3 + 1] = G;
	rgb[pixelindex * 3 + 2] = B;
}

extern "C" int drawJuliaSetGPU() {

	unsigned char* png = new unsigned char[1000 * 1000 * 3];
	//unsigned char* png;

	unsigned char* dev_map;

	cudaMalloc(
		(void**)&dev_map,
		1000 * 1000 * 3 * sizeof(unsigned char)
	);

	dim3 grid(1000, 1000);

	kernel << <grid, 1 >> > (dev_map);

	cudaMemcpy(png, dev_map, 1000 * 1000 * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	COMMON_LFK::PNGManager pngManager(1000, 1000, png);
	pngManager.writePNG("D:\\test.png");

	return 0;
}