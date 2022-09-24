#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "../01_drawJuliaSet/complexNumberGPU.cuh"

__device__ int julia(int x, int y) {
	const double scale = 1.5;
	const double dim = 1000;
	const double iteration = 150;
	const double end = 5000;
	COMMON_LFK::ComplexNumberGPU<double> c(-0.8, 0.156);

	double jx = scale * (dim / 2 - x) / (dim / 2);
	double jy = scale * (dim / 2 - y) / (dim / 2);

	COMMON_LFK::ComplexNumberGPU<double> a(jx, jy);
	double max = -std::numeric_limits<double>::max();
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

	return 1.0*(end - max) / end;
}

__global__ void kernel(unsigned char *rgb) {
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
	double tempB = temp * (1 - temp)* (1 - temp);

	int R = tempR * red * intensity * 255;
	int G = tempG * green * intensity * 255;
	int B = tempB * blue * intensity * 255;

	if (R > 255) { R = 255; }
	else if (R < 0) { R = 0; }
	if (G > 255) { G = 255; }
	else if (G < 0) { G = 0; }
	if (B > 255) { B = 255; }
	else if (B < 0) { B = 0; }

	rgb[pixelindex * 3] = R;
	rgb[pixelindex * 3 + 1] = G;
	rgb[pixelindex * 3 + 2] = B;
}

int main() {
	std::cout << "Begin" << std::endl;


	unsigned char *dev_map;

	cudaMalloc(
		(void**)dev_map,
		1000 * 1000 * 3 * sizeof(unsigned char)
	);

	dim3 grid(100, 100);

	//kernel << <grid, 1 >> > (dev_map);


	return 0;
}