#include"drawJuliaSet.h"

GPU_CUDA_L::DrawJuliaSet::DrawJuliaSet(int _dim)
{
	dim = _dim;
	rgb = new unsigned char[dim * dim * 3];
	c.setReal(-0.8);
	c.setImaginary(0.156);
}

GPU_CUDA_L::DrawJuliaSet::DrawJuliaSet(int _dim, double _scale, double real, double imaginary)
{
	dim = _dim;
	rgb = new unsigned char[dim * dim * 3];
	scale = _scale;
	c.setReal(real);
	c.setImaginary(imaginary);
}

GPU_CUDA_L::DrawJuliaSet::~DrawJuliaSet()
{
	delete[] rgb;
}

void GPU_CUDA_L::DrawJuliaSet::setIteration(int _iteration)
{
	iteration = _iteration;
}

void GPU_CUDA_L::DrawJuliaSet::setBaseColor(unsigned char r, unsigned char g, unsigned char b)
{
	red = r;
	green = g;
	blue = b;
}

void GPU_CUDA_L::DrawJuliaSet::setScale(double _scale)
{
	scale = _scale;
}

void GPU_CUDA_L::DrawJuliaSet::setEnd(int _end)
{
	end = _end;
	if (end < 0) {
		std::cout << "Error occurs because the ending must be positive." << std::endl;
		exit(1);
	}
}

void GPU_CUDA_L::DrawJuliaSet::setConstantComplexNumber(double real, double imaginary)
{
	c.setReal(real);
	c.setImaginary(imaginary);
}

void GPU_CUDA_L::DrawJuliaSet::draw()
{
	for (int y = 0; y < dim; y++) {
		for (int x = 0; x < dim; x++) {
			int pixelindex = y * dim + x;

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
			double tempB = temp * ( 1 -temp )* ( 1 -temp);

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
	}
}

unsigned char * GPU_CUDA_L::DrawJuliaSet::getRGB()
{
	return rgb;
}

double GPU_CUDA_L::DrawJuliaSet::julia(int x, int y)
{
	double jx = scale * (dim / 2 - x) / (dim / 2);
	double jy = scale * (dim / 2 - y) / (dim / 2);

	COMMON_LFK::ComplexNumber<double> a(jx, jy);
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