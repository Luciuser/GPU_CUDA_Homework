/*
 * @Author: Bingyang Jin
 * @Date: 2022-09-22 11:09:23
 * @Editor: Bingyang Jin
 * @FilePath: /src/01_drawJuliaSet/drawJuliaSet.h
 * @Description: create the file
 */

#ifndef __draw_julia_set__
#define __draw_julia_set__

#include<iostream>
#include"complexNumber.h"
#include"PNGManager.h"

namespace GPU_CUDA_L {

	class DrawJuliaSet {
	public:
		DrawJuliaSet(int _dim);
		DrawJuliaSet(int _dim, double _scale, double real, double imaginary);
		~DrawJuliaSet();

		void setIteration(int _iteration);
		void setBaseColor(unsigned char r, unsigned char g, unsigned char b);
		void setScale(double _scale);
		void setEnd(int _end);
		void setConstantComplexNumber(double real, double imaginary);
		void draw();	// draw julia set with CPU
		unsigned char* getRGB(); // return rgb ptr

	private:
		double julia(int x, int y); // calculate the answer of julia function

		unsigned char *rgb = nullptr; // the bitmap picture, size = dim * dim
		double intensity = 6; // the picture intensity
		double red = 0.6;
		double green = 0.15;
		double blue = 1.0;
		double scale = 1.5; // control the julia picture size
		int dim = 0; // the pixel number of this picture
		int iteration = 200; // iteration number
		int end = 1000;	// iteration end
		COMMON_LFK::ComplexNumber<double> c; // constant number
	};

}
#endif //__draw_julia_set__