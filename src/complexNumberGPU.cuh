/*
 * @Author: Bingyang Jin
 * @Date: 2022-09-22 20:53:36
 * @Editor: Bingyang Jin
 * @FilePath: /src/01_drawJuliaSet/complexNumberGPU.h
 * @Description: create the file
 */

#ifndef __complex_number_gpu__
#define __complex_number_gpu__

#include<iostream>
#include "cublas_v2.h"  

namespace COMMON_LFK {

	template<typename T> class ComplexNumberGPU {
	public:
		ComplexNumberGPU() {}
		ComplexNumberGPU(T real, T imaginary) : r(real), i(imaginary) {}
		~ComplexNumberGPU() {}

		__device__ void setReal(T real) { r = real; }
		__device__ void setImaginary(T imaginary) { i = imaginary; }
		__device__ T real() { return r; }
		__device__ T imaginary() { T i; }
		__device__ T magnitude2() { return r * r + i * i; }
		__device__ ComplexNumberGPU operator*(const ComplexNumberGPU&c) {
			return ComplexNumber(r * c.r - i * c.i, r * c.i + c.r * i);
		}
		__device__ ComplexNumberGPU operator+(const ComplexNumberGPU&c) {
			return ComplexNumber(r + c.r, i + c.i);
		}

	private:
		T r;
		T i;
	};

}
#endif //__complex_number_gpu__