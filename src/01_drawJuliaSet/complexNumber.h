/*
 * @Author: Bingyang Jin
 * @Date: 2022-09-22 11:19:24
 * @Editor: Bingyang Jin
 * @FilePath: /src/01_drawJuliaSet/complexNumber.h
 * @Description: create the file
 */

#ifndef __complex_number__
#define __complex_number__

#include<iostream>

namespace COMMON_LFK {

	template<typename T> class ComplexNumber {
	public:
		ComplexNumber(){}
		ComplexNumber(T real, T imaginary) : r(real), i(imaginary) {}
		~ComplexNumber() {}

		void setReal(T real) { r = real; }
		void setImaginary(T imaginary) { i = imaginary; }
		T real() { return r; }
		T imaginary() { T i; }
		T magnitude2() { return r * r + i * i; }
		ComplexNumber operator*(const ComplexNumber&c) {
			return ComplexNumber(r * c.r - i * c.i, r * c.i + c.r * i);
		}
		ComplexNumber operator+(const ComplexNumber&c) {
			return ComplexNumber(r + c.r, i + c.i);
		}

	private:
		T r;
		T i;
	};

	//template<typename T> class ComplexNumberGPU {
	//public:
	//	ComplexNumberGPU() {}
	//	ComplexNumberGPU(T real, T imaginary) : r(real), i(imaginary) {}
	//	~ComplexNumberGPU() {}

	//	__device__ void setReal(T real) { r = real; }
	//	__device__ void setImaginary(T imaginary) { i = imaginary; }
	//	__device__ T real() { return r; }
	//	__device__ T imaginary() { T i; }
	//	__device__ T magnitude2() { return r * r + i * i; }
	//	__device__ ComplexNumber operator*(const ComplexNumber&c) {
	//		return ComplexNumber(r * c.r - i * c.i, r * c.i + c.r * i);
	//	}
	//	__device__ ComplexNumber operator+(const ComplexNumber&c) {
	//		return ComplexNumber(r + c.r, i + c.i);
	//	}

	//private:
	//	T r;
	//	T i;
	//};

}
#endif //__complex_number__