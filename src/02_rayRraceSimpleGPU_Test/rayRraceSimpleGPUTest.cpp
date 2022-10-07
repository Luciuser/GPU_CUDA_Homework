/*
 * @Author: Bingyang Jin
 * @Date: 2022-10-06 15:10:27
 * @Editor: Bingyang Jin
 * @FilePath: /src/01_drawJuliaSetGPU_Test/drawJuliaSetGPUTest.cpp
 * @Description: create the file
 */

#include<iostream>

extern "C" int drawRayDynamic();
extern "C" int drawRayNoConst();
extern "C" int drawRayConst();
extern "C" int drawRayShare();
extern "C" int drawRayThreads();

int main() {
	
	std::cout << "Begin" << std::endl;

	//drawRayDynamic();
	//drawRayNoConst();
	//drawRayConst();
	//drawRayThreads();
	drawRayShare();

	std::cout << "Finish" << std::endl;

	return 0;
}