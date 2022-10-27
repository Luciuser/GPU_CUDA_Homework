/*
 * @Author: Bingyang Jin
 * @Date: 2022-10-26 20:35:07
 * @Editor: Bingyang Jin
 * @FilePath: /src/10_heatSourceGPU_Test/heatSourceGPUTest.cpp
 * @Description: create the file
 */

#include<iostream>

//extern "C" int drawWithCPU(int argc, char** argv);
extern int drawWithCPU(int argc, char** argv);


int main(int argc, char** argv) {
	
	std::cout << "Begin" << std::endl;

	drawWithCPU(argc, argv);

	std::cout << "Finish" << std::endl;

	return 0;
}