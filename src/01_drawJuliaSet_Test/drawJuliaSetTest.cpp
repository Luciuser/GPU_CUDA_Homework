/*
 * @Author: Bingyang Jin
 * @Date: 2022-09-22 11:09:23
 * @Editor: Bingyang Jin
 * @FilePath: /src/01_drawJuliaSet_Test/drawJuliaSetTest.cpp
 * @Description: create the file
 */

#include <iostream>
#include "../01_drawJuliaSet/drawJuliaSet.h"
#include"../01_drawJuliaSet/PNGManager.h"

int main() {
	std::cout << "Begin" << std::endl;

	GPU_CUDA_L::DrawJuliaSet drawJuliaSet(1000);
	drawJuliaSet.setIteration(150);
	drawJuliaSet.setEnd(5000);

	drawJuliaSet.draw();

	COMMON_LFK::PNGManager PNGManager(1000, 1000, drawJuliaSet.getRGB());
	PNGManager.writePNG("D://test.png");

#if 0
	// 测试 CUDA
	if (!InitCUDA())
	{
		return 0;
	}

	printf("HelloWorld, CUDA has been initialized.\n");
	system("pause");

	return 0;

#endif // 1


#if 0
	// 测试 PNGManager 类的功能
	unsigned char* ptr = new unsigned char[100 * 100 * 3];

	for (int i = 0; i < 100 * 100; i++) {
		ptr[i * 3] = 155;
		ptr[i * 3 + 1] = 0;
		ptr[i * 3 + 2] = 245;
	}

	COMMON_LFK::PNGManager pngManager(100, 100, ptr);
	pngManager.writePNG("D://test.png");
#endif // 0

	return 0;
}