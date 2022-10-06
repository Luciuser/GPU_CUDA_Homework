cmake_minimum_required(VERSION 3.3)
################################################################################
set(CMAKE_CXX_STANDARD 14)

IF(MSVC)  # win32平台
    add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()
 
# packages
find_package(CUDA)
# nvcc flags
#set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)
#set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52;-G;-g)
if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 61 75)
endif()
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../extern/image)
message(${CUDA_TOOLKIT_ROOT_DIR})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

# 添加源文件 
set(main_SRC main.cpp)

# 通用文件
file(GLOB_RECURSE Common_SRC "Common/*.h" "Common/*.cpp")

# CUDA测试
file(GLOB_RECURSE CUDATest "00_CUDA_Test/*.cpp")
 
# 第一次作业，绘制Julia图
file(GLOB_RECURSE DrawJuliaSet "01_drawJuliaSet/*.h" "01_drawJuliaSet/*.cpp")	
file(GLOB_RECURSE DrawJuliaSetTest "01_drawJuliaSet_Test/*.cpp")	
file(GLOB_RECURSE DrawJuliaSetGPU "01_drawJuliaSetGPU/*.cuh" "01_drawJuliaSetGPU/*.cu")	
file(GLOB_RECURSE DrawJuliaSetGPUTest "01_drawJuliaSetGPU_Test/*.cu" "01_drawJuliaSetGPU_Test/*.cpp")	
 
#源文件分组 
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${main_SRC}) 
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${Common_SRC})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${CUDATest})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${DrawJuliaSet})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${DrawJuliaSetTest})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${DrawJuliaSetGPU})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${DrawJuliaSetGPUTest})

CUDA_ADD_EXECUTABLE(CUDATestExe ${CUDATest} ${Common_SRC})
target_link_libraries (CUDATestExe cublas)
  
CUDA_ADD_EXECUTABLE(DrawJuliaSetExe ${Common_SRC} ${DrawJuliaSet} ${DrawJuliaSetTest})
target_link_libraries (DrawJuliaSetExe cublas)

CUDA_ADD_EXECUTABLE(DrawJuliaSetGPUExe ${Common_SRC} ${DrawJuliaSetGPU} ${DrawJuliaSetGPUTest})
target_link_libraries (DrawJuliaSetGPUExe cublas)