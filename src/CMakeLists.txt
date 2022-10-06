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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../extern/book/common)
message(${CUDA_TOOLKIT_ROOT_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../extern/image)
message(${CUDA_TOOLKIT_ROOT_DIR})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib)

# 添加源文件 
set(main_SRC main.cpp)

# 通用文件 
file(GLOB_RECURSE Common_SRC "common/*.h" "common/*.cpp")

# 课本上的通用文件
file(GLOB_RECURSE CommonBook_SRC "common_book/*.h" "common_book/GL/*.h")

# CUDA测试
file(GLOB_RECURSE CUDATest "00_CUDA_Test/*.cpp")
 
# 第一次作业，绘制Julia图
file(GLOB_RECURSE DrawJuliaSet "01_drawJuliaSet/*.h" "01_drawJuliaSet/*.cpp")	
file(GLOB_RECURSE DrawJuliaSetTest "01_drawJuliaSet_Test/*.cpp")	
file(GLOB_RECURSE DrawJuliaSetGPU "01_drawJuliaSetGPU/*.cuh" "01_drawJuliaSetGPU/*.cu")	
file(GLOB_RECURSE DrawJuliaSetGPUTest "01_drawJuliaSetGPU_Test/*.cu" "01_drawJuliaSetGPU_Test/*.cpp")	
 
# 第二次作业，光线追踪简易版(white style)
file(GLOB_RECURSE RayTracingSimpleGPU "02_rayRraceSimpleGPU/*.cuh" "02_rayRraceSimpleGPU/*.cu")	
file(GLOB_RECURSE RayTracingSimpleGPUTest "02_rayRraceSimpleGPU_Test/*.cpp")	

#源文件分组 
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${main_SRC}) 
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${Common_SRC})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${CommonBook_SRC})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${CUDATest})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${DrawJuliaSet})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${DrawJuliaSetTest})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${DrawJuliaSetGPU})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${DrawJuliaSetGPUTest})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${RayTracingSimpleGPU})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${RayTracingSimpleGPUTest})

CUDA_ADD_EXECUTABLE(CUDATestExe ${CUDATest} ${Common_SRC}) 
target_link_libraries (CUDATestExe cublas)
  
CUDA_ADD_EXECUTABLE(DrawJuliaSetExe ${Common_SRC} ${DrawJuliaSet} ${DrawJuliaSetTest})
target_link_libraries (DrawJuliaSetExe cublas)

CUDA_ADD_EXECUTABLE(DrawJuliaSetGPUExe ${Common_SRC} ${DrawJuliaSetGPU} ${DrawJuliaSetGPUTest})
target_link_libraries (DrawJuliaSetGPUExe cublas)

CUDA_ADD_EXECUTABLE(RayTracingSimpleGPUExe ${Common_SRC} ${CommonBook_SRC} ${RayTracingSimpleGPU} ${RayTracingSimpleGPUTest})
target_link_libraries (RayTracingSimpleGPUExe cublas)
#target_link_libraries (RayTracingSimpleGPUExe glut64)