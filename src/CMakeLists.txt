cmake_minimum_required(VERSION 3.3)
################################################################################
set(CMAKE_CXX_STANDARD 14)

IF(MSVC)  # win32平台
    add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()

# 添加源文件
set(main_SRC main.cpp)

file(GLOB_RECURSE DrawJuliaSet "01_drawJuliaSet/*.h" "01_drawJuliaSet/*.cpp")	
file(GLOB_RECURSE DrawJuliaSetTest "01_drawJuliaSet_Test/*.cpp")	

#源文件分组
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${main_SRC})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${DrawJuliaSet})
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${DrawJuliaSetTest})
 
add_executable(DrawJuliaSet ${DrawJuliaSet} ${DrawJuliaSetTest})
