cmake_minimum_required(VERSION 3.30)

project(class_learn)

# vcpkg
# set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

# MV
set(MVS_LIB "C:/Program Files (x86)/MVS/Development")
set(MVFG_LIB "C:/Program Files (x86)/MVS/Development/MVFG")

file(GLOB MVS_LIBS "${MVS_LIB}/Libraries/win64/*.lib")
file(GLOB MVFG_LIBS "${MVFG_LIB}/Libraries/win64/*.lib")

# flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 创建共享配置库
add_library(shared_config INTERFACE)

# opencv: vcpkg install opencv4
set(OpenCV_ROOT "C:/Users/yeisme/lib/opencv")
find_package(OpenCV REQUIRED)

# 配置共享库
target_include_directories(shared_config INTERFACE 
    "${MVS_LIB}/Includes" 
    "${MVFG_LIB}/Includes"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(shared_config INTERFACE 
    ${MVS_LIBS} 
    ${MVFG_LIBS}
    ${OpenCV_LIBS}
)

if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
endif()

# 添加子目录
add_subdirectory(class1)
add_subdirectory(class2)
add_subdirectory(class3)
add_subdirectory(class4)
add_subdirectory(class5)
add_subdirectory(class6)