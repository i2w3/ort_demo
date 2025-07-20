cmake_minimum_required(VERSION 3.10)

project(ORTDemo CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Use CUDA for GPU acceleration" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# OpenCV
find_package(OpenCV REQUIRED)
# if (OpenCV_FOUND)
#     message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
#     message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
# else ()
#     message(FATAL_ERROR "OpenCV Not Found!")
# endif (OpenCV_FOUND)

# ONNXRuntime
find_package(ONNXRuntime REQUIRED)
# if (ONNXRuntime_FOUND)
#     message(STATUS "ONNXRuntime_LIBS: ${ONNXRuntime_LIBRARIES}")
#     message(STATUS "ONNXRuntime_INCLUDE_DIRS: ${ONNXRuntime_INCLUDE_DIRS}")
# else ()
#     message(FATAL_ERROR "ONNXRuntime Not Found!")
# endif (ONNXRuntime_FOUND)


add_executable(test_cls src/test_cls.cpp src/cls.cpp)

target_include_directories(test_cls PRIVATE ${OpenCV_INCLUDE_DIRS} ${ONNXRuntime_INCLUDE_DIRS} ./include)
target_link_libraries(test_cls ${OpenCV_LIBS} ${ONNXRuntime_LIBRARIES})
