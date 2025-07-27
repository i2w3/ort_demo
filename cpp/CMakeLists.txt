cmake_minimum_required(VERSION 3.10)

project(ORTDemo 
        LANGUAGES CXX 
        VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-declarations")

option(USE_CUDA "Use CUDA for GPU acceleration" OFF)
option(USE_LOGGER "Use custom logger" OFF)
option(USE_PARALLEL "Use parallel processing" ON)
configure_file(./include/version.h.in version.h)

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


# add_executable(test_cls src/test_cls.cpp src/cls.cpp)
# target_include_directories(test_cls PRIVATE ${PROJECT_BINARY_DIR} ${OpenCV_INCLUDE_DIRS} ${ONNXRuntime_INCLUDE_DIRS} ./include)
# target_link_libraries(test_cls ${OpenCV_LIBS} ${ONNXRuntime_LIBRARIES})

# add_executable(test_det src/test_det.cpp src/det.cpp)
# target_include_directories(test_det PRIVATE ${PROJECT_BINARY_DIR} ${OpenCV_INCLUDE_DIRS} ${ONNXRuntime_INCLUDE_DIRS} ./include)
# target_link_libraries(test_det ${OpenCV_LIBS} ${ONNXRuntime_LIBRARIES})

add_library(ppocr SHARED src/ppocr.cpp src/det.cpp src/cls.cpp)
target_include_directories(ppocr PRIVATE ${PROJECT_BINARY_DIR} ${OpenCV_INCLUDE_DIRS} ${ONNXRuntime_INCLUDE_DIRS} ./include)
target_link_libraries(ppocr ${OpenCV_LIBS} ${ONNXRuntime_LIBRARIES})
if (USE_PARALLEL)
    target_compile_definitions(ppocr PRIVATE USE_PARALLEL)
endif()

add_executable(test_ppocr src/test_ppocr.cpp)
target_include_directories(test_ppocr PRIVATE ${PROJECT_BINARY_DIR} ${OpenCV_INCLUDE_DIRS} ./include)
target_link_libraries(test_ppocr ppocr)