# - Find ONNX Runtime (Optimized for Conda CUDA Environment)
#
# This module finds ONNX Runtime library installed in a Conda environment
# with specific support for CUDA builds. It prioritizes Conda paths and
# includes CUDA provider libraries when available.
#
# This module sets the following variables:
#
#  ONNXRuntime_FOUND          - TRUE if ONNX Runtime was found.
#  ONNXRuntime_INCLUDE_DIRS   - The directory containing ONNX Runtime headers.
#  ONNXRuntime_LIBRARIES      - The ONNX Runtime libraries to link against.
#  ONNXRuntime_VERSION        - The version of the found ONNX Runtime.
#  ONNXRuntime_CUDA_FOUND     - TRUE if CUDA provider libraries are available.
#  ONNXRuntime_CUDA_LIBS      - CUDA provider libraries when available.
#
include(FindPackageHandleStandardArgs)

message(STATUS "Searching for ONNX Runtime...")

# Get conda environment path from environment variable or use explicit path
set(ONNXRUNTIME_CONDA_PATHS
    $ENV{CONDA_PREFIX}
)

# Filter out empty paths
set(ONNXRUNTIME_VALID_PATHS "")
foreach(path IN LISTS ONNXRUNTIME_CONDA_PATHS)
    if(path AND EXISTS "${path}")
        list(APPEND ONNXRUNTIME_VALID_PATHS "${path}")
    endif()
endforeach()

# Add fallback search paths
list(APPEND ONNXRUNTIME_VALID_PATHS
    $ENV{ONNXRUNTIME_DIR}
    /usr/local
    /usr
    /opt
)

# Find the include directory
find_path(ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime/core/session/onnxruntime_c_api.h
    HINTS ${ONNXRUNTIME_VALID_PATHS}
    PATH_SUFFIXES include
    DOC "Path to ONNXRuntime include directory"
    NO_DEFAULT_PATH
)

# Fallback to default search if not found in conda
if(NOT ONNXRuntime_INCLUDE_DIR)
    find_path(ONNXRuntime_INCLUDE_DIR
        NAMES onnxruntime/core/session/onnxruntime_c_api.h
        PATH_SUFFIXES include
        DOC "Path to ONNXRuntime include directory"
    )
endif()

# Find the main library
find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_VALID_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "Path to ONNXRuntime library"
    NO_DEFAULT_PATH
)

# Fallback to default search if not found in conda
if(NOT ONNXRuntime_LIBRARY)
    find_library(ONNXRuntime_LIBRARY
        NAMES onnxruntime
        PATH_SUFFIXES lib lib64
        DOC "Path to ONNXRuntime library"
    )
endif()

# Find CUDA provider libraries (optional)
find_library(ONNXRuntime_CUDA_LIBRARY
    NAMES onnxruntime_providers_cuda
    HINTS ${ONNXRUNTIME_VALID_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "Path to ONNXRuntime CUDA provider library"
    NO_DEFAULT_PATH
)

find_library(ONNXRuntime_SHARED_LIBRARY
    NAMES onnxruntime_providers_shared
    HINTS ${ONNXRUNTIME_VALID_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "Path to ONNXRuntime shared provider library"
    NO_DEFAULT_PATH
)

# Try to determine the version from multiple sources
set(ONNXRuntime_VERSION "")

# Method 1: From VERSION_NUMBER file
if(ONNXRuntime_INCLUDE_DIR)
    set(ONNXRUNTIME_VERSION_FILE "${ONNXRuntime_INCLUDE_DIR}/onnxruntime/VERSION_NUMBER")
    if(EXISTS "${ONNXRUNTIME_VERSION_FILE}")
        file(READ "${ONNXRUNTIME_VERSION_FILE}" _version)
        string(STRIP "${_version}" ONNXRuntime_VERSION)
    endif()
endif()

# Method 2: From library path pattern
if(NOT ONNXRuntime_VERSION AND ONNXRuntime_LIBRARY)
    get_filename_component(_libname "${ONNXRuntime_LIBRARY}" NAME)
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" _version "${_libname}")
    if(_version)
        set(ONNXRuntime_VERSION "${_version}")
    endif()
endif()

# Method 3: From conda meta information
if(NOT ONNXRuntime_VERSION)
    foreach(path IN LISTS ONNXRUNTIME_CONDA_PATHS)
        if(path AND EXISTS "${path}/conda-meta")
            file(GLOB _meta_files "${path}/conda-meta/onnxruntime-*.json")
            if(_meta_files)
                list(GET _meta_files 0 _meta_file)
                file(READ "${_meta_file}" _meta_content)
                string(REGEX MATCH "\"version\": \"([0-9.]+)\"" _match "${_meta_content}")
                if(CMAKE_MATCH_1)
                    set(ONNXRuntime_VERSION "${CMAKE_MATCH_1}")
                    break()
                endif()
            endif()
        endif()
    endforeach()
endif()

# Set CUDA-related variables
if(ONNXRuntime_CUDA_LIBRARY AND ONNXRuntime_SHARED_LIBRARY)
    set(ONNXRuntime_CUDA_FOUND TRUE)
    set(ONNXRuntime_CUDA_LIBS ${ONNXRuntime_CUDA_LIBRARY} ${ONNXRuntime_SHARED_LIBRARY})
else()
    set(ONNXRuntime_CUDA_FOUND FALSE)
    set(ONNXRuntime_CUDA_LIBS "")
endif()

# Store results
set(ONNXRuntime_INCLUDE_DIRS ${ONNXRuntime_INCLUDE_DIR})
set(ONNXRuntime_LIBRARIES ${ONNXRuntime_LIBRARY})

# Use the standard CMake function
find_package_handle_standard_args(ONNXRuntime
    FOUND_VAR ONNXRuntime_FOUND
    REQUIRED_VARS ONNXRuntime_LIBRARIES ONNXRuntime_INCLUDE_DIRS
    VERSION_VAR ONNXRuntime_VERSION
)

# Create imported targets for modern CMake
if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
    add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIRS}"
    )
    
    if(ONNXRuntime_CUDA_FOUND)
        add_library(ONNXRuntime::CUDA SHARED IMPORTED)
        set_target_properties(ONNXRuntime::CUDA PROPERTIES
            IMPORTED_LOCATION "${ONNXRuntime_CUDA_LIBRARY}"
            INTERFACE_LINK_LIBRARIES "ONNXRuntime::ONNXRuntime"
        )
    endif()
endif()

# Mark variables as advanced
mark_as_advanced(
    ONNXRuntime_INCLUDE_DIR
    ONNXRuntime_LIBRARY
    ONNXRuntime_CUDA_LIBRARY
    ONNXRuntime_SHARED_LIBRARY
)

# Status messages
if(ONNXRuntime_FOUND)
    if(NOT ONNXRuntime_FIND_QUIETLY)
        message(STATUS "Found ONNXRuntime: ${ONNXRuntime_LIBRARIES}")
        message(STATUS "  Includes: ${ONNXRuntime_INCLUDE_DIRS}")
        message(STATUS "  Version: ${ONNXRuntime_VERSION}")
        if(ONNXRuntime_CUDA_FOUND)
            message(STATUS "  CUDA Provider: Found")
        else()
            message(STATUS "  CUDA Provider: Not found")
        endif()
    endif()
else()
    if(ONNXRuntime_FIND_REQUIRED)
        message(FATAL_ERROR "ONNXRuntime not found. Please ensure ONNX Runtime is installed in your conda environment.")
    endif()
endif()