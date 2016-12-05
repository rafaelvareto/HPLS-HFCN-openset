cmake_minimum_required(VERSION 3.0)

# Add folder where are supportive functions
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)

# Include qt current directories in order to use promoted objects
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cvtools)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plsh)

# Basic info about project
project(PLSHface VERSION 1.0)

file(GLOB CVTOOLS_SOURCE_FILES "cvtools/*.cpp")
file(GLOB CVTOOLS_INCLUDE_FILES "cvtools/*.h")
file(GLOB PLSH_SOURCE_FILES "plsh/*.cpp")
file(GLOB PLSH_INCLUDE_FILES "plsh/*.h")

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Boost REQUIRED)
find_package(Boost 1.54.0 COMPONENTS filesystem system program_options date_time regex REQUIRED)
#include_directories(${Boost_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
find_package(OpenCV COMPONENTS opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui REQUIRED)
find_path(OpenCV_INCLUDE_DIR "opencv/cv.h" PATHS "${OpenCV_DIR}" PATH_SUFFIXES "include" DOC "")

# Print some message showing some found packages
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_library(${PROJECT_NAME}_CVTOOLS ${CVTOOLS_SOURCE_FILES} ${CVTOOLS_INCLUDE_FILES})
target_link_libraries(${PROJECT_NAME}_CVTOOLS ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME}_CVTOOLS PUBLIC cvtools/)

add_library(${PROJECT_NAME}_PLSH ${PLSH_SOURCE_FILES} ${PLSH_INCLUDE_FILES})
target_link_libraries(${PROJECT_NAME}_PLSH  ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME}_PLSH PUBLIC plsh/)

add_executable(${PROJECT_NAME} ${CVTOOLS_SOURCE_FILES} ${PLSH_SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_CVTOOLS)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_PLSH)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIR})
