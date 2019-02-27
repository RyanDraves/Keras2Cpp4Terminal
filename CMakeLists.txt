cmake_minimum_required(VERSION 2.8.9)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O1")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
set(CMAKE_LINUX_RPATH 1)

project (neural_net)
set (SOURCE keras2cpp/keras_model.cc)
add_library(neural_net MODULE ${SOURCE})
