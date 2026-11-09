cmake_minimum_required(VERSION 3.20)
project(pino_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnxruntime CONFIG REQUIRED)

add_executable(pino_app
    "pino.cpp"
    "pino_inference.cpp"
    "pino_inference.h"
)

target_include_directories(pino_app PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pino_app PRIVATE
    onnxruntime::onnxruntime
)