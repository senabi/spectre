cmake_minimum_required(VERSION 3.5.1)

project(safeside VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS off)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(ASM)

add_executable(spectre_v1_pht
    spectre.cc
    read_latency.S
)
