cmake_minimum_required(VERSION 3.20)
project(NeuralNet CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address,undefined")

# ── Accelerate (Apple BLAS / AMX) ─────────────────────────────────────────────
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
message(STATUS "Accelerate: ${ACCELERATE_FRAMEWORK}")

# ── Metal ─────────────────────────────────────────────────────────────────────
find_library(METAL_FRAMEWORK      Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)
if(METAL_FRAMEWORK)
  message(STATUS "Metal: ${METAL_FRAMEWORK}")
else()
  message(STATUS "Metal: not found — stretch goal disabled")
endif()

# ── Core library ──────────────────────────────────────────────────────────────
add_library(neuralnet
  src/matrix.cpp
  src/layer.cpp
  src/network.cpp
  src/optimizer.cpp
  src/dataloader.cpp
  src/serializer.cpp
)

target_include_directories(neuralnet PUBLIC include metal)
target_link_libraries(neuralnet PUBLIC ${ACCELERATE_FRAMEWORK})
target_compile_definitions(neuralnet PUBLIC ACCELERATE_NEW_LAPACK)

# ── Metal compute kernel ───────────────────────────────────────────────────────
if(METAL_FRAMEWORK)
    set(METAL_SRC ${CMAKE_SOURCE_DIR}/metal/matmul.metal)
    set(METAL_AIR ${CMAKE_BINARY_DIR}/matmul.air)
    set(METAL_LIB ${CMAKE_BINARY_DIR}/default.metallib)

    add_custom_command(
        OUTPUT ${METAL_AIR}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SRC} -o ${METAL_AIR}
        DEPENDS ${METAL_SRC}
    )
    add_custom_command(
        OUTPUT ${METAL_LIB}
        COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
        DEPENDS ${METAL_AIR}
    )
    add_custom_target(metal_lib DEPENDS ${METAL_LIB})

    target_sources(neuralnet PRIVATE metal/matmul_metal.mm)
    add_dependencies(neuralnet metal_lib)
    target_link_libraries(neuralnet PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    target_compile_definitions(neuralnet PUBLIC USE_METAL)
endif()

# ── Executables ───────────────────────────────────────────────────────────────
add_executable(train tools/train.cpp)
target_link_libraries(train PRIVATE neuralnet)

add_executable(bench_matmul benchmarks/bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE neuralnet)

# ── Tests (Catch2 via FetchContent) ───────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)

foreach(t test_matrix test_gradients test_mnist)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE neuralnet Catch2::Catch2WithMain)
endforeach()

include(CTest)
include(Catch)
catch_discover_tests(test_matrix)
catch_discover_tests(test_gradients)
catch_discover_tests(test_mnist)
