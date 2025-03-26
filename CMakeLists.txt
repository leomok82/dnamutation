cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mutation)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/path/to/pybind11/share/pybind11")
find_package(pybind11 REQUIRED)

set(SOURCES
    mutation.cpp

)


add_executable(mutation ${SOURCES})

target_link_libraries(mutation "${TORCH_LIBRARIES}" pybind11::embed)
set_property(TARGET mutation PROPERTY CXX_STANDARD 17)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET mutation
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:mutation>)
endif (MSVC)