cmake_minimum_required(VERSION 3.14...3.17)

project(gaussiandeconvolution VERSION "0.1")

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(pybind11)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

set(CMAKE_SYSTEM_NAME Windows)
set(TOOLCHAIN_PREFIX x86_64-w64-mingw32)
#set(TOOLCHAIN_PREFIX i686-w64-mingw32)
set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES /usr/lib/gcc/${TOOLCHAIN_PREFIX}/10.2.0)
# cross compilers to use for C and C++
set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
#set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}-gcc-posix)
#set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++-posix)
set(CMAKE_RC_COMPILER ${TOOLCHAIN_PREFIX}-windres)

#include_directories(/usr/include)

set(python_module_name1 gdposteriormodel)
pybind11_add_module(${python_module_name1} MODULE 
  src/gaussiandeconvolution/nestedsampler/gdposteriormodel_wrapper.cpp
  src/gaussiandeconvolution/nestedsampler/gdposteriormodel.cpp
)
install(TARGETS ${python_module_name1} DESTINATION ./nestedsampler)

set(python_module_name2 mcmcposteriorsampler)
pybind11_add_module(${python_module_name2} MODULE 
  src/gaussiandeconvolution/mcmcsampler/mcmcsampler_wrapper.cpp
  src/gaussiandeconvolution/mcmcsampler/mcmcsampler.cpp
  src/gaussiandeconvolution/shared_functions/probability_distributions.cpp
  )
install(TARGETS ${python_module_name2} DESTINATION ./mcmcsampler)

set(python_module_name3 shared_functions)
pybind11_add_module(${python_module_name3} MODULE 
  src/gaussiandeconvolution/shared_functions/shared_functions_wrapper.cpp
  src/gaussiandeconvolution/shared_functions/general_functions.cpp
  src/gaussiandeconvolution/shared_functions/gd_samplers.cpp
  src/gaussiandeconvolution/shared_functions/gd_scorers.cpp
  src/gaussiandeconvolution/shared_functions/global_random_generator.cpp
  src/gaussiandeconvolution/shared_functions/probability_distributions.cpp
  src/gaussiandeconvolution/shared_functions/mcmc_convergence_statistics.cpp
)
install(TARGETS ${python_module_name3} DESTINATION ./shared_functions)