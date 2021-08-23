cmake_minimum_required(VERSION 3.14...3.17)

project(scBayesDec VERSION "0.1")

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
# Fetch pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(pybind11)

# Fetch boost
FetchContent_Declare(
  boostorg
  GIT_REPOSITORY https://github.com/boostorg/math
  GIT_TAG        master
)
FetchContent_MakeAvailable(boostorg)
include_directories(${boostorg_SOURCE_DIR})
include_directories(${boostorg_SOURCE_DIR}/include)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

set(python_module_name_norm gdposteriormodelnorm)
pybind11_add_module(${python_module_name_norm} MODULE 
  src/scBayesDec/nestedsamplernorm/gdposteriormodelnorm_wrapper.cpp
  src/scBayesDec/nestedsamplernorm/gdposteriormodelnorm.cpp
)
install(TARGETS ${python_module_name_norm} DESTINATION ./nestedsamplernorm)

set(python_module_name_gamma gdposteriormodelgamma)
pybind11_add_module(${python_module_name_gamma} MODULE 
  src/scBayesDec/nestedsamplergamma/gdposteriormodelgamma_wrapper.cpp
  src/scBayesDec/nestedsamplergamma/gdposteriormodelgamma.cpp
  src/scBayesDec/shared_functions/probability_distributions.cpp
)
install(TARGETS ${python_module_name_gamma} DESTINATION ./nestedsamplergamma)

set(python_module_name_nested_norm mcmcposteriorsamplernorm)
pybind11_add_module(${python_module_name_nested_norm} MODULE 
  src/scBayesDec/mcmcsamplernorm/mcmcsamplernorm_wrapper.cpp
  src/scBayesDec/mcmcsamplernorm/mcmcsamplernorm.cpp
  src/scBayesDec/shared_functions/probability_distributions.cpp
  )
install(TARGETS ${python_module_name_nested_norm} DESTINATION ./mcmcsamplernorm)

set(python_module_name_nested_gamma mcmcposteriorsamplergamma)
pybind11_add_module(${python_module_name_nested_gamma} MODULE 
  src/scBayesDec/mcmcsamplergamma/mcmcsamplergamma_wrapper.cpp
  src/scBayesDec/mcmcsamplergamma/mcmcsamplergamma.cpp
  src/scBayesDec/shared_functions/probability_distributions.cpp
  )
install(TARGETS ${python_module_name_nested_gamma} DESTINATION ./mcmcsamplergamma)

set(python_module_name3 shared_functions)
pybind11_add_module(${python_module_name3} MODULE 
  src/scBayesDec/shared_functions/shared_functions_wrapper.cpp
  src/scBayesDec/shared_functions/general_functions.cpp
  src/scBayesDec/shared_functions/gd_samplers.cpp
  src/scBayesDec/shared_functions/gd_scorers.cpp
  src/scBayesDec/shared_functions/global_random_generator.cpp
  src/scBayesDec/shared_functions/probability_distributions.cpp
  src/scBayesDec/shared_functions/mcmc_convergence_statistics.cpp
)
install(TARGETS ${python_module_name3} DESTINATION ./shared_functions)