cmake_minimum_required(VERSION 3.14...3.17)

project(gd2dist VERSION "0.1")

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

set(python_module_name1 gdposteriormodel)
pybind11_add_module(${python_module_name1} MODULE 
  src/gd2dist/nestedsampler/gdposteriormodel_wrapper.cpp
  src/gd2dist/nestedsampler/gdposteriormodel.cpp
)
install(TARGETS ${python_module_name1} DESTINATION ./nestedsampler)

set(python_module_name2 mcmcposteriorsampler)
pybind11_add_module(${python_module_name2} MODULE 
  src/gd2dist/mcmcsampler/mcmcsampler_wrapper.cpp
  src/gd2dist/mcmcsampler/mcmcsampler.cpp
  src/gd2dist/shared_functions/probability_distributions.cpp
  )
install(TARGETS ${python_module_name2} DESTINATION ./mcmcsampler)

set(python_module_name3 shared_functions)
pybind11_add_module(${python_module_name3} MODULE 
  src/gd2dist/shared_functions/shared_functions_wrapper.cpp
  src/gd2dist/shared_functions/general_functions.cpp
  src/gd2dist/shared_functions/gd_samplers.cpp
  src/gd2dist/shared_functions/gd_scorers.cpp
  src/gd2dist/shared_functions/global_random_generator.cpp
  src/gd2dist/shared_functions/probability_distributions.cpp
  src/gd2dist/shared_functions/mcmc_convergence_statistics.cpp
)
install(TARGETS ${python_module_name3} DESTINATION ./shared_functions)