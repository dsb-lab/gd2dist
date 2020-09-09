# gaussDeconv2dist

Package which allow the deconvolution of two random variables using bayesian approaches.

## Installation

The package can be installed from the PYPI repository with the command:

```shell
pip install gaussiandeconvolution
```

The package is compiled for most part of usual operating systems. In case of problems, you can always compile the package from the git repository. The requirements for installation are:
 1. CMake
 2. A C++ compiler and at least c++11 standard (g++, Visual Studio, Clang...)
 3. The scikit-build library for python (if not, pip install scikit-build)

In the gaussian deconvolution folder, create the binary installation.

```shell
python setup.py bdist_wheel
```

And install it.

```shell
pip install ./dist/*
```

If everything is okey, you should be happily running the code after a few seconds of compilation ;)

## Small tutorial

The package constains two different models for performing the deconvolution:

### mcmcsampler

The linux

This model performs
