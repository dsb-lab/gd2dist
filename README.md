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
 3. The scikit-build library for python (if not, `pip install scikit-build`)

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

The package behaves very similar to the [scikit-learn](https://scikit-learn.org/) package.

Consider that we have two arrays of data, one with some noise `dataNoise` and the second with the convolved data `dataConvolved`.
Import the package

```python
import gaussiandeconvolution as gd
```

Declare one of the two models. The models consider by default one gaussian for the noise and one gaussian for the convolved data. Consider that we want to fit the noise to one and the convolved data with three.

```python
model = gd.mcmcsampler(K=1, Kc=3)
```
or

```python
model = gd.nestedsampler(K=1, Kc=3)
```

Once declared, fit the model:

```python
model.fit(dataNoise,dataConvolved)
```

With the model fit, we sample from the model

```python
model.sample_autofluorescence(size=100)
model.sample_deconvolution(size=100)
model.sample_convolution(size=100)
```

or score at certain positions. This will return the mean value, as well as any specified percentiles (by default at 0.05 and 0.95).

```python
x = np.arange(0,1,0.1)
model.score_autofluorescence(x, percentiles=[0.05,0.5,0.95])
model.score_deconvolution(x, percentiles=[0.05,0.5,0.95])
model.score_convolution(x, percentiles=[0.05,0.5,0.95])
```

In addition, for the mcmcsampler, it is possible to obtain some resume statistics of the sampler.

```python
model.statistics()
```

### Which model should I use?
