# scBayesDeconv

Package which allow the deconvolution of two added random variables using bayesian mixture approaches.

```
Z = X + Y
```

where `X` we call it the autofluorescence,`Y` the deconvolution and`Z` the convolution; which are random variables. If we have a sample of values from distribution `X` and `Z`, the package tryes to deconvolve the signal to obtain a distribution of `Y`. The kinds of bayesian mixtures are implemented:

1. Gaussian
2. Gamma

## Installation

The package can be installed from the PyPi repository with the command:

```shell
pip install scBayesDeconv
```

### Problems with nstallation from PyPi
In case of problems, you can always compile the package from the git repository. The requirements for installation are:
 1. CMake
 2. A C++ compiler and at least c++11 standard (g++, Visual Studio, Clang...)
 3. The scikit-build library for python (if not, `pip install scikit-build`)

In the gaussian deconvolution folder, create the binary installation.

```shell
python setup.py bdist_wheel
```

This will generate a wheel in automatically created `dist` folder. Now, we can install it.

```shell
pip install ./dist/*
```

If everything is okey, you should be happily running the code after a few seconds of compilation ;)

## Small tutorial

The package behaves very similar to the [scikit-learn](https://scikit-learn.org/) package.

Consider that we have two arrays of data, one with some noise `dataNoise` and the second with the convolved data `dataConvolved`.
Import the package

```python
import scBayesDeconv as gd
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

With the model fit, we can sample from the model

```python
model.sample_autofluorescence(size=100)
model.sample_deconvolution(size=100)
model.sample_convolution(size=100)
```

or evaluate at certain positions. This will return the mean value, as well as any specified percentiles (by default at 0.05 and 0.95).

```python
x = np.arange(0,1,0.1)
model.score_autofluorescence(x, percentiles=[0.05,0.5,0.95])
model.score_deconvolution(x, percentiles=[0.05,0.5,0.95])
model.score_convolution(x, percentiles=[0.05,0.5,0.95])
```

In addition, for the mcmcsampler, it is possible to obtain some resume statistics of the sampler in order to check if the sampling process has converged to the posterior.

```python
model.statistics()
```

An rhat close to 1 indicates that the posterior chains have mixed appropiately. neff is an indicator of the effective number of independent samples drawn from the model. For more information, have a look to the [Stan](https://mc-stan.org/) package and its associated bayesian statistics book, [Chaper 11](http://www.stat.columbia.edu/~gelman/book/).

### Which model should I use?

Both models correspond to the same posterior likelihood with the only difference on how samples from this posterior are drawn. 

The **mcmc** sampler is based in Gibbs and MCMC markov chain steps with help of indicator variables. This are extensively explained in the book of [Gelman](http://www.stat.columbia.edu/~gelman/book/). Such sampler have the benefit of converging *fast* to a mode of the posterior and have the nice property of concentrating around *solutions with sparse number of components*. Howerver, the posterior distribution of the bayesian deconvolution model is multimodal and, for big noises, can lead to high degeneracies of the system. In such cases, samplers based in markov chains have severe dificulties to converge. Fro the above reasons, this sampler should be used mainly for exploratory purposes in order to have a general idea of the deconvolution as well as the number of components required to describe appropiately the posterior.

The **nested** sampler is based in the ideas of nested sampling introduced by [Skilling](https://projecteuclid.org/euclid.ba/1340370944). Such sampling methods have more power in order to explore complex distributions with multimodalities and complex degeneracies. The counterpart is that the sampler does not select component sparse regions of the space and the exploration becomes fast computationally expensive with the number of components. In order to speed the computation, we wrapped the well documented and recently published library for dynamic nested sampling [Dynesty](https://dynesty.readthedocs.io/en/latest/) around C++ in order to obtain reasonable sampling times for samples of data of the order of magnitude tipically encountered flow citometry datasets. The posteriors obtained through this sampling method capture better the complexity of the gaussian deconvolution in a non-prohibitive amount of time, in contrast to the mcmc sampler.

Overall, one should use the mcmc sampler for exploration and number of components selection and feed that information to a selected nested sampler model in order to obtain the most reliable results within reasonable computational time.
