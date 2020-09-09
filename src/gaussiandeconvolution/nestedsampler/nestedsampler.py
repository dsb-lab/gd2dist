import dynesty as dn
from .gdposteriormodel import gdposteriormodel
import numpy as np
import inspect
from scipy.stats import norm

from ..shared_functions import *

class nestedsampler(gdposteriormodel):
    def __init__(self, K = 1, Kc = 1, *kargs):
        gdposteriormodel.__init__(self,[],[],K,Kc)

        return

    def fit(self, dataNoise, dataConvolution, **kwargs):
        self.data = dataNoise
        self.datac = dataConvolution

        #separate kargs for the two different samplers functions
        #nested sampler function
        nestedsampler_args = [k for k, v in inspect.signature(dn.DynamicNestedSampler).parameters.items()]
        nestedsampler_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in nestedsampler_args}
        if not ("sample" in nestedsampler_dict.keys()):
            nestedsampler_dict["sample"] = "rslice"

        #run nested function
        run_nested_args = [k for k, v in inspect.signature(dn.NestedSampler).parameters.items()]
        run_nested_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in run_nested_args}
        #make fit
        gdposteriormodel.__init__(self,dataNoise,dataConvolution,self.K,self.Kc)
        self.dynestyModel = dn.DynamicNestedSampler(self.logLikelihood, self.prior, 3*self.K+3*self.Kc, **nestedsampler_dict)
        self.dynestyModel.run_nested(**run_nested_dict)
        self.samples = self.dynestyModel.results["samples"]
        weightMax = np.max(self.dynestyModel.results["logz"])
        self.weights = np.exp(self.dynestyModel.results["logz"]-weightMax)
        self.weights = self.weights/np.sum(self.weights)

        return

    def sample_autofluorescence(self, size = 1):

        return  sample_autofluorescence(self.samples,self.K,self.Kc,weights=self.weights,size=size)

    def sample_deconvolution(self, size = 1):

        return  sample_deconvolution(self.samples,self.K,self.Kc,weights=self.weights,size=size)

    def sample_convolution(self, size = 1):

        return  sample_convolution(self.samples,self.K,self.Kc,weights=self.weights,size=size)

    def score_autofluorescence(self, x, percentiles = [0.05, 0.95], size = 500):

        return  score_autofluorescence(self.samples, x, self.K,self.Kc, percentiles = percentiles, weights=self.weights, size=size)

    def score_deconvolution(self, x, percentiles = [0.05, 0.95], size = 500):

        return  score_deconvolution(self.samples, x, self.K, self.Kc, percentiles = percentiles, weights=self.weights, size=size)

    def score_convolution(self, x, percentiles = [0.05, 0.95], size = 500):

        return  score_convolution(self.samples, x, self.K, self.Kc, percentiles = percentiles, weights=self.weights, size=size)
