import dynesty as dn
from .gdposteriormodelgamma import gdposteriormodelgamma
import numpy as np
import inspect
from scipy.stats import norm
import pickle as pk

from ..shared_functions import *

class nestedsamplergamma(gdposteriormodelgamma):
    """
    Class for the nested sampler of the deconvolution gaussian model
    """
    def __init__(self, K = 1, Kc = 1):
        """
        Constructor of the class


        Parameters
        --------------
            K: int, Number of components of the noise distribution
            Kc: int, Number of components of the convolved distribution

        Returns
        --------------
            nothing
        """
        gdposteriormodel.__init__(self,[],[],K,Kc)
        self.fitted = False

        return

    def fit(self, dataNoise, dataConvolution, **kwargs):
        """
        Fit the model to the posterior distribution

        Parameters
        ------------
            dataNoise: list
                1D array witht he data of the noise
            dataConvolution: list
                1D array witht he data of the convolution
            **kwargs: 
                Arguments to be passed to the *DynamicNestedSampler* and *run_nested* functions from the dynesty package

        Returns
        ------------
            Nothing
        """
        self.data = dataNoise
        self.datac = dataConvolution
        self.dataMin = np.min([dataNoise,dataConvolution])
        self.dataMax = np.max([dataNoise,dataConvolution])

        #separate kargs for the two different samplers functions
        #nested sampler function
        nestedsampler_args = [k for k, v in inspect.signature(dn.NestedSampler).parameters.items()]
        nestedsampler_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in nestedsampler_args}
        if not ("sample" in nestedsampler_dict.keys()):
            nestedsampler_dict["sample"] = "rslice"

        #run nested function
        run_nested_args = [k for k, v in inspect.signature(dn.NestedSampler).parameters.items()]
        run_nested_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in run_nested_args}
        #make fit
        gdposteriormodel.__init__(self,dataNoise,dataConvolution,self.K,self.Kc)
        dynestyModel = dn.NestedSampler(self.logLikelihood, self.prior, 3*self.K+3*self.Kc, **nestedsampler_dict)
        dynestyModel.run_nested(**run_nested_dict)
        self.results = {}
        self.results["samples"] = dynestyModel.results["samples"]
        self.results["logwt"] = dynestyModel.results["logwt"]
        self.results["evidence"] =  dynestyModel.results["logz"][-1]
        weightMax = np.max(self.results["logwt"])
        self.weights = np.exp(self.results["logwt"]-weightMax)
        self.weights = self.weights/np.sum(self.weights)
        self.samples = self.results["samples"]

        self.fitted = True

        return

    def prune(self, order = -1):
        """
        Prune the number of samples to remove samples with weights orders of magnitude lower than the main one. 
        This may speed up drawing samples when the number of draws is huge.

        Parameters
        --------------
            order: int, order of manitude below which prune the samples

        returns:
            nothing 
        """

        if order == -1:

            weightMax = np.max(self.results["logwt"])
            self.weights = np.exp(self.results["logwt"]-weightMax)
            self.weights = self.weights/np.sum(self.weights)
            self.samples = self.results["samples"]

        else:
            weightMax = np.max(self.results["logwt"])
            self.weights = np.exp(self.results["logwt"]-weightMax)
            self.weights = self.weights/np.sum(self.weights)

            select = (self.weights>self.weights.max()*10**-order)
            self.weights = self.weights[select]
            self.weights = self.weights/np.sum(self.weights)
            self.samples = self.results["samples"][select]

        return

    def save(self, name):
        """
        Pickle save the model.

        Parameters
        ----------------
            name: string, name in which to store the model

        Return:
            nothing
        """

        if self.fitted:
            pickling_on = open(name+".pickle","wb")
            pk.dump({"K":self.K, "Kc":self.Kc, "weights":self.results["logwt"], "samples":self.results["samples"], "evidence":self.results["evidence"]}, pickling_on)
            pickling_on.close()
        else:
            print("The model has not been fitted so there is nothing to save.")
        return

    def load(self, name):
        """
        Pickle load the model.

        Parameters
        ----------------
            name: string, name from which to recover the model

        Return:
            nothing
        """
        pickle_off = open(name+".pickle","rb")
        aux = pk.load(pickle_off)
        pickle_off.close()

        self.K = aux["K"]
        self.Kc = aux["Kc"]
        self.results = {}
        self.results["logwt"] = aux["weights"]
        self.results["samples"] = aux["samples"]
        self.results["evidence"] = aux["evidence"]
        
        self.prune()

        self.fitted = True

        return

    def sample_autofluorescence(self, size = 1, style = "full", pos = None):
        """
        Generate samples from the fitted posterior distribution according to the noise distribution

        Parameters
        --------------
            size: int, number of samples to be drawn
            style: string ("full" or "single"), sample from the posterior and then sample, or all the samples from the same posterior draw
            pos: if style = "single", draw from the posterior from which to choose

        Returns:
            list: list, 1D array with *size* samples from the model
        """

        if style=="full":
            return  np.array(sample_autofluorescence(self.samples,self.K,self.Kc,weights=self.weights,size=size))
        elif style=="single":
            if pos == None:
                pos = np.random.choice(range(len(self.samples)), p=self.weights) 
                return  np.array(sample_autofluorescence(self.samples,self.K,self.Kc,weights=self.weights,size=size,pos=pos))
            else:
                return  np.array(sample_autofluorescence(self.samples,self.K,self.Kc,weights=self.weights,size=size,pos=pos))

        return

    def sample_deconvolution(self, size = 1, style = "full", pos = None):
        """
        Generate samples from the fitted posterior distribution according to the deconvolved distribution

        Parameters
        --------------
            size: int, number of samples to be drawn
            style: string ("full" or "single"), sample from the posterior and then sample, or all the samples from the same posterior draw
            pos: if style = "single", draw from the posterior from which to choose

        Returns:
            list: list, 1D array with *size* samples from the model
        """

        if style=="full":
            return  np.array(sample_deconvolution(self.samples,self.K,self.Kc,weights=self.weights,size=size))
        elif style=="single":
            if pos == None:
                pos = np.random.choice(range(len(self.samples)), p=self.weights) 
                return  np.array(sample_deconvolution(self.samples,self.K,self.Kc,weights=self.weights,size=size,pos=pos))
            else:
                return  np.array(sample_deconvolution(self.samples,self.K,self.Kc,weights=self.weights,size=size,pos=pos))

        return

    def sample_convolution(self, size = 1, style = "full", pos = None):
        """
        Generate samples from the fitted posterior distribution according to the convolved distribution

        Parameters
        --------------
            size: int, number of samples to be drawn
            style: string ("full" or "single"), sample from the posterior and then sample, or all the samples from the same posterior draw
            pos: if style = "single", draw from the posterior from which to choose

        Returns:
            list: list, 1D array with *size* samples from the model
        """

        if style=="full":
            return  np.array(sample_convolution(self.samples,self.K,self.Kc,weights=self.weights,size=size))
        elif style=="single":
            if pos == None:
                pos = np.random.choice(range(len(self.samples)), p=self.weights) 
                return  np.array(sample_convolution(self.samples,self.K,self.Kc,weights=self.weights,size=size,pos=pos))
            else:
                return  np.array(sample_convolution(self.samples,self.K,self.Kc,weights=self.weights,size=size,pos=pos))

        return

    def score_autofluorescence(self, x, percentiles = [0.05, 0.95], size = 500):
        """
        Evaluate the mean and percentiles of the the pdf at certain position acording to the noise distribution

        Parameters
        --------------
            x: list/array, positions where to evaluate the distribution
            percentiles: list/array, percentiles to be evaluated
            size: int, number of samples to draw from the posterior to make the statistics, bigger numbers give more stability

        Returns:
            list: list, 2D array with the mean and all the percentile evaluations at all points in x
        """

        return  np.array(score_autofluorescence(self.samples, x, self.K,self.Kc, percentiles = percentiles, weights=self.weights, size=size))

    def score_deconvolution(self, x, percentiles = [0.05, 0.95], size = 500):
        """
        Evaluate the mean and percentiles of the the pdf at certain position acording to the deconvolved distribution

        Parameters
        --------------
            x: list/array, positions where to evaluate the distribution
            percentiles: list/array, percentiles to be evaluated
            size: int, number of samples to draw from the posterior to make the statistics, bigger numbers give more stability

        Returns:
            list: list, 2D array with the mean and all the percentile evaluations at all points in x
        """

        return  np.array(score_deconvolution(self.samples, x, self.K, self.Kc, percentiles = percentiles, weights=self.weights, size=size))

    def score_convolution(self, x, percentiles = [0.05, 0.95], size = 500):
        """
        Evaluate the mean and percentiles of the the pdf at certain position acording to the convolved distribution

        Parameters
        --------------
            x: list/array, positions where to evaluate the distribution
            percentiles: list/array, percentiles to be evaluated
            size: int, number of samples to draw from the posterior to make the statistics, bigger numbers give more stability

        Returns:
            list: list, 2D array with the mean and all the percentile evaluations at all points in x
        """

        return  np.array(score_convolution(self.samples, x, self.K, self.Kc, percentiles = percentiles, weights=self.weights, size=size))
