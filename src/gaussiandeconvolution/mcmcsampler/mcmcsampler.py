from .mcmcposteriorsampler import fit
from scipy.stats import norm

from ..shared_functions import *


class mcmcsampler:
    def __init__(self, K=1, Kc=1, alpha = 1, alphac = 1):
        self.K = 1
        self.Kc = 1
        self.alpha = 1
        self.alphac = 1

        return

    def fit(self, data, datac, iterations = 1000, ignored_iterations = 1000, chains = 1, sigmawidth = 0.1, initialConditions = [], showProgress = True):

        self.data = data
        self.datac = datac
        self.iterations = iterations
        self.ignored_iterations = ignored_iterations
        self.chains = chains
        self.sigmawidth = sigmawidth
        self.posterior = fit(data, datac, ignored_iterations, ignored_iterations, chains, self.K, self.Kc, self.alpha, self.alphac, sigmawidth, initialConditions, showProgress)
        
        return

    def sample_autofluorescence(self, size = 1):

        return  sample_autofluorescence(self.samples,self._K,self._Kc,size=size)

    def sample_deconvolution(self, size = 1):

        return  sample_deconvolution(self.samples,self._K,self._Kc,size=size)

    def sample_convolution(self, size = 1):

        return  sample_convolution(self.samples,self._K,self._Kc,size=size)

    def score_autofluorescence(self, x, size = 100):

        return  score_autofluorescence(self.samples, x, self._K,self._Kc, size=size)

    def score_deconvolution(self, x, size = 100):

        return  sample_deconvolution(self.samples, x, self._K, self._Kc, size=size)

    def score_convolution(self, x, size = 100):

        return  sample_convolution(self.samples, x, self._K, self._Kc, size=size)
