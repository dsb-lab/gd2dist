from .mcmcposteriorsampler import fit
from scipy.stats import norm
import pandas as pd
import numpy as np

from ..shared_functions import *

class mcmcsampler:
    def __init__(self, K=1, Kc=1, alpha = 1, alphac = 1):
        self.K = K
        self.Kc = Kc
        self.alpha = alpha
        self.alphac = alphac

        return

    def fit(self, data, datac, iterations = 1000, ignored_iterations = 1000, chains = 1, sigmawidth = 0.1, initialConditions = [], showProgress = True, seed = 0):

        self.data = data
        self.datac = datac
        self.iterations = iterations
        self.ignored_iterations = ignored_iterations
        self.chains = chains
        self.sigmawidth = sigmawidth
        self.samples = np.array(fit(data, datac, ignored_iterations, iterations, chains, self.K, self.Kc, self.alpha, self.alphac, sigmawidth, initialConditions, showProgress, seed))
        
        return

    def sample_autofluorescence(self, size = 1):

        return  sample_autofluorescence(self.samples,self.K,self.Kc,size)

    def sample_deconvolution(self, size = 1):

        return  sample_deconvolution(self.samples,self.K,self.Kc,size)

    def sample_convolution(self, size = 1):

        return  sample_convolution(self.samples,self.K,self.Kc,size)

    def score_autofluorescence(self, x, percentiles = [0.05, 0.95], size = 100):

        return  score_autofluorescence(self.samples, x, self.K, self.Kc, percentiles, size)

    def score_deconvolution(self, x, percentiles = [0.05, 0.95], size = 100):

        return  score_deconvolution(self.samples, x, self.K, self.Kc, percentiles, size)

    def score_convolution(self, x, percentiles = [0.05, 0.95], size = 100):

        return  score_convolution(self.samples, x, self.K, self.Kc, percentiles, size)

    def sampler_statistics(self, sort="weight"):
        self.sampler_statistics = pd.DataFrame(columns=["Mean","Std","5%","50%","95%","Rhat","Neff"])

        samples = self.samples.copy()

        if sort == "weight":
            argsort = np.argsort(samples[:,0:self.K],axis=1)
            samples[:,0:self.K] = samples[:,0:self.K][argsort]
            samples[:,self.K:2*self.K] = samples[:,self.K:2*self.K][argsort]
            samples[:,2*self.K:3*self.K] = samples[:,2*self.K:3*self.K][argsort]

            argsort = np.argsort(samples[:,3*self.K:3*self.K+self.Kc],axis=1)
            samples[:,3*self.K:3*self.K+self.Kc] = samples[:,3*self.K:3*self.K+self.Kc][argsort]
            samples[:,(3*self.K+self.Kc):(3*self.K+2*self.Kc)] = samples[:,(3*self.K+self.Kc):(3*self.K+2*self.Kc)][argsort]
            samples[:,(3*self.K+2*self.Kc):(3*self.K+3*self.Kc)] = samples[:,(3*self.K+2*self.Kc):(3*self.K+3*self.Kc)][argsort]

        if sort == "mean":
            argsort = np.argsort(samples[:,self.K:2*self.K],axis=1)
            samples[:,0:self.K] = samples[:,0:self.K][argsort]
            samples[:,self.K:2*self.K] = samples[:,self.K:2*self.K][argsort]
            samples[:,2*self.K:3*self.K] = samples[:,2*self.K:3*self.K][argsort]

            argsort = np.argsort(samples[:,3*self.K+self.Kc:3*self.K+2*self.Kc],axis=1)
            samples[:,3*self.K:3*self.K+self.Kc] = samples[:,3*self.K:3*self.K+self.Kc][argsort]
            samples[:,(3*self.K+self.Kc):(3*self.K+2*self.Kc)] = samples[:,(3*self.K+self.Kc):(3*self.K+2*self.Kc)][argsort]
            samples[:,(3*self.K+2*self.Kc):(3*self.K+3*self.Kc)] = samples[:,(3*self.K+2*self.Kc):(3*self.K+3*self.Kc)][argsort]

        measures = np.zeros(7)
        for i in range(3*self.K+3*self.Kc):
            measures[0] = np.mean(samples[:,i])
            measures[1] = np.std(samples[:,i])
            measures[2:5] = np.percentile(samples[:,i],[5,50,95])
            measures[5] = rstat(samples[:,i],self.chains)
            measures[6] = effnumber(samples[:,i],self.chains)

            #Name the component
            if i < self.K:
                name = "weight_K"+str(1+i)
            elif i < 2*self.K:
                name = "mean_K"+str(1+i-self.K)
            elif i < 3*self.K:
                name = "std_K"+str(1+i-2*self.K)
            elif i < 3*self.K+self.Kc:
                name = "weight_Kc"+str(1+i-3*self.K)
            elif i < 3*self.K+2*self.Kc:
                name = "mean_Kc"+str(1+i-3*self.K-self.Kc)
            else:
                name = "std_Kc"+str(1+i-3*self.K-2*self.Kc)

            self.sampler_statistics = self.sampler_statistics.append(pd.Series(measures, ["Mean","Std","5%","50%","95%","Rhat","Neff"], name=name))

        return self.sampler_statistics
