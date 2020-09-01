import dynesty as dn
from .gdposteriormodel import gdposteriormodel
import numpy as np
import inspect
from scipy.stats import norm

class nestedSampler(gdposteriormodel):
    def __init__(self, K = 1, Kc = 1, *kargs):
        gdposteriormodel.__init__(self,[],[],K,Kc)

    def fit(self, dataNoise, dataConvolution, **kwargs):

        #separate kargs for the two different samplers functions
        #nested sampler function
        nestedsampler_args = [k for k, v in inspect.signature(dn.NestedSampler).parameters.items()]
        nestedsampler_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in nestedsampler_args}
        #run nested function
        run_nested_args = [k for k, v in inspect.signature(dn.NestedSampler).parameters.items()]
        run_nested_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in run_nested_args}
        #make fit
        gdposteriormodel.__init__(self,dataNoise,dataConvolution,self._K,self._Kc)
        self.dynestyModel = dn.NestedSampler(self.logLikelihood, self.prior, 3*self._K+3*self._Kc, **nestedsampler_dict)
        self.dynestyModel.run_nested(**run_nested_dict)

        return

    def score_autofluorescence(self, x, samples = 100, percentiles=[5,95]):
        weightMax = np.max(self.dynestyModel.results["logz"])
        weights = np.exp(self.dynestyModel.results["logz"]-weightMax)
        weights = weights/np.sum(weights)
        choices = np.random.choice(range(len(weights)), samples, p=weights)
        
        evalSamples = []
        for i in choices:
            aux = []
            s = self.dynestyModel.results["samples"][i]
            for j in range(self._K):
                aux.append(s[j]*norm.pdf(x,loc=s[self._K+j],scale=s[2*self._K+j]))
            evalSamples.append(np.sum(aux,axis=0))

        return np.mean(evalSamples,axis=0), np.percentile(evalSamples,percentiles,axis=0)

    def score_deconvolution(self, x, samples = 100, percentiles=[5,95]):
        weightMax = np.max(self.dynestyModel.results["logz"])
        weights = np.exp(self.dynestyModel.results["logz"]-weightMax)
        weights = weights/np.sum(weights)
        choices = np.random.choice(range(len(weights)), samples, p=weights)
        
        evalSamples = []
        for i in choices:
            aux = []
            s = self.dynestyModel.results["samples"][i]
            for j in range(self._Kc):
                aux.append(s[3*self._K+j]*norm.pdf(x,loc=s[3*self._K+self._Kc+j],scale=s[3*self._K+2*self._Kc+j]))
            evalSamples.append(np.sum(aux,axis=0))

        return np.mean(evalSamples,axis=0), np.percentile(evalSamples,percentiles,axis=0)

    def score_convolution(self, x, samples = 100, percentiles=[5,95]):
        weightMax = np.max(self.dynestyModel.results["logz"])
        weights = np.exp(self.dynestyModel.results["logz"]-weightMax)
        weights = weights/np.sum(weights)
        choices = np.random.choice(range(len(weights)), samples, p=weights)
        
        evalSamples = []
        for i in choices:
            aux = []
            s = self.dynestyModel.results["samples"][i]
            for j in range(self._K):
                for k in range(self._Kc):
                    aux.append(s[j]*s[3*self._K+k]*norm.pdf(x,loc=s[self._K+j]+s[3*self._K+self._Kc+k],scale=np.sqrt(s[2*self._K+j]**2+s[3*self._K+2*self._Kc+k]**2)))
            evalSamples.append(np.sum(aux,axis=0))

        return np.mean(evalSamples,axis=0), np.percentile(evalSamples,percentiles,axis=0)

    def sample_autofluorescence(self, samples = 1):
        weightMax = np.max(self.dynestyModel.results["logz"])
        weights = np.exp(self.dynestyModel.results["logz"]-weightMax)
        weights = weights/np.sum(weights)
        choices = np.random.choice(range(len(weights)), samples, p=weights)
        
        evalSamples = np.zeros(samples)
        for j,i in enumerate(choices):
            s = self.dynestyModel.results["samples"][i][0:self._K]
            choice = np.random.choice(range(self._K),p=s)
            s = self.dynestyModel.results["samples"][i]
            evalSamples[j] = norm.rvs(loc=s[self._K+choice],scale=s[2*self._K+choice])

        return evalSamples

    def sample_deconvolution(self, samples = 1):
        weightMax = np.max(self.dynestyModel.results["logz"])
        weights = np.exp(self.dynestyModel.results["logz"]-weightMax)
        weights = weights/np.sum(weights)
        choices = np.random.choice(range(len(weights)), samples, p=weights)
        
        evalSamples = np.zeros(samples)
        for j,i in enumerate(choices):
            s = self.dynestyModel.results["samples"][i][3*self._K:3*self._K+self._Kc]
            choice = np.random.choice(range(self._Kc),p=s)
            s = self.dynestyModel.results["samples"][i]
            evalSamples[j] = norm.rvs(loc=s[3*self._K+self._Kc+choice],scale=s[3*self._K+2*self._Kc+choice])

        return evalSamples

    def sample_convolution(self, samples = 1):
        weightMax = np.max(self.dynestyModel.results["logz"])
        weights = np.exp(self.dynestyModel.results["logz"]-weightMax)
        weights = weights/np.sum(weights)
        choices = np.random.choice(range(len(weights)), samples, p=weights)
        
        evalSamples = np.zeros(samples)
        for j,i in enumerate(choices):
            s = self.dynestyModel.results["samples"][i][0:self._K]
            choice = np.random.choice(range(self._K),p=s)
            s = self.dynestyModel.results["samples"][i][3*self._K:3*self._K+self._Kc]
            choice2 = np.random.choice(range(self._Kc),p=s)
            s = self.dynestyModel.results["samples"][i]
            evalSamples[j] = norm.rvs(loc=s[self._K+choice]+s[3*self._K+self._Kc+choice2],scale=np.sqrt(s[2*self._K+choice]**2+s[3*self._K+2*self._Kc+choice2]**2))

        return evalSamples
