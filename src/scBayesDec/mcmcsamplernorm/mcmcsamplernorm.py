from .mcmcposteriorsamplernorm import fit
from scipy.stats import norm
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.cluster import KMeans

from ..shared_functions import *

class mcmcsamplernorm:
    """
    Class for the mcmc sampler of the deconvolution gaussian model
    """
    def __init__(self, K=1, Kc=1):
        """
        Constructor of the class


        Parameters
        -------------
            K: int, Number of components of the noise distribution
            Kc: int, Number of components of the convolved distribution
            **kwargs:
                alpha: float, parameter to determine the hyperprior of the noise weight components
                alphac: float, parameter to determine the hyperprior of the target weight components

        """
        
        self.K = K
        self.Kc = Kc

        self.fitted = False

        return

    def fit(self, dataNoise, dataConvolution, iterations = 1000, ignored_iterations = 1000, chains = 1, priors = None, method_initialisation = "kmeans", initial_conditions = [], show_progress = True, seed = 0):
        """
        Fit the model to the posterior distribution


        Parameters
        -------------
            dataNoise: list/npArray, 1D array witht he data of the noise
            dataConvolution: list/npArray, 1D array witht he data of the convolution
            iterations: int, number of samples to be drawn and stored for each chain during the sampling
            ignored_iterations: int, number of samples to be drawn and ignored for each chain during the sampling
            chains: int, number of independently initialised realisations of the markov chain
            priors: array, parameter of the priors gamma distribution acording to the definition of the wikipedia
            kconst: float, parameter k of the prior gamma distribution
            initialConditions: list, 1D array with all the parameters required to initialise manually all the components of all the chains the chains
            show_progress: bool, indicate if the method should show the progress in the generation of the new data
            seed: int, value to initialise the random generator and obtain reproducible results

        Returns
        ---------------
            Nothing
        """

        self.data = dataNoise
        self.datac = dataConvolution
        self.iterations = iterations
        self.ignored_iterations = ignored_iterations
        self.chains = chains

        if priors == None:
            self.priors = np.zeros(10)
            self.priors[0] = 1/self.K
            self.priors[1] = (np.max(dataNoise)+np.min(dataNoise))/2
            self.priors[2] = 3*(np.max(dataNoise)-np.min(dataNoise))
            self.priors[3] = 10*(np.max(dataNoise)-np.min(dataNoise))
            self.priors[4] = 1.1

            self.priors[5] = 1/self.Kc
            self.priors[6] = (np.max(dataConvolution)+np.min(dataConvolution))/2
            self.priors[7] = 3*(np.max(dataConvolution)-np.min(dataConvolution))
            self.priors[8] = 10*(np.max(dataConvolution)-np.min(dataConvolution))
            self.priors[9] = 1.1
        else:
            self.priors = priors

        if initial_conditions != []:
            self.initial_conditions = initial_conditions
        elif method_initialisation == "kmeans":
            K =self.K
            Kc = self.Kc
            y = np.zeros([chains,(K+Kc)*3])
            model = KMeans(n_clusters=K)
            model.fit(dataNoise.reshape(-1,1))
            ids = model.predict(dataNoise.reshape(-1,1))
            #Add weights autofluorescence
            for i in range(K):
                for j in range(chains):
                    y[j,i] = np.sum(ids==i)/len(ids)
            #Add means autofluorescence
            for i in range(K):
                for j in range(chains):
                    y[j,K+i] = np.mean(dataNoise[ids==i])
            #Add std autofluorescence
            for i in range(K):
                for j in range(chains):
                    y[j,2*K+i] = np.std(dataNoise[ids==i])
                        
            model = KMeans(n_clusters=Kc)
            model.fit(dataConvolution.reshape(-1,1))
            ids = model.predict(dataConvolution.reshape(-1,1))
            #Add weights autofluorescence
            for i in range(Kc):
                for j in range(chains):
                    y[j,3*K+i] = np.sum(ids==i)/len(ids)
            #Add means autofluorescence
            for i in range(Kc):
                for j in range(chains):
                    y[j,3*K+Kc+i] = np.mean(dataConvolution[ids==i])
            #Add std autofluorescence
            for i in range(Kc):
                for j in range(chains):
                    y[j,3*K+2*Kc+i] = np.std(dataConvolution[ids==i])
            self.initial_conditions = y
        elif method_initialisation == "random":
            self.initial_conditions = []
        else: 
            self.initial_conditions = []

        self.samples = np.array(fit(dataNoise, dataConvolution, ignored_iterations, iterations, chains, self.K, self.Kc, self.priors, self.initial_conditions, show_progress, seed))
        
        self.fitted = True

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
            pk.dump({"K":self.K, "Kc":self.Kc, "priors": self.priors, "iterations": self.iterations,
                     "ignored_iterations": self.ignored_iterations,
                     "chains":self.chains, "samples":self.samples}, pickling_on)
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
        self.Kc = aux ["Kc"]
        self.priors = aux["priors"]
        self.iterations = aux["iterations"]
        self.ignored_iterations = aux["ignored_iterations"]
        self.chains = aux["chains"]
        self.samples = aux["samples"]

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
            return  np.array(sample_autofluorescence(self.samples,self.K,self.Kc,size=size))
        elif style=="single":
            if pos == None:
                pos = np.random.choice(range(len(self.samples))) 
                return  np.array(sample_autofluorescence(self.samples,self.K,self.Kc,size=size,pos=pos))
            else:
                return  np.array(sample_autofluorescence(self.samples,self.K,self.Kc,size=size,pos=pos))

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
            return  np.array(sample_deconvolution(self.samples,self.K,self.Kc,size=size))
        elif style=="single":
            if pos == None:
                pos = np.random.choice(range(len(self.samples))) 
                return  np.array(sample_deconvolution(self.samples,self.K,self.Kc,size=size,pos=pos))
            else:
                return  np.array(sample_deconvolution(self.samples,self.K,self.Kc,size=size,pos=pos))

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
            return  np.array(sample_convolution(self.samples,self.K,self.Kc,size=size))
        elif style=="single":
            if pos == None:
                pos = np.random.choice(range(len(self.samples))) 
                return  np.array(sample_convolution(self.samples,self.K,self.Kc,size=size,pos=pos))
            else:
                return  np.array(sample_convolution(self.samples,self.K,self.Kc,size=size,pos=pos))

        return

    def score_autofluorescence(self, x, percentiles = [5, 95], size = 100):
        """
        Evaluate the mean and percentiles of the the pdf at certain position acording to the convolved distribution

        Parameters
        -------------
            x: list/array, positions where to evaluate the distribution
            percentiles: list/array, percentiles to be evaluated
            size: int, number of samples to draw from the posterior to make the statistics, bigger numbers give more stability

        Returns
        -------------
            list: list, 2D array with the mean and all the percentile evaluations at all points in x
        """
        yT = []
        for l in range(size):
            i = np.random.choice(self.iterations)
            y = np.zeros(len(x))
            for k in range(self.K):
                mu = self.samples[i,self.K+k]
                sigma = self.samples[i,2*self.K+k]
                    
                y += self.samples[i,k]*norm.pdf(x,loc=mu,scale=sigma)
            yT.append(y)

        return  np.mean(yT,axis=0),np.percentile(yT,percentiles,axis=0)

    def score_deconvolution(self, x, percentiles = [5, 95], size = 100):
        """
        Evaluate the mean and percentiles of the the pdf at certain position acording to the deconvolved distribution

        Parameters
        -------------
            x: list/array, positions where to evaluate the distribution
            percentiles: list/array, percentiles to be evaluated
            size: int, number of samples to draw from the posterior to make the statistics, bigger numbers give more stability

        Returns
        -------------
            list: list, 2D array with the mean and all the percentile evaluations at all points in x
        """

        yT = []
        for l in range(size):
            i = np.random.choice(self.iterations)
            y = np.zeros(len(x))
            for j in range(self.Kc):
                mu = self.samples[i,3*self.K+self.Kc+j]
                sigma = self.samples[i,3*self.K+2*self.Kc+j]
                    
                y += self.samples[i,3*self.K+j]*norm.pdf(x,loc=mu,scale=sigma)
            yT.append(y)

        return  np.mean(yT,axis=0),np.percentile(yT,percentiles,axis=0)

    def score_convolution(self, x, percentiles = [5, 95], size = 100):
        """
        Evaluate the mean and percentiles of the the pdf at certain position acording to the convolved distribution

        Parameters
        -------------
            x: list/array, positions where to evaluate the distribution
            percentiles: list/array, percentiles to be evaluated
            size: int, number of samples to draw from the posterior to make the statistics, bigger numbers give more stability

        Returns
        -------------
            list: list, 2D array with the mean and all the percentile evaluations at all points in x
        """

        yT = []
        for l in range(size):
            i = np.random.choice(self.iterations)
            y = np.zeros(len(x))
            for j in range(self.Kc):
                for k in range(self.K):
                    mu1 = self.samples[i,self.K+k]
                    mu2 = self.samples[i,3*self.K+self.Kc+j]
                    sigma1 = self.samples[i,2*self.K+k]
                    sigma2 = self.samples[i,3*self.K+2*self.Kc+j]
                    mu = mu1
                    s = np.sqrt(sigma1**2+sigma2**2)
                    
                    y += self.samples[i,k]*self.samples[i,3*self.K+j]*norm.pdf(x,loc=mu,scale=s)
            yT.append(y)

        return  np.mean(yT,axis=0),np.percentile(yT,percentiles,axis=0)


    def sampler_statistics(self, sort="weight"):
        """
        Show statistics of correct mixing of the mcmc sampler
        
        Args:
            sort: ["weight", "none", "means"], method for sorting the samples from the different chains

        Returns
        -------------
            DataFrame: DataFrame the mean, std, percentiles, mixing ratio(rhat) and effective number of samples for each parameter of the model
        """
        
        self.sampler_statistics = pd.DataFrame(columns=["Mean","Std","5%","50%","95%","Rhat","Neff"])

        samples = self.samples.copy()

        if sort == "weight":
            argsort = np.argsort(samples[:,0:self.K],axis=1)
            samples[:,0:self.K] = np.take_along_axis(samples[:,0:self.K],argsort,axis=1)
            samples[:,self.K:2*self.K] = np.take_along_axis(samples[:,self.K:2*self.K],argsort,axis=1)
            samples[:,2*self.K:3*self.K] = np.take_along_axis(samples[:,2*self.K:3*self.K],argsort,axis=1)

            argsort = np.argsort(samples[:,3*self.K:3*self.K+self.Kc],axis=1)
            samples[:,3*self.K:3*self.K+self.Kc] = np.take_along_axis(samples[:,3*self.K:3*self.K+self.Kc],argsort,axis=1)
            samples[:,(3*self.K+self.Kc):(3*self.K+2*self.Kc)] = np.take_along_axis(samples[:,(3*self.K+self.Kc):(3*self.K+2*self.Kc)],argsort,axis=1)
            samples[:,(3*self.K+2*self.Kc):(3*self.K+3*self.Kc)] = np.take_along_axis(samples[:,(3*self.K+2*self.Kc):(3*self.K+3*self.Kc)],argsort,axis=1)

        if sort == "mean":
            argsort = np.argsort(samples[:,self.K:2*self.K],axis=1)
            samples[:,0:self.K] = np.take_along_axis(samples[:,0:self.K],argsort,axis=1)
            samples[:,self.K:2*self.K] = np.take_along_axis(samples[:,self.K:2*self.K],argsort,axis=1)
            samples[:,2*self.K:3*self.K] = np.take_along_axis(samples[:,2*self.K:3*self.K],argsort,axis=1)

            argsort = np.argsort(samples[:,3*self.K+self.Kc:3*self.K+2*self.Kc],axis=1)
            samples[:,3*self.K:3*self.K+self.Kc] = np.take_along_axis(samples[:,3*self.K:3*self.K+self.Kc],argsort,axis=1)
            samples[:,(3*self.K+self.Kc):(3*self.K+2*self.Kc)] = np.take_along_axis(samples[:,(3*self.K+self.Kc):(3*self.K+2*self.Kc)],argsort,axis=1)
            samples[:,(3*self.K+2*self.Kc):(3*self.K+3*self.Kc)] = np.take_along_axis(samples[:,(3*self.K+2*self.Kc):(3*self.K+3*self.Kc)],argsort,axis=1)

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
