#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
from scipy.signal import correlate
from pickle import load, dump

def natural_estimate(data,w):
    sol = [0 for i in range(len(w))]
    l = len(data)
    for n,i in enumerate(w):
        sol[n] = np.sum(np.exp(1j*i*data)/l)
    return np.array(sol)

# Density Deconvolution using Fourier inversion along the lines of M.H. Neumann 1996
# Parameters:
# aut         :data noise
# data        :corrupted data
# bandwidth   :bandwidth of the deconvolution kernel
# dw          :stepsizee of the frequency space
# w_lims      :limits of the frequency space sample
# dx          :stepsize of the real space
# x_lims      :limits of the real space sample
# cut_off     :If true, the deconvolution incorporates a cutoff in the lines of M.H. Newmann 1996
def FT_deconv(aut,data,d1=1,d2=1,dw=0.01,w_lims=[-100,100],dx=0.1,x_lims=[-100,100],cut_off=True):
    w = np.arange(w_lims[0],w_lims[1],dw)
    x = np.arange(x_lims[0],x_lims[1],dx)
    hn = np.sqrt((len(aut)+1)**(1/(2*d1+2*d2))-1)
    Kw = st.norm.pdf(w,0,hn)
    phiY = natural_estimate(data,w)
    phie = natural_estimate(aut,w)
    if cut_off==True:
        #Generate the cutoff
        I = np.ones(len(Kw))
        I[np.where(phie<=len(aut)**-0.5)[0]]=0
        deconv = np.dot(I*phiY*Kw/phie,np.exp(-1j*w.reshape(-1,1)*x))*dw
    else:
        deconv = np.dot(phiY*Kw/phie,np.exp(-1j*w.reshape(-1,1)*x))*dw

    return deconv, x

def MISE(data1, data2, dx):
    val = np.sum((data1-data2)**2)*dx
    
    return val

def AISE(data1, data2, dx):
    val = 1-0.5*np.sum(np.abs(data1-data2))*dx
    
    return val