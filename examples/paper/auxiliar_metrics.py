#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
from scipy.signal import correlate
from pickle import load, dump

## Deconvolution metrics

def MISE(data1, data2, dx):
    val = np.sum((data1-data2)**2)*dx
    
    return val

def AISE(data1, data2, dx):
    val = 1-0.5*np.sum(np.abs(data1-data2))*dx
    
    return val