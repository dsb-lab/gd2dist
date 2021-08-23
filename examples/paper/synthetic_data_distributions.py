import numpy as np
import numpy.random as rd
import scipy.stats as st
import matplotlib.pyplot as plt
import pickle as pk
import sys

#Definir les funcions per a fer test
def f_norm(size):
    return st.norm.rvs(0,1,size)

def f_skew(size):
    return st.skewnorm.rvs(7,loc=-1.32,scale=1.65,size=size)

def f_bimodal_asym(size):
    std1 = 0.41
    std2 = 0.71
    sample = np.concatenate([st.norm.rvs(-0.82-0.16,std1,int(size*0.4)),st.norm.rvs(0.82-0.16,std2,int(size*0.6))])
    rd.shuffle(sample)
    return sample

def f_bimodal_sym(size):
    std1 = 0.5
    std2 = 0.5
    sample = np.concatenate([st.norm.rvs(-0.9,std1,int(size*0.5)),st.norm.rvs(0.9,std2,int(size*0.5))])
    rd.shuffle(sample)
    return sample

def f_trimodal_asym(size):
    std1 = 0.65
    std2 = 0.35
    std3 = 0.35
    mu1 = -1.7
    mu2 = 0
    mu3 = 1.3
    sample = np.concatenate([st.norm.rvs(mu1+0.08,std1,int(size*0.2)),st.norm.rvs(mu2+0.08,std2,int(size*0.6)),st.norm.rvs(mu3+0.08,std3,int(size*0.2))])
    rd.shuffle(sample)
    return sample

def f_trimodal_sym(size):
    std1 = 0.65
    std2 = 0.15
    std3 = std1
    mu1 = -1.3
    mu2 = 0
    mu3 = -mu1
    sample = np.concatenate([st.norm.rvs(mu1,std1,int(size*0.25)),st.norm.rvs(mu2,std2,int(size*0.5)),st.norm.rvs(mu3,std3,int(size*0.25))])
    rd.shuffle(sample)
    return sample

def f_student(size):
    return st.t.rvs(3,scale=0.585,size=size)

def f_laplace(size):
    return st.laplace.rvs(loc=0,scale=0.72,size=size)

def f_laplace_sharp(size):
    std1 = 0.2
    std2 = 1
    sample = np.concatenate([st.laplace.rvs(0,std1,int(size*0.5)),st.laplace.rvs(0,std2,int(size*0.5))])
    rd.shuffle(sample)
    return sample

def f_test1(size):
    std1 = 0.5
    std2 = 0.5
    sample = np.concatenate([st.norm.rvs(-0.9,std1,int(size*0.5)),st.norm.rvs(0.9,std2,int(size*0.5))])
    rd.shuffle(sample)
    return sample

def f_test2(size):
    std1 = 0.6
    std2 = 0.6
    sample = np.concatenate([st.norm.rvs(-1.05+0.62,std1,int(size*0.8)), st.norm.rvs(1.05+0.62,std2,int(size*0.2))])
    rd.shuffle(sample)
    return sample