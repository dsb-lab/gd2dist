import numpy as np
import numpy.random as rd
import scipy.stats as st
import matplotlib.pyplot as plt
import pickle as pk
import sys

#Definir les funcions per a fer test
def f_norm(x):
    return st.norm.pdf(x,0,1)

def f_skew(x):
    return st.skewnorm.pdf(x,7,loc=-1.32,scale=1.65)

def f_bimodal_asym(x):
    std1 = 0.41
    std2 = 0.71
    sample = 0.4*st.norm.pdf(x,-0.82-0.16,std1)+0.6*st.norm.pdf(x,0.82-0.16,std2)
    return sample

def f_bimodal_sym(x):
    std1 = 0.5
    std2 = 0.5
    sample = 0.5*st.norm.pdf(x,-0.9,std1)+0.5*st.norm.pdf(x,0.9,std2)
    return sample

def f_trimodal_asym(x):
    std1 = 0.65
    std2 = 0.35
    std3 = 0.35
    mu1 = -1.7
    mu2 = 0
    mu3 = 1.3
    sample = 0.2*st.norm.pdf(x,mu1+0.08,std1)+0.6*st.norm.pdf(x,mu2+0.08,std2)+0.2*st.norm.pdf(x,mu3+0.08,std3)
    return sample

def f_trimodal_sym(x):
    std1 = 0.65
    std2 = 0.15
    std3 = std1
    mu1 = -1.3
    mu2 = 0
    mu3 = -mu1
    sample = 0.25*st.norm.pdf(x,mu1,std1)+0.5*st.norm.pdf(x,mu2,std2)+0.25*st.norm.pdf(x,mu3,std3)
    return sample

def f_student(x):
    return st.t.pdf(x,3,scale=0.585)

def f_laplace(x):
    return st.laplace.pdf(x,loc=0,scale=0.72)

def f_laplace_sharp(x):
    std1 = 0.2
    std2 = 1
    sample = 0.5*st.laplace.pdf(x,0,std1)+0.5*st.laplace.pdf(x,0,std2)
    return sample

def f_test1(x):
    std1 = 0.5
    std2 = 0.5
    sample = 0.5*st.norm.pdf(x,-0.9,std1)+0.5*st.norm.pdf(x,0.9,std2)
    return sample

def f_test2(x):
    std1 = 0.6
    std2 = 0.6
    sample = 0.8*st.norm.pdf(x,-1.05+0.62,std1)+ 0.2*st.norm.pdf(x,1.05+0.62,std2)
    return sample