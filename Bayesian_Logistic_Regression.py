# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 18:22:23 2018

@author: bsingh46
"""

import pymc as pm, pandas as pd, seaborn
import numpy as np
import pymc as pm
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import time


#from io import StringIO

# =============================================================================
#
# Attribute Information: (class attribute has been moved to last column)
#
#    #  Attribute                     Domain
#    -- -----------------------------------------
#    1. Sample code number            id number
#    2. Clump Thickness               1 - 10
#    3. Uniformity of Cell Size       1 - 10
#    4. Uniformity of Cell Shape      1 - 10
#    5. Marginal Adhesion             1 - 10
#    6. Single Epithelial Cell Size   1 - 10
#    7. Bare Nuclei                   1 - 10
#    8. Bland Chromatin               1 - 10
#    9. Normal Nucleoli               1 - 10
#   10. Mitoses                       1 - 10
#   11. Class:                        (2 for benign, 4 for malignant)
#
# =============================================================================

### READ THE BREAST CANCER WISCONSIN DATASET

df = pd.read_csv('breast-cancer-wisconsin.data.txt', sep=",", header=None)

### DATA CLEANING

## remove garbage values
df=df.ix[~(df[6]=="?")]
## replace 2's and 4's with 0's and 1's
df[11]=np.where(df[10]==2,0,1)

#df.describe()
#df.dtypes
##convert datatypes to numeric
df=df.convert_objects(convert_numeric=True)

###GLM MODEL - CLASSICAL LOGISTIC REGRESSION

dta = df[[1,2,3,4,5,6,7,8,9,11]].copy()
dta.columns=['beta_clump_t', 'beta_cell_size', 'beta_cell_shape', 'beta_marg',
             'beta_epi', 'beta_bare_nucl',  'beta_chromatin','beta_norm_nuclei','beta_mito','CANCER']

corr = dta.corr()
## create a correlation heatmap
corr_heatmapsns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#generate box plots for EDA
sns.boxplot(dta.CANCER,dta.beta_clump_t)
sns.boxplot(dta.CANCER,dta.beta_cell_size)
sns.boxplot(dta.CANCER,dta.beta_cell_shape)
sns.boxplot(dta.CANCER,dta.beta_marg)
sns.boxplot(dta.CANCER,dta.beta_epi)
sns.boxplot(dta.CANCER,dta.beta_bare_nucl)
sns.boxplot(dta.CANCER,dta.beta_chromatin)
sns.boxplot(dta.CANCER,dta.beta_norm_nuclei)
sns.boxplot(dta.CANCER,dta.beta_mito)

## define formula
formula = 'CANCER ~ beta_clump_t+beta_cell_size+ beta_cell_shape+beta_marg+beta_epi+beta_bare_nucl+beta_chromatin+beta_norm_nuclei+beta_mito'

## evaluate Logistic regression
mod1 = smf.glm(formula=formula, data=dta, family=sm.families.Binomial()).fit()
mod1.summary()


### LOGISTIC REGRESSION - BAYESIAN APPROACH

#df.head(5)
x1 = df[1]
x2 = df[2]
x3 = df[3]
x4=df[7]
y=df[11]


## define hyper priors of our model
tau = pm.Gamma('tau', 1.e-3, 1.e-3, value=10.)
sigma = pm.Lambda('sigma', lambda tau=tau: tau**-.5)
beta0 =  pm.Normal('beta0',  0., 1e-6, value=0.)
beta_clump_t =  pm.Normal('beta_clump_t',  0., 1e-6, value=0.)
beta_cell_size =  pm.Normal('beta_cell_size',  0., 1e-6, value=0.)
beta_cell_shape = pm.Normal('beta_cell_shape',  0., 1e-6, value=0.)
beta_chromatin = pm.Normal('beta_chromatin',  0., 1e-6, value=0.)




########################################   MODEL 1  ###################################################

## given, betas  and observed x we predict y
# "model" the observed y values: again, I reiterate that PyMC treats y as
# evidence -- as fixed; it's going to use this as evidence in updating our belief
# about the "unobserved" parameters (b0, b1, and err), which are the
# things we're interested in inferring after all

logit_p =  (beta0 + beta_clump_t*x1 + beta_cell_size*x2 + beta_cell_shape*x3 + beta_chromatin*x4)
@pm.observed
def y(logit_p=logit_p, value=df[11]):
    return pm.bernoulli_like(df[11], pm.invlogit(logit_p))

## sample by MCMC
m = pm.MCMC([beta0, beta_clump_t, beta_cell_size, beta_cell_shape, beta_chromatin, tau, sigma, logit_p, y],calc_deviance=True)
m.sample(100000, 1000)
D1=m.deviance




########################################   MODEL 2  ###################################################

logit_p= (beta0 + beta_clump_t*x1 + beta_cell_size*x2 + beta_cell_shape*x3)
@pm.observed
def y1(logit_p=logit_p, value=df[11]):
    return pm.bernoulli_like(df[11], pm.invlogit(logit_p))

m = pm.MCMC([beta0, beta_clump_t, beta_cell_size, beta_cell_shape, tau, sigma, logit_p, y],calc_deviance=True)
m.sample(100000, 1000)
D2=m.deviance


########################################   MODEL 3  ###################################################
logit_p= (beta0 + beta_clump_t*x1 + beta_cell_size*x2 )
@pm.observed
def y(logit_p=logit_p, value=df[11]):
    return pm.bernoulli_like(df[11], pm.invlogit(logit_p))

logit_p= (beta0 + beta_clump_t*x1 + beta_cell_size*x2 )

m = pm.MCMC([beta0, beta_clump_t, beta_cell_size,  tau, sigma, logit_p, y],calc_deviance=True)
m.sample(100000, 1000)
D3=m.deviance


########################################   MODEL 4  ###################################################

logit_p= (beta0 + beta_clump_t*x1 )
@pm.observed
def y(logit_p=logit_p, value=df[11]):
    return pm.bernoulli_like(df[11], pm.invlogit(logit_p))

logit_p= (beta0 + beta_clump_t*x1)

m = pm.MCMC([beta0, beta_clump_t,  tau, sigma, logit_p, y],calc_deviance=True)
m.sample(100000, 1000)
D4=m.deviance


########################################   MODEL 5  ###################################################
logit_p= (beta0 + beta_cell_size*x2 )
@pm.observed
def y(logit_p=logit_p, value=df[11]):
    return pm.bernoulli_like(df[11], pm.invlogit(logit_p))

logit_p= (beta0 +  beta_cell_size*x2 )

m = pm.MCMC([beta0,  beta_cell_size,  tau, sigma, logit_p, y],calc_deviance=True)
m.sample(100000, 1000)
D5=m.deviance






### BIC for a model with given deviance, number of predictors and number of obs

def BIC(dev,p,n):
    Deviance=dev
    BIC = Deviance + p*np.log(n)
    return(BIC)


 ##Calculate BIC for all models
BIC1=BIC(D1,5,len(df))
BIC2=BIC(D2,4,len(df))
BIC3=BIC(D3,3,len(df))
BIC4=BIC(D4,2,len(df))
BIC5=BIC(D5,2,len(df))


print(D1,D2,D3,D4,D5)
print(BIC1,BIC2,BIC3,BIC4,BIC5)


###### Model 1 has the least deviance

for node in [beta0,beta_clump_t,beta_cell_size,beta_cell_shape,beta_chromatin]:
    stats=node.stats()
    print(node.__name__,stats['mean'],stats['standard deviation'])

# =============================================================================
#MODEL 1 PARAMETERS
# beta0 -585.145665803 627.066760615
# betaSalad -6.884676104 683.684380304
# betaSandwich -502.058009768 790.46217509
# betaWater 1194.59477175 599.309635069
# =============================================================================



########################   LOG REGR ON FULL MODEL   #########################
df = pd.read_csv('breast-cancer-wisconsin.data.txt', sep=",", header=None)

###### Remove Missing Values
df=df.ix[~(df[6]=="?")]

df[11]=np.where(df[10]==2,0,1)

df.describe()


df=df.convert_objects(convert_numeric=True)
df.dtypes

x1 = df[1]
x2 = df[2]
x3 = df[3]
x4 = df[4]
x5 = df[5]
x6 = df[6]
x7=df[7]
x8 = df[8]
x9=df[9]
y=df[11]


### hyperpriors
tau = pm.Gamma('tau', 1.e-3, 1.e-3, value=10.)
sigma = pm.Lambda('sigma', lambda tau=tau: tau**-.5)

### parameters
# fixed effects
beta0 =  pm.Normal('beta0',  0., 1e-6, value=0.)
beta_clump_t =  pm.Normal('beta_clump_t',  0., 1e-6, value=0.)
beta_cell_size =  pm.Normal('beta_cell_size',  0., 1e-6, value=0.)
beta_cell_shape = pm.Normal('beta_cell_shape',  0., 1e-6, value=0.)
beta_marg =  pm.Normal('beta_marg',  0., 1e-6, value=0.)
beta_epi =  pm.Normal('beta_epi',  0., 1e-6, value=0.)
beta_bare_nucl = pm.Normal('beta_bare_nucl',  0., 1e-6, value=0.)
beta_chromatin = pm.Normal('beta_chromatin',  0., 1e-6, value=0.)
beta_norm_nuclei = pm.Normal('beta_norm_nuclei',  0., 1e-6, value=0.)
beta_mito = pm.Normal('beta_mito',  0., 1e-6, value=0.)


logit_p =  (beta0 + beta_clump_t*x1 + beta_cell_size*x2 + beta_cell_shape*x3 +
            beta_marg*x4 + beta_epi*x5 + beta_bare_nucl*x6 + beta_chromatin*x7 +
            beta_norm_nuclei*x8 + beta_mito*x9)
@pm.observed
def y(logit_p=logit_p, value=df[11]):
    return pm.bernoulli_like(df[11], pm.invlogit(logit_p))

m = pm.MCMC([beta0, beta_clump_t, beta_cell_size, beta_cell_shape, beta_marg,
             beta_epi, beta_bare_nucl,  beta_chromatin,beta_norm_nuclei,beta_mito,
             tau, sigma, logit_p, y],calc_deviance=True)
a=time.time()
m.sample(100000, 1000)
b=time.time()
Time_full_model=b-a
D1=m.deviance

full_model_BIC=BIC(D1,4,len(df))



##### Plotting Results

pm.Matplot.plot(m)

#### Print out each value

for node in [beta0, beta_clump_t, beta_cell_size, beta_cell_shape, beta_marg,beta_epi, beta_bare_nucl,  beta_chromatin,beta_norm_nuclei,beta_mito]:
    stats=node.stats()
    print(node.__name__,stats['mean'],stats['standard deviation'])


#### TURNS OUT BIC FOR FULL MODEL IS BETTER THAN MODEL 1



