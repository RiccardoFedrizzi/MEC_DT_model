import random
import copy, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from statistics import mean, stdev
from math import sqrt

from scipy.optimize import curve_fit

from sklearn import neighbors
from sklearn.metrics import mean_squared_error

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

nucs = ['nuc1','nuc2','nuc3']
labels = ['VM','BM','CONT']
colors = ['C0','C1','C2','C3','C4','C5','C6']

metrics = ['mbps','power', 'cpu_perc']

bwtlist = np.arange(100, 800, 100) 
cpulist = np.arange(0 , 100, 10 )
# Matrix Array [N,2] containing the combinations of cpu/bwt 
cpu_bwt_grid = np.array(np.meshgrid(cpulist, bwtlist)).T.reshape(-1, 2)

# x is in the form [ cpu , throughput ]
def poly3d( x , p00,p01,p02,p03,p10,p11,p12,p20,p21,p30):
    return (    p00                     +
                p01 * x[0]              +
                p02 * x[0]**2           +
                p03 * x[0]**3           +
                p10           * x[1]    +
                p11 * x[0]    * x[1]    +
                p12 * x[0]**2 * x[1]    +
                p20           * x[1]**2 +
                p21 * x[0]    * x[1]**2 +
                p30           * x[1]**3 )

class MyPolyfit:
    """ Class to use Poly fit as KNN and SVR
    The params modeling the KPIs are ordered as in the training set: [power, cpu_perc, mbps]
    """
    def __init__(self, degree=3):
        self.params = [[],[],[]]

    def fit(self, X, Y):
        X_in = X.transpose()
        for i in range(Y.shape[1]):
            self.params[i] , _ = curve_fit( poly3d, X_in, Y[:,i] )

    def predict(self, X):
        pred = np.empty(shape=( X.shape[0] , len(self.params) ), dtype='float')
        for i in range(X.shape[0]):
            for kpi in range(len(self.params)):
                pred[i,kpi] = poly3d( X[i,:] , *self.params[kpi] )
        return pred


def training( ds, num_samples=None, m='svr', knn_neighbors=5, svr_c=100000, svr_gamma=0.1, svr_epsilon=0.1):
    if num_samples == None:
        train = ds
    else:
        train = ds.loc[(np.arange( 1,num_samples+1) , slice(None),slice(None))]
    cpuin = np.array( train.index.get_level_values('cpu') ) 
    bwtin = np.array( train.index.get_level_values('bwt') ) 
    X_in = np.column_stack(  (cpuin,bwtin))
    Y_in = train.values

    if m=='knn':
        model = neighbors.KNeighborsRegressor(n_neighbors = knn_neighbors)
    if m=='svr':
        # svr = SVR(kernel="rbf", C=svr_c, gamma=svr_gamma, epsilon=svr_epsilon)
        svr = SVR(kernel="rbf", C=svr_c)#, gamma=svr_gamma, epsilon=svr_epsilon)
        model = MultiOutputRegressor(svr)
    if m=='poly':
        model = MyPolyfit(m)

    model.fit( X_in , Y_in)

    return model

##### Slice the dataset
def slice_ds(ds , time=None,cpu=None,bwt=None):
    slice_t=time
    slice_c=cpu
    slice_b=bwt
    if time==None: slice_t = slice(None)
    if cpu ==None: slice_c = slice(None)
    if bwt ==None: slice_b = slice(None)

    return ds.loc[ (slice_t,slice_b,slice_c) ]


def filter_all_ds( ds_in ):
    ds_out = pd.DataFrame()
    for cpubwt in range(cpu_bwt_grid.shape[0]):
        cpu = cpu_bwt_grid[cpubwt,0]
        bwt = cpu_bwt_grid[cpubwt,1]

        ds = slice_ds( ds_in , cpu=cpu, bwt=bwt )

        measures_to_discard=5
        ds_new = copy.deepcopy(ds)
        # ds_new.drop(ds_new.tail(measures_to_discard).index, inplace = True)
        ds_new.drop( index=ds_new.index[:measures_to_discard] , inplace=True )
        
        for kpi in ds:
            meas = ds[ kpi ].values
            meas_filt = copy.deepcopy(meas)

            alpha = 0.2
            for i in range(1,len(meas)):
                meas_filt[i] = alpha*meas[i] + (1-alpha)*meas_filt[i-1]
            meas_filt = meas_filt[measures_to_discard:len(meas_filt)]
            ds_new[kpi] = meas_filt

        ds_out = pd.concat( [ds_out , ds_new] , axis=0 )

    return ds_out

# return a DS with the average KPI value for each WP
def mean_ds( ds_in ):
    ds_out = pd.DataFrame()
    for cpubwt in range(cpu_bwt_grid.shape[0]):
        cpu = cpu_bwt_grid[cpubwt,0]
        bwt = cpu_bwt_grid[cpubwt,1]

        ds_tmp = slice_ds( ds_in , cpu=cpu, bwt=bwt )
        ds_new = ds_tmp.head(1)
        ds_new = copy.deepcopy(ds_new)
        mean = np.mean( ds_tmp.values ,axis=0 )
        for i in range(len(mean)):
            ds_new.iloc[0,i] = mean[i]
            
        ds_out = pd.concat( [ds_out , ds_new] , axis=0 )

    return ds_out
