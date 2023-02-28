#!/usr/bin/env python

#  Copyright 2023 
#  Center for Global Discovery and Conservation Science, Arizona State University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
# Original authors:
#   Phillip Brodrick, phil.brodrick AT gmail.com
#   Dana Chadwick, dana.chadwick AT jpl.nasa.gov
#
# ...reimplementing R code from the autoplsr R package described in:
#   Schmidtlein, S., H. Feilhauer, and H. Bruelheide. 2012. Mapping plant
#    strategy types using remote sensing. Jour. of Veg. Sci. 23:395–405.
#	 
# Modified by:
#   Nicholas Vaughn, nickvaughn AT asu.edu
#
# PLSR-Press
# This methodology was developed in this current form by former and current
# members of the Asner Lab at GDCS. Please give proper attribution when using
# this code for public:
#
#  Martin, R. E., K. D. Chadwick, P. G. Brodrick, L. Carranza-Jimenez,
#    N. R. Vaughn, and G. P. Asner. 2018. An Approach for Foliar Trait
#    Retrieval from Airborne Imaging Spectroscopy of Tropical Forests.
#    Remote Sensing 10:199

import argparse
parser = argparse.ArgumentParser(description="Run PLSR given a settings file")
parser.add_argument("settings_file",help="JSON-formatted settings file")
parser.add_argument("output_dir",help="Output directory")
args = parser.parse_args()
#  args = parser.parse_args(["dr_plsr_settings.json","testing/output"])

import pandas as pd
import numpy as np

from scipy import stats
from sklearn import preprocessing
from sklearn import ensemble
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
import joblib
import time
from functools import partial
import json
import os
import warnings
from collections import OrderedDict

# User settings in the JSON file
################################
# csv file              : CSV file with all data - should have a row for each
#                         input spectra. Spectra can be grouped into clusters if
#                         there is a column identifying cluster ID. The spectral
#                         data should be column-wise with a column for each band
#                         in sequential order. Columns names can have a preface,
#                         e.g. B001, B002 ..., which can be removed to get a band
#                         index as an integer.
# band preface          : What preceeds numbers in the band columns - usually "B"
# chems                 : List of chems to fit, e.g. ["LMA"]
# chem transforms       : Matching list of transforms "log", "sqrt", "square", 
#                         "inverse" or "none" - like ["none"]
# cluster col           : Field that contains cluster id (assuming multiple 
#                         pixels in each cluster). If not cluster-based, 
#                         use "none"
# bad bands             : List of (1-based) band numbers removed before 
#                         normalization e.g. [1,2,3,4,211,212,213,214]
# ignore columns        : list of quoted column names that should be ignored in
#                         the analysis, e.g. ["ID","SomeStat"]
# brightness normalize  : true/false do brightness normalization
# ndvi minimum          : Minimum NDVI value, -1 for no limit
# ndvi maximum          : Maximum NDVI value, -1 for no limit
#                         NDVI will only be computed if one or more of the above 
#                         is != -1
# ndvi red band         : Band representing red for NDVI computation, e.g. 34
# ndvi nir band         : Band representing red for NDVI computation, e.g. 46
# brightness maximum    : Minimum NDVI value, -1 for no limit
# brightness minimum    : Maximum NDVI value, -1 for no limit
# iterations            : Number of iterations of the algorithm - for each
#                         iteration, the fraction of clusters specified in 
#                         "iteration holdout" will be used to make a test set
#                         and the model will be fit on the rest of the data
#                         (except that specified in "test set holdout"). Use -1
#                         to use jackknife mode, i.e. if negative value in
#                         "iteration holdout"
# iteration holdout     : Fraction of data used for validation at each iteraton -
#                         can be 0 for no validation set, use negative values
#                         (i.e. -n) to use jackknife mode, where n clusters are
#                         held out each time, and each iteration is mutually
#                         exclusive.
# test set holdout      : Fraction of data used for global holdout test set - can
#                         be 0 for no global holdout set, ignored in jackkife
#                         mode
# samples per cluster   : Specify a number of samples per cluster for training 
#                         data, use -1 for no limit
# min pixel per cluster : Mimimum number of samples per cluster
# max components        : Maximum number of components checked with press stat
# use degen removal     : true/false use procedure to find and remove degenerate
#                         (less significant) input features. Similar to the A4 
#                         function in autopls
# scale features        : true/false Fit scaler to input features
# n jobs                : Number of parallel jobs run by sklearn functions
# random seed           : Seed value for random number generator (-1 for none)

def get_plsr_vip_scores(model):
    ##Variable Importance in Projection - see
    ## Mehmood, T., K. H. Liland, L. Snipen, and S. Sæbø. 2012. A review of 
    ##   variable selection methods in Partial Least Squares Regression. 
    ##   Chemometrics and Intelligent Laboratory Systems 118:62–69.
    ## Code taken from:
    ## https://github.com/scikit-learn/scikit-learn/issues/7050#issuecomment-345208503
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
    return vips

#############################################################################
# Get PRESS for a specific number of components using a k-fold CV with
# degenerate feature removal algorithm
def internal_autoselect_iteration_degen_removal(X,Y,ncomp,degenpercent=10):
      comp = ncomp
      kf = KFold(n_splits=10,shuffle=True,random_state=10)
      kf.get_n_splits(X)
      kf_jk_perf = np.zeros((10,X.shape[1]))
      kf_coeff = np.zeros((10,X.shape[1]+1))
      ind = 0
      loc_score = []
      for train_index, test_index in kf.split(X):
          y_mean = Y[train_index].mean()
          x_mean = X[train_index,:].mean(axis=0)
          model = PLSRegression(n_components=ncomp,scale=False,tol=1e-9)
          model.fit(X[train_index,:],Y[train_index])
          kf_coeff[ind,1:] = np.squeeze(model.coef_)
          #  kf_coeff[ind,0] = model.y_mean_ - np.dot(model.x_mean_ , model.coef_)
          kf_coeff[ind,0] = y_mean - np.dot(x_mean , model.coef_)
          jack_pred = np.ones_like(Y) * np.nan

          # Jackknife proceedure - compute model performance with each band set to
          # all 0s to detect bands that don't do anything
          for jkn in range(0,X.shape[1]):
              jk_coeff = kf_coeff[ind,:].copy()
              jk_coeff[jkn] = 0
              pred = plsr_predict(jk_coeff,X[test_index,:])
              kf_jk_perf[ind,jkn] = np.sqrt(np.nanmean(np.power(pred-Y[test_index],2)))
          ind += 1

          # this is for the score without looking at dropping out features
          loc_score.append(np.sqrt(np.nanmean(np.power(pred-Y[test_index],2))))

 
      # now select for components.  Autopls looks at the p-values of a students t-test of significance on the coefficients - 
      # here we look at performance instead.  They appear to be roughly analagous overall. 
      out_score = []
      for train_index, test_index in kf.split(X):
          y_mean = Y[train_index].mean()
          x_mean = X[train_index,:].mean(axis=0)
          model = PLSRegression(n_components=ncomp,scale=False,tol=1e-9)
          model.fit(X[train_index,:],Y[train_index])
          out_score.append(np.sqrt(np.nanmean(np.power(model.predict(X[test_index,:])-Y[test_index],2))))
      model = PLSRegression(n_components=ncomp,scale=False,tol=1e-9)
      model.fit(X,Y)
      out_coeff = np.zeros(X.shape[1]+1)
      out_coeff[1:] = np.squeeze(model.coef_)
      #  out_coeff[0] = model.y_mean_ - np.dot(model.x_mean_ , model.coef_)
      out_coeff[0] = y_mean - np.dot(x_mean , model.coef_)

      return out_coeff, np.nanmean(out_score), get_plsr_vip_scores(model)

# Get PRESS for a specific number of components using a k-fold CV
def internal_autoselect_iteration(X,Y,ncomp):
    comp = ncomp
    kf = KFold(n_splits=10,shuffle=True,random_state=10)
    kf.get_n_splits(X)
    pred_y = np.ones_like(Y) * np.nan
    for train_index, test_index in kf.split(X):
        model = PLSRegression(n_components=ncomp,scale=False,tol=1e-9)
        model.fit(X[train_index,:],Y[train_index])
        pred_y[test_index] = model.predict(X[test_index,:]).squeeze()

    return np.sqrt(np.nanmean(np.power(pred_y - Y,2)))



# Check each number of components from 1 to max_comp, and fit final model with
# the number of components assocaited with best score. 
# Similar to A1 in autopls
def autoselect_plsr(X,Y,max_comp):
    complist = np.arange(1,max_comp+1)[::-1]
    scores = list(map(
        partial(internal_autoselect_iteration,X,Y), complist.tolist()))

    ncomp =  complist[np.argmin(scores)]
    #  print(pd.DataFrame({"NComp":complist,"score":scores}))
    #  print("Lowest score: {}".format(min(scores)))
    #  print("Selected number of components = {}".format(ncomp))

    y_mean = Y.mean()
    x_mean = X.mean(axis=0)
    model = PLSRegression(n_components=ncomp,scale=False,tol=1e-9)
    model.fit(X,Y)
    loc_coeff = np.array([
          *(y_mean - np.dot(x_mean , model.coef_)).tolist(),
          *np.squeeze(model.coef_).tolist()])
    return loc_coeff, get_plsr_vip_scores(model)


# Check each number of components from 1 to max_comp, and fit final model with
# the number of components assocaited with best score. Use degenerate feature
# removal proceedure for each of the runs
# Similar to the A4 function in autopls
def autoselect_plsr_degen_removal(X,Y,max_comp):
    complist = np.arange(2,max_comp)[::-1]
    returned_coeff = []
    returned_vip = []
    scores = []
    for comp in complist:
        rc, s, vip = internal_autoselect_iteration_degen_removal(X,Y, comp)
        returned_coeff.append(rc)
        returned_vip.append(vip)
        scores.append(s)

    scores = np.array(scores)
    loc_coeff = returned_coeff[np.argmin(scores)]
    loc_vip = returned_vip[np.argmin(scores)]

    return loc_coeff, loc_vip


# apply a forward or backward transformation, intended for chem data if needed
def apply_transform(Y,transform,invert=False):
    if (transform.lower() == 'none' or transform == None):
        return Y
    if (invert == False):
        if (transform == 'log'):
            return np.log(Y)
        if (transform == 'sqrt'):
            return np.sqrt(Y)
        if (transform == 'square'):
            return np.power(Y,2)
        if (transform == 'inverse'):
            return 1.0 / Y
    else:
        if (transform == 'log'):
            return np.exp(Y)
        if (transform == 'sqrt'):
            return np.power(Y,2)
        if (transform == 'square'):
            return np.sqrt(Y)
        if (transform == 'inverse'):
            return 1.0 / Y

# perform brightness normalization on either a matrix or a vector of reflectance coefficients
def brightness_normalize(X):
    norm = np.linalg.norm(X,axis=1)
    bnspec = X.astype(float) / norm.reshape(X.shape[0],1)
    return bnspec, norm

# Computed NDVI from specified columns for red and nir reflectance
def compute_ndvi(x,red,nir):
    ndvi = (x[:,nir] - x[:,red]) / (x[:,nir] + x[:,red])
    return ndvi

## Functions to find test and validation sets
def create_holdout_clusters(X_full,clusterids,fraction,reserved=None,selected_clusters=None):
    """Randomly select N*fraction clusters for inclusion in a test set. Will not
       include any rows already reserved"""
    if reserved is not None:
        available = ~reserved
    else:
        available = np.repeat(True,X_full.shape[0])
    if selected_clusters is None:
        unique_clusters = np.unique(clusterids[available])
        num_clusters = len(unique_clusters)
        num_test = int(num_clusters * fraction)
        selected_clusters = \
            np.random.choice(unique_clusters,size=num_test,replace=False)
    selected_rows = np.repeat(False,X_full.shape[0])
    for clust in selected_clusters:
        selected_rows[np.logical_and(clusterids == clust,available)] = True
    return selected_rows, list(selected_clusters)


def gather_training_clusters(X_full, clusterids, n_per_clust, reserved=None):
    """Randomly select up to N rows per cluster for inclusion in a training data
       set. If n_per_clust is < 1, then all rows for each cluster will be
       included. Will not include any rows already reserved"""
    if reserved is not None:
        available = ~reserved
    else:
        available = np.repeat(True,X_full.shape[0])
    unique_clusters = np.unique(clusterids[available])
    selected_rows = np.repeat(False,X_full.shape[0])
    for clust in unique_clusters:
        wclust = \
            np.where(np.logical_and(clusterids == clust,available))[0]
        if (n_per_clust > 0) and (len(wclust) > n_per_clust):
            wclust = np.random.choice(wclust,size=n_per_clust)
        selected_rows[wclust] = True
    return selected_rows


# basic plsr prediction assuming that the coefficient vector has the intercept as the first element,
# and coefficients matching the reflectance matrix as the rest of the elements
def plsr_predict(in_coeff,x_mat):
    return np.sum(in_coeff[1:].reshape(1,-1) * x_mat,axis=1) + in_coeff[0]

def plsr_rmse(y_pred, y_obs):
    return np.sqrt(np.nanmean(np.power(y_pred-y_obs,2)))

# print active performance results
def regression_results(y,x):
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.squeeze(x),np.squeeze(y))
    tmpdf = pd.DataFrame({
                    'slope':slope,
                    'intercept':intercept,
                    'r squared':r_value**2,
                    'rmse':plsr_rmse(x,y)}, index=[0])
    return(tmpdf)
 
    
def map_subset_boolean_array(full_membership, sub_membership):
    combined = np.zeros_like(full_membership)
    w_in = np.where(full_membership)[0]
    combined[w_in[sub_membership]] = True
    return combined

########################################### Begin Main Code ########################################


def main():
    
    ##Make sure output folder exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ## Write settings to output folder for posterity
    with open(args.settings_file) as fref:
        with open(os.path.join(args.output_dir,"saved_settings.json"),'w') as wref:
            wref.writelines(fref.readlines())

    ##Read in the user settings from JSON file
    with open(args.settings_file) as fref:
        user_settings = json.load(fref) 
    
    def get_setting(name):
        if not name in user_settings:
            raise RuntimeError("User conifig file is missing entry '{}'".format(name))
        return user_settings[name]
    
    try:
        np.random.seed(get_setting("random seed"))
    except:
        pass

    ##Read in the given CSV file
    df = pd.read_csv(get_setting("csv file"))
    
    ##Drop the specified ignore columns
    for name in get_setting("ignore columns"):
        df = df.drop(name,1)
    
    header =  list(df)
    #df = np.array(df)
    
    print(header)
    
    ########################## Data frame prep - get it cleaned up before looping through chems #################
    
    # get the chem list / chem transforms of interest
    chemlist = get_setting("chems")
    chemtransform_list = get_setting("chem transforms")

    # This will help figure out x columns later
    nonx_columns = list(get_setting("chems"))
      
    # Find or make a column of cluster ID
    if get_setting("cluster col").lower() != "none":
        nonx_columns.append(get_setting("cluster col"))
        try:
            cluster_col = header.index(get_setting("cluster col"))
        except ValueError as exc:
            raise RuntimeError("Cluster index column '{}' not found in CSV".format(get_setting("cluster col")))
    else:
        tmpname = "CLUSTID"
        print('Creating cluster ID column "{}"'.format(tmpname))
        nonx_columns.append(tmpname)
        cluster_col = 0
        df.insert(cluster_col,tmpname,np.arange(df.shape[0]))
        header.insert(cluster_col,tmpname)

    # If there is only one pixel per cluster then we will run in single pixel
    # mode
    uniq_clusters = np.unique(df.iloc[:,cluster_col])
    num_clusters = len(uniq_clusters)
    if df.shape[0] == num_clusters:
        if (get_setting("iteration holdout") <= 0) and \
            (get_setting("iterations") > 1):
            warnings.warn("Having no iteration holdout with multiple iterations"
                          +" is not suggested")

    # Identify the x columns by name and index, and remove bad bands
    # Also, identify bands used for ndvi
    x_col_names = []
    unfilt_x_names = []
    nir_idx = -1
    red_idx = -1
    for idx, name in enumerate(header):
        if name in nonx_columns:
            continue
        unfilt_x_names.append(name)
        band_index = float(name.replace(get_setting("band preface"),""))
        if band_index in get_setting("bad bands"):
            print("Skipping field {} because it is in bad bands list"\
                    .format(name))
            continue
        if band_index == get_setting("ndvi red band"):
            red_idx = idx
        if band_index == get_setting("ndvi nir band"):
            nir_idx = idx
        x_col_names.append(name)
    x_col = [header.index(b) for b in x_col_names] 

    # Find columns for red and nir relative to the columns in X 
    if nir_idx in x_col:
        nir = x_col.index(nir_idx)
    else:
        nir = -1
    if nir < 0:
        print("nir band not found")
    else:
        print("nir band found {} in csv, {} in X".format(nir_idx,nir))

    if red_idx in x_col:
        red = x_col.index(red_idx)
    else:
        red = -1
    if red < 0:
        print("red band not found")
    else:
        print("red band found {} in csv, {} in X".format(red_idx,red))

    ##Clean out any rows in the data frame where all X features are <=0 
    X_unfiltered = df[x_col_names].to_numpy().astype(np.float64)
    good_rows = ~np.all(X_unfiltered <= 0,axis=1)

    # Identify rows with ndvi outside limit
    acceptable_ndvi = good_rows.copy()
    if (nir < 0 or red < 0):
        print('Cannot compute NDVI, skipping')
    else:
        print('Computing NDVI and checking against allowed extrema')
        if get_setting("ndvi maximum") > -1:
            ndvimax = get_setting("ndvi maximum")
        else:
            ndvimax = np.inf
        if get_setting("ndvi minimum") > -1:
            ndvimin = get_setting("ndvi minimum")
        else:
            ndvimin = -np.inf
        tmp_ndvi = compute_ndvi(X_unfiltered[good_rows,:],red,nir)
        acceptable_ndvi[good_rows] = np.logical_and(
                                         tmp_ndvi>=ndvimin,
                                         tmp_ndvi<=ndvimax)
        good_rows = np.logical_and(good_rows,acceptable_ndvi)

    ## Do brightness normalization if requested, and check limits
    acceptable_brightness = good_rows.copy()
    if (get_setting("brightness minimum") >= 0) or \
            (get_setting("brightness maximum") >= 0):
        print("Computing brightness normalized spectra")
        X_bn, X_norm = brightness_normalize(X_unfiltered[good_rows,:])
        print("Checking against allowed brightness extrema")
        if get_setting("brightness maximum") > -1:
            brightnessmax = get_setting("brightness maximum")
        else:
            brightnessmax = np.inf
        if get_setting("brightness minimum") > -1:
            brightnessmin = get_setting("brightness minimum")
        else:
            brightnessmin = -np.inf
        good_bn = np.logical_and(
                      X_norm>=brightnessmin,
                      X_norm<=brightnessmax)
        acceptable_brightness[good_rows] = good_bn
        ##Update good_rows
        good_rows = np.logical_and(good_rows,acceptable_brightness)
    else:
        print("No brightness check limits, skipping")

    # If requested do a scaling of the good X data and save the scaler
    if get_setting("scale features"):
        x_scaler = preprocessing.StandardScaler()
        x_scaler.fit(df.loc[good_rows,x_col_names].to_numpy().astype(np.float64))
        if get_setting("brightness normalize"):
            scl_out = os.path.join(args.output_dir,'X_BN_scaler.pkl.z')
        else:
            scl_out = os.path.join(args.output_dir,'X_scaler.pkl.z')
        joblib.dump(X, scl_out, compress=True)
    else:
        x_scaler = None
    
    del X_unfiltered

    good_x = good_rows
    ################# step through each chem, run all PLSR code.  Each chem is independent ############
    # Store mean coefficients for all chems
    chem_coef_table = pd.DataFrame()
    for chem_index, chem_name in enumerate(chemlist):
        repstr = "Running chem {}".format(chem_name)
        poundstr = "#"*len(repstr)
        print(poundstr)
        print(poundstr)
        print(repstr)
        print(poundstr)
        print(poundstr)

        # Stores global test set and validation set membership for each
        # iteration
        all_bool = df.iloc[:,[cluster_col]]
        all_bool.insert(all_bool.shape[1],"NDVI Check",acceptable_ndvi * 1)
        all_bool.insert(all_bool.shape[1],"Brightness Check",acceptable_brightness * 1)
        all_bool.insert(all_bool.shape[1],"Good X Data",good_x * 1)
        
        # find the column of the dataframe corresponding to the chem of interest
        try:
            resp_col = header.index(chem_name)
        except ValueError as exc:
            raise RuntimeError(f"Response column '{chem_name}' not found in input CSV")
        
        trans_name = get_setting("chem transforms")[chem_index].lower()
        chem_file = "{}_{}".format(chem_name,trans_name)

        # Identify any rows that have invalid y data
        good_y = np.isfinite(df[chem_name])
        all_bool.insert(all_bool.shape[1],"Good {} {}".format(chem_name,trans_name),good_y * 1)

        chem_good = np.logical_and(good_x, good_y)

        # Collect as numpy arrays
        #########################
        ##X
        if get_setting("brightness normalize"):
            print("Normalizing brightness")
            X, _ = brightness_normalize(
                    df.loc[chem_good,x_col_names].to_numpy().astype(np.float64))
        else:
            X = df.loc[chem_good,x_col_names].to_numpy().astype(np.float64)
        if x_scaler is not None:
            print("Applying scaler to X data")
            X = x_scaler.transform(X)
        ##Y
        if trans_name != "none":
            print("Applying transform {} to {}".format(trans_name, chem_name))
            label_name = "{}-transformed {}".format(trans_name,chem_name)
        else:
            print("Applying no transform to {}".format(
                     chem_name))
            label_name = chem_name
        Y = apply_transform(df.loc[chem_good,chem_name].to_numpy().astype(np.float64),
                            get_setting("chem transforms")[chem_index])
        Y_raw = df.loc[chem_good,chem_name].to_numpy().astype(np.float64)
        ##Cluster IDs
        chem_clust = df.iloc[np.where(chem_good)[0],cluster_col].to_numpy()
   
        ##Histogram of transformed chem values
        fig = plt.figure(facecolor='white',figsize=(6,6))
        plt.hist(Y[np.isfinite(Y)],color='gray')
        plt.xlabel(label_name)
        fig.savefig(os.path.join(args.output_dir,'Hist_{}.png'.format(chem_file)),dpi=100,format='png',facecolor='white',bbox_inches='tight')
        plt.close()
     
        ## Figure out if we are running jackknife mode
        ##############################################
        if (get_setting("iterations") < 0) or \
                (get_setting("iteration holdout") < 0):
            jack_clusters = np.unique(df.iloc[np.where(chem_good)[0],cluster_col])
            num_jack_clust = len(jack_clusters)
            jackknife_mode = True
            ##How many clusters nominally held out each iteration
            jack_maxclust = np.abs(get_setting("iteration holdout"))
            num_iterations = num_jack_clust // jack_maxclust
            if (num_jack_clust % jack_maxclust > 0):
                num_iterations += 1
            ##Select the clusters held out at each iteration
            jack_dict = {}
            perm_clust = np.random.permutation(jack_clusters).tolist()
            idx = 0
            while perm_clust:
                if idx not in jack_dict:
                    jack_dict[idx] = []
                jack_dict[idx].append(perm_clust.pop())
                idx += 1
                if idx == num_iterations:
                    idx = 0
        else:
            jackknife_mode = False
            num_iterations = get_setting("iterations")

        ## Create a global test set if "test set holdout" is non-zero
        if get_setting("test set holdout") >= 0 and (not jackknife_mode):
            global_test, global_test_ids = \
                create_holdout_clusters(X,
                                        chem_clust,
                                        get_setting("test set holdout"),
                                        reserved=None)
            print("Created global test set with clusters {}".format(
                       global_test_ids))
        else:
            global_test = np.repeat(False,X.shape[0])
            global_test_ids = []

        # global_test refers to the subset chem_good = True, remap to add to
        # all_bool data frame
        all_bool.insert(all_bool.shape[1],"Global {}".format(chem_name),
                map_subset_boolean_array(chem_good, global_test) * 1)

        X_glob = X[global_test,:]
        Y_glob = Y_raw[global_test]
        clusters_glob = chem_clust[global_test]
        ##Y_obs averaged by cluster in global set
        Y_obs_glob_bycluster = np.array([
            np.mean(Y_glob[clusters_glob == clust]) \
              for clust in global_test_ids])

        ##################### start the iteration loop ########################
        # at each iteration, we will pull out a random sample of pixels from different clusters,
        # and then fit the PLSR to that data set.  As we go, we'll update the overall coefficients
        # and check performance of the local PLSR, along with the cumulative, developing model

        # Stores coefficients from all iterations
        all_coeff = np.ones((num_iterations,X.shape[1]+1)) * np.nan
        # Stores variable importance in projection scores from all iterations
        all_vip = np.ones((num_iterations,X.shape[1])) * np.nan
        # Stores perfomance as RMSE of each iteration
        coeff_perf = np.ones(num_iterations)*np.inf
        # Store regression stats across all iterations and test sets
        all_results_table = pd.DataFrame({
                               "Iteration":[],
                               "Set":[],
                               "Coef":[],
                               "Slope":[],
                               "Intercept":[],
                               "R-Squared":[],
                               "RMSE":[]}).astype({
                                   "Iteration":int,
                                   "Set":str,
                                   "Coef":str,
                                   "Slope":float,
                                   "Intercept":float,
                                   "R-Squared":float,
                                   "RMSE":float})

        
        ##Iterations
        jack_pred = np.ones_like(Y) * np.nan
        for iter in range(0,num_iterations):
            itername = iter+1
            if jackknife_mode:
                repstr = "Leave-{} out jackknife iteration {} of {}".format(jack_maxclust,itername,num_iterations)
            else:
                repstr = "Iteration {} of {}".format(itername,num_iterations)
            poundstr = "#"*len(repstr)
            print(repstr)
            print(poundstr)
            # Create the holdout set if requested 
            if jackknife_mode:
                iter_test, iter_test_ids = \
                        create_holdout_clusters(
                            X,
                            chem_clust,
                            1,
                            reserved=global_test,
                            selected_clusters=jack_dict[iter])
                print("Created iteration test set with clusters {}".format(
                               iter_test_ids))
            else:
                if get_setting("iteration holdout") >= 0:
                    iter_test, iter_test_ids = create_holdout_clusters(
                                                X,
                                                chem_clust,
                                                get_setting("iteration holdout"),
                                                reserved=global_test)
                    print("Created iteration test set with clusters {}".format(
                               iter_test_ids))
                else:
                    iter_test = np.repeat(False,X.shape[0])
                    iter_test_ids = []
            # global_test refers to the subset chem_good = True, remap to add to
            # all_bool data frame
            all_bool.insert(all_bool.shape[1],"Iter {} {} test".format(itername, chem_name),
                map_subset_boolean_array(chem_good, iter_test) * 1)

            iter_reserved = np.logical_or(global_test, iter_test)

            # Create a training set by selecting 1 to N pixels from each
            # cluster (not in test sets) randomly. N is the maximum of 
            # Ncluster and N specified by user
            ############################################################
            iter_train = gather_training_clusters(
                             X,
                             chem_clust,
                             get_setting("samples per cluster"),
                             reserved=iter_reserved)
            all_bool.insert(all_bool.shape[1],"Iter {} {} train".format(itername, chem_name),
                map_subset_boolean_array(chem_good, iter_train) * 1)
    
            ##Gather the data for training
            X_train = X[iter_train,:]
            Y_train = Y[iter_train]
            clusters_train = chem_clust[iter_train]
    
            ##Gather the data for validation - might be empty
            X_test = X[iter_test,:]
            Y_test = Y_raw[iter_test]
            clusters_test = chem_clust[iter_test]
            
            ## initialize and train model
            #############################
            if (get_setting("use degen removal") == True):
                iter_coeff, iter_vip = \
                        autoselect_plsr_degen_removal(X_train,Y_train,get_setting("max components"))
            else:
                iter_coeff, iter_vip = \
                        autoselect_plsr(X_train,Y_train,get_setting("max components"))
    
            ##Add the local coefficients to the combined coeff array
            all_coeff[iter,:] = iter_coeff
            all_vip[iter,:] = iter_vip


            # Predict from this local fit on the training set
            #################################################
            if np.any(iter_train):
                Y_pred_train_iter = \
                    apply_transform(plsr_predict(iter_coeff,X_train),trans_name,invert=True)
                Y_obs_train_iter = Y_train.copy()
            if iter_train.sum() > 3:
                regres = regression_results(Y_pred_train_iter, Y_obs_train_iter)
                print(f'iteration {itername} training set all samples:')
                print(regres)
                all_results_table = all_results_table.append(dict(zip(
                    all_results_table.columns,
                    (itername,"Iter train","Iter",*regres.iloc[0,:]))),
                    ignore_index = True)

            # Predict from this local fit on the validation set
            ###################################################
            if np.any(iter_test):
                Y_pred_iter = \
                    apply_transform(plsr_predict(iter_coeff,X_test),trans_name,invert=True)
                Y_obs_iter = Y_test.copy()

                if jackknife_mode:
                    jack_pred[iter_test] = Y_pred_iter
                     
            ##If enough rows, run a regression on the pred vs obs for test set
            if iter_test.sum() > 3:
                regres = regression_results(Y_pred_iter, Y_obs_iter)
                print(f'iteration {itername} holdout all samples:')
                print(regres)
                all_results_table = all_results_table.append(dict(zip(
                    all_results_table.columns,
                    (itername,"Iter holdout","Iter",*regres.iloc[0,:]))),
                    ignore_index = True)
         
                ##### find cluster averages
                Y_pred_iter_bycluster = np.array([
                    np.mean(Y_pred_iter[clusters_test == clust]) \
                      for clust in iter_test_ids])
                Y_obs_bycluster = np.array([
                    np.mean(Y_obs_iter[clusters_test == clust]) \
                      for clust in iter_test_ids])
                regres = regression_results(Y_pred_iter_bycluster,Y_obs_bycluster)
                print(f'iteration {itername} holdout all samples by cluster:')
                print(regres)
                all_results_table = all_results_table.append(dict(zip(
                    all_results_table.columns,
                    (itername,"Iter holdout by cluster","Iter",*regres.iloc[0,:]))),
                    ignore_index = True)
    
                ##Compute performance at row-level
                coeff_perf[iter] = plsr_rmse(Y_pred_iter, Y_obs_iter)

                ##Get average of all sets of coefficients that perform better
                ## than median RSE on the test set
                med_perf = np.median(coeff_perf[coeff_perf>0])
                ## Phil had this as > median, but for RMSE this would be bad??
                val_iter = np.isfinite(coeff_perf)
                mean_vip = np.nanmean(all_vip[val_iter,:],axis=0)
                
                low_rmse = np.logical_and(val_iter, coeff_perf <= med_perf)
                mean_coeff = np.nanmean(all_coeff[low_rmse,:],axis=0)
            else:
                # We don't evaluate performance for each iteration, so just take
                # overall mean of already-run iterations
                coeff_perf[iter] = 1
                mean_coeff = np.nanmean(all_coeff,axis=0)
                mean_vip = np.nanmean(all_vip,axis=0)

            ##If a global holdout set exists, then apply to that
            ####################################################
            if np.any(global_test):
                Y_obs_glob = Y_glob.copy()
         
                ##Apply the iteration coeffs to this global set
                ###############################################
                Y_pred_glob_iter = \
                    apply_transform(plsr_predict(iter_coeff,X_glob),trans_name,invert=True)
                regres = regression_results(Y_pred_glob_iter,Y_glob)
                print(f'global holdout all samples (iteration {itername} coefficients):')
                print(regres)
                all_results_table = all_results_table.append(dict(zip(
                    all_results_table.columns,
                    (itername,"Global holdout","Iter",*regres.iloc[0,:]))),
                    ignore_index = True)
                ##### find cluster averages
                Y_pred_glob_iter_bycluster = np.array([
                    np.mean(Y_pred_glob_iter[clusters_glob == clust]) \
                      for clust in global_test_ids])
                ## print stats
                regres = regression_results(Y_pred_glob_iter_bycluster,Y_obs_glob_bycluster)
                print(f'global holdout per cluster (iteration {itername} coefficients):')
                print(regres)
                all_results_table = all_results_table.append(dict(zip(
                    all_results_table.columns,
                    (itername,"Global holdout by cluster","Iter",*regres.iloc[0,:]))),
                    ignore_index = True)

                ##Apply the current mean coeffs to this global set
                ##################################################
                Y_pred_glob_mean = plsr_predict(mean_coeff,X_glob)
                regres = regression_results(Y_pred_glob_mean,Y_glob)
                print('global holdout all samples (mean coefficients):')
                print(regres)
                all_results_table = all_results_table.append(dict(zip(
                    all_results_table.columns,
                    (itername,"Global holdout","Mean",*regres.iloc[0,:]))),
                    ignore_index = True)
                ##### find cluster averages
                Y_pred_glob_mean_bycluster = np.array([
                    np.mean(Y_pred_glob_mean[clusters_glob == clust]) \
                      for clust in global_test_ids])
                ## print stats
                regres = regression_results(Y_pred_glob_mean_bycluster,Y_obs_glob_bycluster)
                print('global holdout per cluster (mean coefficients):')
                print(regres)
                all_results_table = all_results_table.append(dict(zip(
                    all_results_table.columns,
                    (itername,"Global holdout by cluster","Mean",*regres.iloc[0,:]))),
                    ignore_index = True)


        #################
        # After iteration
        #################
        if jackknife_mode:
            regres = regression_results(jack_pred,Y_raw)
            print('jackknife (iter coefficients):')
            print(regres)
            all_results_table = all_results_table.append(dict(zip(
                all_results_table.columns,
                (itername,"Jackknife","Iter",*regres.iloc[0,:]))),
                ignore_index = True)

            fig = plt.figure(facecolor='white',figsize=(6,6))
            plt.scatter(Y_raw,jack_pred,color='red',s=5)
            np.savetxt(os.path.join(args.output_dir,'Scatter_jackknife_iter_{}_data.csv'.format(chem_file)),np.vstack([jack_pred,Y_raw]),delimiter=',')
            ub = np.max(np.append(jack_pred,Y_raw))
            lb = np.min(np.append(jack_pred,Y_raw))
            plt.plot([lb,ub],[lb,ub],color='black',ls='--')
            plt.xlim([lb,ub])
            plt.ylim([lb,ub])
            plt.xlabel('Observed {}'.format(chem_name))
            plt.ylabel('Jackknife coeff predicted {}'.format(chem_name))
            fig.savefig(os.path.join(args.output_dir,'Scatter_jackknife_iter_{}.png'.format(chem_file)),dpi=100,format='png',facecolor='white',bbox_inches='tight')
            plt.close()

            jack_pred_mean = plsr_predict(mean_coeff, X)
            regres = regression_results(jack_pred_mean,Y_raw)
            print('jackknife (mean coefficients):')
            print(regres)
            all_results_table = all_results_table.append(dict(zip(
                all_results_table.columns,
                (itername,"Jackknife","Mean",*regres.iloc[0,:]))),
                ignore_index = True)

            fig = plt.figure(facecolor='white',figsize=(6,6))
            plt.scatter(Y_raw,jack_pred_mean,color='red',s=5)
            np.savetxt(os.path.join(args.output_dir,'Scatter_jackknife_mean_{}_data.csv'.format(chem_file)),np.vstack([jack_pred_mean,Y_raw]),delimiter=',')
            ub = np.max(np.append(jack_pred_mean,Y_raw))
            lb = np.min(np.append(jack_pred_mean,Y_raw))
            plt.plot([lb,ub],[lb,ub],color='black',ls='--')
            plt.xlim([lb,ub])
            plt.ylim([lb,ub])
            plt.xlabel('Observed {}'.format(chem_name))
            plt.ylabel('Jackknife mean coeff predicted {}'.format(chem_name))
            fig.savefig(os.path.join(args.output_dir,'Scatter_jackknife_mean_{}.png'.format(chem_file)),dpi=100,format='png',facecolor='white',bbox_inches='tight')
            plt.close()

        ##Write out the full results table to CSV
        all_results_table.to_csv(os.path.join(args.output_dir,"All_results_{}.csv".format(chem_file)))
        ##Write out the membership data frame
        all_bool.to_csv(os.path.join(args.output_dir,"All_memberships_{}.csv".format(chem_file)))
        ##Write out all coefs to CSV
        np.savetxt(os.path.join(args.output_dir,"All_coefs_{}.csv".format(chem_file)),all_coeff,delimiter=',')


        #  # If a global holdout exists, make figures and save
        #  ###############################################################
        if np.any(global_test):
            ##Scatterplot of global test with mean coefs by cluster
            fig = plt.figure(facecolor='white',figsize=(6,6))
            plt.scatter(Y_obs_glob_bycluster,Y_pred_glob_mean_bycluster,color='red',s=5)
            np.savetxt(os.path.join(args.output_dir,'Scatter_global_mean_{}_data.csv'.format(chem_file)),np.vstack([Y_obs_glob_bycluster,Y_pred_glob_mean_bycluster]),delimiter=',')
            ub = np.max(np.append(Y_obs_glob,Y_pred_glob_mean))
            lb = np.min(np.append(Y_obs_glob,Y_pred_glob_mean))
            plt.plot([lb,ub],[lb,ub],color='black',ls='--')
            plt.xlim([lb,ub])
            plt.ylim([lb,ub])
            plt.xlabel('Global holdout cluster {}'.format(chem_name))
            plt.ylabel('Mean coeff predicted {}'.format(chem_name))
            fig.savefig(os.path.join(args.output_dir,'Scatter_global_mean_{}.png'.format(chem_file)),dpi=100,format='png',facecolor='white',bbox_inches='tight')
            plt.close()

        ##Weightings figure
        fig = plt.figure(facecolor='white',figsize=(10,6))
        wavelengths = [float(b.replace(get_setting("band preface"),"")) for b in x_col_names]
        for iter in np.where(coeff_perf >=0)[0]:
            plt.scatter(wavelengths,all_coeff[iter,1:],facecolor='blue',alpha=0.3,s=2)
        plt.scatter(wavelengths,mean_coeff[1:],color='black',s=4)
        if wavelengths[0] > 300:
            plt.xlabel('Wavelength [nm]')
        else:
            plt.xlabel('Band index')
        plt.ylabel('PLSR Coeff')
        fig.savefig(os.path.join(args.output_dir,'Coeff_ranges_{}.png'.format(chem_file)),dpi=100,format='png',facecolor='white',bbox_inches='tight')
        plt.close()

        ##VIP figure
        fig = plt.figure(facecolor='white',figsize=(10,6))
        for iter in np.where(coeff_perf >=0)[0]:
            plt.scatter(wavelengths,all_vip[iter,:],facecolor='blue',alpha=0.3,s=2)
        plt.scatter(wavelengths,mean_vip,color='black',s=4)
        if wavelengths[0] > 300:
            plt.xlabel('Wavelength [nm]')
        else:
            plt.xlabel('Band index')
        plt.ylabel('VIP Score')
        fig.savefig(os.path.join(args.output_dir,'VIP_{}.png'.format(chem_file)),dpi=100,format='png',facecolor='white',bbox_inches='tight')
        plt.close()
        np.savetxt(os.path.join(args.output_dir,'VIP_{}_data.csv'.format(chem_file)),np.vstack([all_vip,mean_vip]),delimiter=',')
    
        ##Add coefs to chem_coef_table
        ##############################
        # First pad coefs with 0s for the bad bands
        full_coeff = [0.0]*len(unfilt_x_names)
        for col, val in zip(x_col_names,mean_coeff[1:]):
            ind = unfilt_x_names.index(col)
            full_coeff[ind] = val
        chem_coef_table = \
            chem_coef_table.append(
                    OrderedDict(zip(
                             ["Chem","Transform","Intercept",*unfilt_x_names],
                             [chem_name,trans_name,mean_coeff[0],*full_coeff])),
                    ignore_index=True)


    coeff_file = os.path.join(args.output_dir,'coeff_{}.csv'.format(chem_file))
    chem_coef_table.to_csv(coeff_file,index=False)
    
    
    
if __name__ == "__main__":
    main()
