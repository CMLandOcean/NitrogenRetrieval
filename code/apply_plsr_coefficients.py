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
# Original author:
#   Phillip Brodrick, phil.brodrick AT gmail.com
#
# Substantially modified by:
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

import numpy as np
import os
import time
import pandas as pd
import argparse
import spectral
from osgeo import gdal
from tqdm import tqdm

###########################################
##Read in baseline wavelength and fwhm info
###########################################
WL_ORIG = FWHM_ORIG = None
dbpath = os.path.join(os.path.dirname(__file__),"gao_2022_wl_fwhm.csv")
try:
    wldat = pd.read_csv(dbpath)
    WL_ORIG = wldat['wl'].to_numpy()
    FWHM_ORIG = wldat['fwhm'].to_numpy()
except Exception as exc:
    raise RuntimeError("Could not read wl and fwhm from orig_wlfwhm.csv")

###################################
##Find reflectance hdr if available
###################################
def find_envi_hdr(map):
    poss = [map+".hdr", os.path.splitext(map)[0]+".hdr"]
    for p in poss:
        if os.path.exists(p):
            return p
    return None

##################################################################
##Get wavelength info for NDVI calculation and possible resampling
##################################################################
def get_wl_fwhm(hdr):
    wl = None
    fwhm = None
    if "wavelength" in hdr:
        wl = [float(v) for v in hdr["wavelength"]]
    if "fwhm" in hdr:
        fwhm = [float(v) for v in hdr["fwhm"]]
    return wl, fwhm

##############################################################
##Get band numbers for red and nir to use in ndvi calculations
##############################################################
def get_ndvi_bands(wl):
    redwl = 670
    nirwl = 790
    wla = np.array(wl).ravel()
    #Get red
    redband = np.argmin(np.abs(redwl - wla))
    #Get nir
    nirband = np.argmin(np.abs(nirwl - wla))
    return redband, nirband

def main():
    parser = argparse.ArgumentParser(
            description='Apply chem equation to IS reflectance')
    parser.add_argument('-bright_min','--bright_min',default=1.50,type=float,
            help="Minimum brightness (refl vector norm) considered, default 1.5")
    parser.add_argument('-bright_max','--bright_max',default=9.90,type=float,
            help="Maximum brightness considered, default 9.9")
    parser.add_argument('-ndvi_min','--ndvi_min',default=0.7,type=float,
            help="Minimum ndvi considered, default 0.7")
    parser.add_argument('-nodata','--nodata',default=-9999,type=float,
            help="Value to represent nodata in output")
    parser.add_argument('-format','--format',default="GTiff",
            help="GDAL format to use for output raster, default GTiff")
    parser.add_argument('-co','--co',action='append',
            help="GDAL creation options for the given format")
    parser.add_argument('-rescale','--rescale',default=1.0,type=float,
            help="Apply this multiplier to refl data first, input expected to "
            "be scaled to 0 to 1.0 - affects bright_min and bright_max")
    parser.add_argument('refl_dat_f',help="ENVI-formatted reflectance map")
    parser.add_argument('output_name',
            help="Output file name, should match given format")
    parser.add_argument('chem_eq_f',
            help="CSV file containing fields for chem name, transform, "
            "intercept, and 214 coefficients")
    args = parser.parse_args()
    np.seterr(divide = 'ignore', invalid = 'ignore')

    #############################
    # Read chemical equation data
    #############################
    coef_tab = pd.read_csv(args.chem_eq_f)
    
    ##Get intercept and slope matrices
    ##################################
    transform_dict = {}
    intercept_dict = {}
    coef_dict = {}
    selected_chems = []
    for ind, row in coef_tab.iterrows():
        chem=row["Chem"]
        selected_chems.append(chem)
        transform_dict[chem] = row["Transform"]
        intercept_dict[chem] = row["Intercept"]
        try:
            coef_dict[chem] = np.array(row.to_list()[3:]).astype(float)
        except ValueError as exc:
            raise RuntimeError("Could not convert coefficient to float, CSV "
            "is not correctly formatted: {}".format(exc))
    num_coef = len(coef_dict[chem])
    print("Found {} chems in {}:\n{}".format(len(selected_chems),
                                             args.chem_eq_f,
                                             selected_chems))
    print("Found {} slope coefficients".format(num_coef))
    if len(selected_chems) < 1:
        raise RuntimeError("No chems found")

    ##Figure out which chems we are running
    #######################################
    print(f"Will produce the following chems: {selected_chems}")
    
    ##Keep track of which bands have coef = 0 for brightness normalization
    ######################################################################
    good_b = [c != 0 for c in coef_dict[selected_chems[0]]]

    ##Try to get reflectance header
    ###############################
    hdrname = find_envi_hdr(args.refl_dat_f)
    if hdrname is None:
        raise RuntimeError("Could not find header file for "+args.refl_dat_f)
    fromhdr = spectral.envi.read_envi_header(hdrname)

    ##Try to get wl, fwhm from hdr file
    ###################################
    inwl, infwhm = get_wl_fwhm(fromhdr)
    if inwl is None:
        raise RuntimeError("Could not get wavelengths from "+hdrname)

    resampler = None
    if (abs(inwl[0]-WL_ORIG[0]) > 0.5) or \
       (len(inwl) != len(WL_ORIG)):
        print("Building band resampler ({}) -> ({})".format(len(inwl),len(WL_ORIG)))
        if infwhm is not None:
            resampler = spectral.BandResampler(inwl, WL_ORIG, infwhm, FWHM_ORIG)
        else:
            resampler = spectral.BandResampler(inwl, WL_ORIG)

        ##Find bands that resampler can't make
        rsones = resampler(np.ones(len(inwl)))
        missing_bands = np.isnan(rsones)

        ##If we have missing bands, make sure coefficients for these bands are 0
        if missing_bands.sum() > 0:
            for chem in selected_chems:
                tmpcoef = np.array(coef_dict[chem])
                if np.any(tmpcoef[missing_bands] != 0):
                    w_B = np.where(tmpcoef[missing_bands] != 0)[0]
                    raise RuntimeError("Missing bands from interpolation have "
                    "non-zero coefficients, indices: {}".format(w_B.tolist()))
            print("Missing bands found, but ignoring because these bands "
                  "have all-zero coefficients")

        inwl = WL_ORIG 
        infwhm = FWHM_ORIG 

    # open up raster sets
    dataset = gdal.Open(args.refl_dat_f,gdal.GA_ReadOnly)
    innodata = dataset.GetRasterBand(1).GetNoDataValue()
    data_trans = dataset.GetGeoTransform()

    data_rows = dataset.RasterYSize
    data_cols = dataset.RasterXSize


    ##########################
    # Create blank output file
    ##########################
    driver = gdal.GetDriverByName(args.format) 
    driver.Register() 

    if args.co is None:
        options = []
    else:
        options = [opt.strip() for opt in args.co]

    outDataset = driver.Create(args.output_name,
                               data_cols,
                               data_rows,
                               len(selected_chems),
                               gdal.GDT_Float32,
                               options=options)

    outDataset.SetProjection(dataset.GetProjection())
    outDataset.SetGeoTransform(dataset.GetGeoTransform())

    ########################
    # Loop through lines [y]
    ########################
    for l in tqdm(range(0,data_rows),ncols=80):

        ##Read data
        ###########
        dat = np.squeeze(dataset.ReadAsArray(0,l,data_cols,1)).astype(np.float32) 

        ##Make an empty output dataset
        ##############################
        outdat = np.ones((1,data_cols)) * args.nodata

        #######################
        ##Filter out bad pixels
        #######################
        dat[dat == -9999] = np.nan
        dat[dat == innodata] = np.nan
        dat[:,np.all(dat == 0,axis=0)] = np.nan

        ##Rescale
        ##Doesn't matter for fit since brightness correction used
        ## however, bright_max and bright_min are scaled to refl 0-1
        ############################################################
        dat = dat * args.rescale

        ##Resample if requested
        #######################
        if resampler is not None:
            tdat = np.hstack([
                resampler(dat[:,c]).reshape(-1,1) for c in range(data_cols)])
            dat = tdat

        ##Compute NDVI
        ##############
        bred, bnir = get_ndvi_bands(inwl)
        ndvi = (dat[bnir,:] - dat[bred,:]) / (dat[bnir,:] + dat[bred,:])

        ##Compute Brightness (as norm) of good bands
        norm = np.linalg.norm(dat[good_b,:],axis=0)

        ##Brightness normalize
        dat = dat / norm

        ##Apply NDVI mask
        #################
        if (args.ndvi_min != -1):
            dat[:, ndvi < args.ndvi_min] = np.nan
      
        ##Apply brightness mask
        #######################
        if (args.bright_min != -1):
            dat[:,norm < args.bright_min] = args.nodata
        if (args.bright_max != -1):
            dat[:,norm > args.bright_max] = args.nodata
      
        ##Work on valid columns, if any
        ###############################
        val = np.all(np.isfinite(dat[good_b,:]),axis=0)
        num_val = val.sum()
        if num_val > 0:
            dat = dat[:,val]
            norm = norm[val][np.newaxis,:]
        else:
            emptydat = np.ones((1,data_cols),dtype=np.float32) * args.nodata
            for band,chem in enumerate(selected_chems):
                outDataset.GetRasterBand(band+1).WriteArray(emptydat,0,l)
            continue

        #####################
        ##Loop through chems 
        #####################
        for band, chem in enumerate(selected_chems):
            slope_mat = coef_dict[chem].reshape(-1,1)
            transform = transform_dict[chem]
          
            # calculate chem
            ################
            output = intercept_dict[chem] + \
                        np.sum(slope_mat[good_b,:]*dat[good_b,:],axis=0)
          
            # reverse transform
            ###################
            if (transform == 'sqrt'):
                output= np.power(output,2)
            if (transform == 'log'):
                output = np.exp(output)
            if (transform == 'square'):
                output = np.sqrt(output)
          
          
            # write output
            ##############
            outdat[:,val] = output
            outDataset.GetRasterBand(band+1).WriteArray(
                                               outdat.reshape((1,data_cols)),0,l)

    #####################################
    ##Set band nodata value and chem name
    #####################################
    for band,chem in enumerate(selected_chems):
        outDataset.GetRasterBand(band+1).SetNoDataValue(args.nodata)
        outDataset.GetRasterBand(band+1).SetDescription(chem)
    del outDataset

    print("Done")

if __name__ == "__main__":
    main()

