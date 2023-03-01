# NitrogenRetrieval - 
Code and coefficients for retrieving Nitrogen content from Tanager reflectance data over agricultural vegetation landscapes.

There are two subfolders: `code` and `coefficients`. The `code` folder contains two python scripts, one to fit and rebuild PLSR coefficients (`fit_plsr_coefficients.py`) and one to apply coefficients to a reflectance map (`apply_plsr_coefficients.py`).  The `coefficients` folder contains a CSV database of the coefficients from the fit of PLSR to GAO reflectance data described in this report.

## Usage
### apply_plsr_coefficients.py

To use the apply script is fairly straightforward:
```
apply_plsr_coefficients.py [-h] [-bright_min BRIGHT_MIN] [-bright_max BRIGHT_MAX]
  [-ndvi_min NDVI_MIN] [-nodata NODATA] [-format FORMAT] [-co CO] [-rescale RESCALE]
  refl_dat_f output_name chem_eq_f
```
#### The three required positional arguments are:
Argument   | Description
---------- | ------------------------------
`refl_dat_f` | ENVI-formatted reflectance map
`output_name` | Output file name, should match given `-format`
`chem_eq_f` | CSV file containing fields for chem name, transform, intercept, and 428 coefficients


#### Optional arguments are: 
Argument   | Description
---------- | ------------------------------
`-h, --help` | ENVI-formatted reflectance map
`-bright_min, -bright_max` | Thresholds on brightness (vector norm of bands with non-zero PLSR coefficients) to remove input spectra from consideration. Default `1.5` and `9.9`, specify `-1` to remove.
`-rescale` | If input reflectance is not scaled 0-1, then default brightness thresholds will not work. This value will be multiplied by input reflectance values.
`-ndvi_min` | Threshold of NDVI to remove spectra from consideration. Default is `0.7`, specify `-1` to remove.
`-nodata` | Output map will have this value representing “no data”. Defaults to `-9999`.
`-format` | GDAL-recognizable shortname for output data format. Defaults to `“GTiff”`
`-co` | GDAL-recognized creation options for the specified data format.

Note that a wavelength/fwhm database (`gao_2022_wl_fwhm.csv`) is included in the code folder and must be located in the same folder as the apply_plsr_coefficients.py script in order for the script to work.

### fit_plsr_coefficients.py
To use the build script is much less straight-forward. This script implements a resampling-based approach to fitting PLSR with minimal tuning. There are multiple ways of dividing the supplied data into training, testing and validation sets. All settings are specified with a JSON-formatted config file. An optional validation set (called global test set in the script) can be specified (as a proportion using parameter test set holdout) that is never included in the training of the data. Samples not in the validation set are iteratively divided into a training and testing set for each of a specified number of iterations. Division at each iteration can be in one of two modes:

Mode | Description
---- | -----------
Bootstrap mode | The number of iterations is user-specified (using iterations parameter) and for each iteration the samples are randomly selected into the two sets with replacement with a proportion specified in iteration holdout being placed into the testing set. If data are clustered, a minimal number of samples per cluster in the training set can be specified with config parameter samples per cluster.
Jackknife mode | The number of iterations is computed based on available sample size and number of samples (or clusters of samples) per iteration defined as a negative integer in config parameter iteration holdout. All samples (or clusters) are divided into n groups, and each group acts as the test set for one iteration. Thus, each group is tested once and only once against a model built from the other groups.


The user specified a maximum number of components (as max components - akin to principal components analysis) to consider in the PLSRegression model fit in each iteration. Within an iteration, and for each number of components from 1 to max components, the PLSR model is fit using a 10-fold cross validation approach, allowing the computation of the PRESS statistic. The optimal number of components is selected using the number associated with the maximum value of the PRESS statistic. The model is fit again to the full training set using the optimal number of components, and the coefficients from this model are retained and applied to the iteration test set to get an RMSE. After all iterations, a across-iteration median RMSE is computed, and the coefficients for all iterations that had RMSE > RMSEmed are averaged by band to get a global coefficient set. These coefficients are applied to the global test / validation set if there is one to get a validation RMSE and R2.


The command line call looks like this:
```
fit_plsr_coefficients.py [-h] settings_file output_dir
```

Which will apply the settings in a JSON-formatted settings_file and write all results to the specified output directory (output_dir).  The settings file has the following entries:
Parameter | Description
--------- | -----------
`csv file` | CSV file with all data - should have a row for each input spectra. Spectra can be grouped into clusters if there is a column identifying cluster ID. The spectral data should be column-wise with a column for each band in sequential order. Columns names can have a preface, e.g. `B001, B002, ...`, which can be removed to get a band index as an integer.
`band preface` | What precedes numbers in the band columns - usually `"B"`
`chems` |  List of chems to fit, e.g. `["LMA"]`
`chem transforms` | Matching list of transforms `"log"`, `"sqrt"`, `"square"`, `"inverse"` or `"none"` - like `["none"]`
`cluster col` | Field that contains cluster id (assuming multiple pixels in each cluster). If not cluster-based, use `"none"`
`bad bands` | List of (1-based) band numbers removed before normalization e.g. `[1,2,3,4,211,212,213,214]`
`ignore columns` | List of quoted column names that should be ignored in  the analysis, e.g. `["ID","SomeStat"]`
`brightness normalize` | `true`/`false` - do brightness normalization (Suggested)
`ndvi minimum` | Minimum NDVI value, `-1` for no limit
`ndvi maximum` | Minimum NDVI value, `-1` for no limit
`ndvi red band` | Band representing red for NDVI computation, e.g. `34`
`ndvi nir band` | Band representing red for NDVI computation, e.g. `46`
`brightness maximum` | Maximum NDVI value, `-1` for no limit
`brightness minimum` | Minimum NDVI value, `-1` for no limit
`iterations` | Number of iterations of the algorithm - for each iteration, the fraction of clusters specified in "iteration holdout" will be used to make a test set and the model will be fit on the rest of the data (except that specified in `test set holdout`). Use `-1` to use jackknife mode, i.e. if negative value in `iteration holdout`
`iteration holdout` | Fraction of data used for validation at each iteration - can be 0 for no validation set, use negative values (i.e. `-n`) to use jackknife mode, where n clusters are held out each time, and each iteration is mutually exclusive.
`test set holdout` | Fraction of data used for global holdout test set - can be 0 for no global holdout set, ignored in jackknife mode.
`samples per cluster` | Specify a minimum number of samples per cluster for training data using bootstrap mode, use `-1` for no limit
`max components` | Maximum number of components checked with PRESS stat
`use degen removal` | `true`/`false` - use procedure to find and remove degenerate (less significant) input features. Similar to the A4 function in autopls. (*Untested and unused for this analysis)
`scale features` | `true`/`false` -  fit scaler to input features (Not suggested, as the scaler would need to be saved to be used for applying fitted coefficients at a later point)
`n jobs` | Number of parallel jobs run by `sklearn` functions
`random seed` | Seed value for random number generator for reproducible results (-1 for random seed)

Example configuration files for the bootstrap and jackknife modes used in the fitting of coefficients in the ATBD report can be found in the `config` folder of this repository.
