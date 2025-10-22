# This script contains code necessary for comparing Histogram and k-NN KL estimators in Supplementary Figure 2-4

from scipy.spatial import cKDTree as KDTree
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ----- Function for calculating true KL and MIND ----- #

def calculate_true_mind(mu_A, mu_B, sigma_A, sigma_B):
    """
    Calculates KL and MIND network using a Histogram binning method
    
    Parameters:
    - mu_A: float, mean of distibution A
    - mu_B: float, mean of distibution B
    - sigma_A: float, standard deviation of distibution A
    - sigma_B: float, standard deviation of distibution B

    Returns:
    - minds: numpy array, MIND matrix
    - jef: numpy array, symmetrised KL (Jeffrey's divergence) matrix
    - dklpq: numpy array, KL (from p to q) matrix
    - dklqp: numpy array, KL (from q to p) matrix
    
    """
    true_kla = np.log(sigma_B / sigma_A) + (sigma_A**2 + (mu_A - mu_B)**2) / (2 * sigma_B**2) - 0.5
    true_klb = np.log(sigma_A / sigma_B) + (sigma_B**2 + (mu_B - mu_A)**2) / (2 * sigma_A**2) - 0.5
    true_jef = true_kla + true_klb
    true_mind = 1/(1+true_jef)

    return true_mind, true_jef, true_kla, true_klb


# ----- Functions for estimating KL and MIND using a k-NN method ----- # https://github.com/isebenius/MIND

def get_KDTree(x): #Inspired by https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    
    # Build a KD tree representation of the samples
    xtree = KDTree(x)
    
    return xtree

def get_KL(x, y, xtree, ytree, k=0): #Inspired by https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape
    
    #Check dimensions
    assert(d == dy)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2+k, p=2)[0][:,1+k]

    if k==0: # tree query results are 1-D if K=1, 2D if K>1, so need to change indexing
        s = ytree.query(x, k=1+k, p=2)[0]
    else:
        s = ytree.query(x, k=1+k, p=2)[0][:,k]
    
    rs_ratio = r/s

    #Remove points with zero, nan, or infinity. This happens when two regions have a vertex with the exact same value – an occurence that basically onnly happens for the single feature MSNs
    #and has to do with FreeSurfer occasionally outputting the exact same value for different vertices.
    rs_ratio = rs_ratio[np.isfinite(rs_ratio)]
    rs_ratio = rs_ratio[rs_ratio!=0.0]
    
    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.

    kl = -np.log(rs_ratio).sum() * d / n + np.log(m / (n - 1.))
    kl = np.maximum(kl, 0)
    
    return kl

def calculate_knn_mind(data_df, feature_cols, region_list, k=0, resample=False, n_samples = 4000, pctiles=None):
    # Initialise
    dklab = np.zeros((len(region_list), len(region_list)))
    dklba = np.zeros((len(region_list), len(region_list)))
    
    # Threshold based on pctiles
    if pctiles is not None:
        data_df = data_df[(data_df['Value'] > np.percentile(data_df['Value'], pctiles[0])) & (data_df['Value'] < np.percentile(data_df['Value'], pctiles[1]))]

    #Get only desired regions
    data_df = data_df.loc[data_df['Label'].isin(region_list)]
    
    #Resample dataset if resample has been set to True and if it is UNIVARIATE ONLY. This should only be done if you are using a single feature which contains repeated values.
    if (len(feature_cols) == 1) and resample==True: 
        n_samples = n_samples
        resampled_dataset = pd.DataFrame(np.zeros((n_samples, len(region_list))), columns = region_list)

        for name, data in data_df.groupby('Label'):
            resampled_dataset[name] = stats.gaussian_kde(data[feature_cols[0]]).resample(n_samples)[0]

        resampled_dataset = resampled_dataset.melt(var_name = 'Label', value_name = feature_cols[0])
        data_df = resampled_dataset

    grouped_data = data_df.groupby('Label', sort=False)

    KDtrees = defaultdict(object)

    for i, (name_x, dat_x) in enumerate(grouped_data):
        tree = get_KDTree(dat_x[feature_cols])
        KDtrees[name_x] = tree
    
    used_pairs = []
    
    for i, (name_x, dat_x) in enumerate(grouped_data):
        
        for j, (name_y, dat_y) in enumerate(grouped_data):
            if name_x == name_y:
                continue

            if set([name_x,name_y]) in used_pairs:
                continue

            dat_x = dat_x[feature_cols]
            dat_y = dat_y[feature_cols]

            KLa = get_KL(dat_x, dat_y, KDtrees[name_x], KDtrees[name_y], k=k)
            KLb = get_KL(dat_y, dat_x, KDtrees[name_y], KDtrees[name_x], k=k)

            dklab[i,j] = KLa
            dklba[i,j] = KLb

            used_pairs.append(set([name_x,name_y]))

    dklab += dklab.T
    dklba += dklba.T

    jef = dklab + dklba
    
    minds = 1 / (1 + jef)
    
    return minds, jef, dklab, dklba


# ----- Function for estimating KL and MIND using a Histogram method ----- #

def calculate_hist_mind(data, features, labels, nbins=128, epsilon=np.finfo(float).eps, pctiles=[0.1, 99.9]):
    """
    Calculates KL and MIND network using a Histogram binning method
    
    Parameters:
    - data: pandas df, voxel dataframe
    - features: string list, name of column in data containing distribution values
    - labels: numpy array, labels
    - nbins: int, number of histogram bins to use (default = 128)
    - epsilon: float, number to add to bin in q if empty to prevent division by 0 errors (default = np.finfo(float).eps)
    - pctiles: float list: min and max percentile of all data to keep (default = [0.1, 99.9])

    Returns:
    - minds: numpy array, MIND matrix
    - jef: numpy array, symmetrised KL (Jeffrey's divergence) matrix
    - dklab: numpy array, KL (from a to b) matrix
    - dklba: numpy array, KL (from b to a) matrix
    
    """

    if data is None:
        raise ValueError("No data given to mindsmv")
    if features is None:
        raise ValueError("No features given to mindsmv")
    if labels is None:
        raise ValueError("No labels given to mindsmv")

    # Get values
    in_data = data[features].to_numpy()[:, np.newaxis]
    labels = data[labels].to_numpy()

    # Number of dimensions for the input data
    dims = in_data.shape[-1]

    # Initialise edges 
    edges = np.zeros((nbins + 1, dims))
    included = labels > 0

    # Preprocess input data
    for i in range(dims):
        # Get data for this dimension
        pin = in_data[..., i]

        # Scale to zero mean and unit variance
        pin_standardized = (pin - np.nanmean(pin)) / np.nanstd(pin)

        # Apply percentile thresholds
        if pctiles != [0, 0]:
            pc = np.nanpercentile(pin_standardized, pctiles)
            pin_standardized[pin_standardized < pc[0]] = np.nan
            pin_standardized[pin_standardized > pc[1]] = np.nan

        # Update the standardized input data
        in_data[..., i] = np.round(pin_standardized, 4)

        # Mask to filter out nans so numpy can calculate histogram bins
        my_included = np.logical_and(included, ~np.isnan(pin_standardized))

        # Compute histogram edges
        _, edges[:, i] = np.histogram(pin_standardized[my_included], bins=nbins)

    # Get unique labels
    unique_labels = np.unique(labels[labels > 0])

    # Initialize ndhist (object holding counts for all bins for all regions for all dimensions)
    ndhist_shape = [len(unique_labels)] + [nbins] * dims
    ndhist = np.zeros(ndhist_shape)

    # Precompute histogram offsets (allows indexing multidimensional histogram bins as flattened vector) 
    hist_offset = np.array([nbins ** k for k in range(dims)])

    # Generate histograms for each region
    for idx, label in enumerate(unique_labels):
        # Find the region of interest mask
        roimask = np.where(labels == label)

        # Initialize bins and histogram
        my_bins = np.zeros((len(roimask[0]), dims), dtype=int)
        if dims > 1:
            my_hist = np.zeros([nbins] * dims)
        else:
            my_hist = np.zeros([nbins])

        # Assign values to bins, ensuring that NaNs are not assigned to a bin
        for j in range(dims):
            vals = in_data[roimask + (j,)]
            valid_vals_mask = ~np.isnan(vals)  # Mask for valid (non-NaN) values
            bin_indices = np.full(vals.shape, -1, dtype=int)  # Initialize all as invalid (-1)
            bin_indices[valid_vals_mask] = np.digitize(vals[valid_vals_mask], edges[:, j]) - 1
            bin_indices = np.clip(bin_indices, 0, nbins - 1)  # Clip valid bin indices
            my_bins[:, j] = bin_indices

        # Populate histogram
        for k in range(my_bins.shape[0]):
            if np.any(my_bins[k, :] == 0):
                continue
            c = np.dot(my_bins[k, :], hist_offset)
            my_hist.flat[c] += 1

        # Normalize the histogram
        ndhist[idx] = my_hist / np.sum(my_hist)
            
    # Compute DKL and JEF
    dklab = np.zeros((len(unique_labels), len(unique_labels)))
    dklba = np.zeros((len(unique_labels), len(unique_labels)))

    for i in range(len(unique_labels)):
        for j in range(i):
            # DKL(A || B)
            s1 = ndhist[i] * np.log(ndhist[i] / (ndhist[j] + epsilon))
            s1[ndhist[i] == 0] = 0
            dklab[i, j] = np.sum(s1)

            # DKL(B || A)
            s1 = ndhist[j] * np.log(ndhist[j] / (ndhist[i] + epsilon))
            s1[ndhist[j] == 0] = 0
            dklba[i, j] = np.sum(s1)

    # Symmetrize matrices
    dklab += dklab.T
    dklba += dklba.T

    # Compute Jeffrey's Divergence
    jef = dklab + dklba

    # Compute MIND similarity metric
    minds = 1 / (1 + jef)

    return minds, jef, dklab, dklba