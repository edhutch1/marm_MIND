# ----- Import packages ----- #

# General
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import product

# Correlation
from scipy.stats import spearmanr
import pingouin as pg

# Spatial autocorrelation testing
from brainsmash.mapgen.base import Base
from brainsmash.mapgen.eval import base_fit

# MLR
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# Machine learning
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score

# Clustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist

# T-test
from scipy.stats import ttest_ind


def threshold_network(net, dens):
    """
    Thresholds network to particular density. 
    Sets values below threshold to 0

    Parameters:
    - net: numpy array, matrix to threshold
    - dens: percentage density at which to threshold image to threshold

    Returns:
    - net_thresh: numpy arryay, thresholded matrix
    """
    net_thresh = net.copy()
    
    # Get triu
    triu_ind = np.triu_indices(net_thresh.shape[0], k=1)
    net_triu = net_thresh[triu_ind]

    # Get prctile to threshold network
    threshold = np.percentile(net_triu, (100-dens))  

    # Threshold
    net_thresh[net_thresh < threshold] = 0

    return net_thresh

def get_pct_intraclass(net, assignment):
    """
    Function to get percentage of edges which connect regions of the same cortical type (i.e. intra-class)
    Ignores edges equal to 0

    Parameters:
    - net: numpy array, similarity matrix
    - assignment: list, class assignments per region in similarity matrix

    Returns:
    - pct_intraclass, float, percentage of edges which are intra-class
    """
    n_intra = 0
    
    for i in range(net.shape[0]):
        for j in range(i+1, net.shape[0]):
            if net[i,j] != 0:
                if assignment[i] == assignment[j]:
                    n_intra += 1

    pct_intraclass = n_intra / np.count_nonzero(net[np.triu_indices(net.shape[0], k=1)])

    return pct_intraclass

def get_pct_interhemispheric(net):
    """
    Function to get proportion of homotopic (interhemispheric) edges
    Ignores edges equal to 0

    Parameters:
    - net: numpy array, similarity matrix

    Returns:
    - pct_interhem, float, percentage of all homotopic edges which exist, i.e. are non-0
    """

    interhem_edges = net[range(0,int(net.shape[0]/2), 1), range(int(net.shape[0]/2),net.shape[0], 1)]
    pct_interhem = len(interhem_edges[interhem_edges>0])/len(interhem_edges)

    return pct_interhem

def get_brainsmash_permuted_maps(map, dist, n_perm, surrogates_filename):
    """
    Use BrainSmash to generate surrogate maps preserving spatial autocorrelation structure

    Parameters:
    - map: numpy array, input map to permute (one hemisphere)
    - dist: numpy array, matrix of Euclidean distances between regional centroids
    - n_perm: number of permutations
    - surrogates_filename: filename at which to save surrogate maps
    """

    # Make sure to run on LH data!
    map = map[0:115]
    dist = dist[0:115,0:115]

    # Calculate brainsmashed surrogate maps
    print("BrainSmashing")
    base = Base(x=map, D=dist, resample=True, seed=42)
    surrogates = base(n=n_perm)

    # Save surrogates
    np.savetxt(surrogates_filename, surrogates)

def get_brainsmash_permuted_matrices(matrix, map, dist, n_perm, surrogates_filename):
    """
    Use BrainSmash permuted maps to permute matrix labels (left hemisphere only)

    Parameters:
    - matrix: numpy array, input matrix to permute (one hemisphere)
    - dist: numpy array, euclidean distance matrix (one hemisphere)
    - map: numpy array, map used to generate spatially autocorrelated null permutations, i.e. mean T1w/T2w or mean MBP expression (one hemisphere)
    - n_perm: number of permutations
    - surrogates_filename: filename containing surrogate maps. If this does not exist then some are generated
    
    Returns:
    - surrogate_matrices: 3-D numpy array, permuted networks
    - perm_ind_list: 2-D numpy array, BrainSmash permuted indices used to permute matrix
    """
    
    # If surrogates do not exist, calculate them
    if not os.path.exists(surrogates_filename):
        get_brainsmash_permuted_maps(map, dist, n_perm, surrogates_filename)

    # Load surrogates
    surrogates = pd.read_csv(surrogates_filename, delimiter=' ', header=None)
    surrogates = surrogates.round(5)

    # Round map data to same numerical precision as brainsmashed data
    map = np.round(map, decimals=5)

    # Generate null matrices
    surrogate_matrices = []
    perm_ind_list = []
    print("Generating null matrices")
    for perm in tqdm(range(n_perm)):
        # Find indices of permuted map in original map
        perm_ind = [np.where(surrogates.iloc[perm,:] == val)[0][0] for val in map]
        # Permute matrix
        matrix_perm = matrix[np.ix_(perm_ind, perm_ind)]

        surrogate_matrices.append(matrix_perm)
        perm_ind_list.append(perm_ind)

    return np.array(surrogate_matrices), np.array(perm_ind)

def get_coarse_matrix(mat_full, coarse_lut, lut):
    """
    Function to get coarse matrix from full matrix

    Parameters:
    - mat: numpy array, FULL matrix
    - coarse_lut: pandas df, coarse look-up table containing all region codes per coarse in a column called ['pax_codes_short']
    - lut: pandas df, FULL look-up table containing one regional code per region in the column ['Code']
    
    Returns:
    - mat_coarse: numpy array: coarse matrix
    """
    
    # Get number of ROIs
    n_rois = coarse_lut.shape[0]

    # Initialise coarse matrix
    mat_coarse = np.zeros((n_rois, n_rois))

    # Get coarse matrix by looping coarse ROIs, selecting corresponding regions from full matrix, then averaging across edges
    for i in range(n_rois):
        roi_i_pax_codes = [int(x) for x in coarse_lut.loc[i, "Codes"].split()]
        roi_i_ind = [np.where(lut['Code'] == pax_code)[0][0] for pax_code in roi_i_pax_codes]
        
        for j in range(i+1, n_rois):
            roi_j_pax_codes = [int(x) for x in coarse_lut.loc[j, "Codes"].split()]
            roi_j_ind = [np.where(lut['Code'] == pax_code)[0][0] for pax_code in roi_j_pax_codes]

            mat_coarse[i,j] = np.mean(mat_full[np.ix_(roi_i_ind, roi_j_ind)])

    return mat_coarse

def get_brainsmashed_region_correlation_p_val(map_a, map_b, regions_ind, dist, surrogates_filename, lh=True, n_perm=1000, test_type='two-tailed'):
    """
    Function to return the p value for the Spearman correlation between two regional maps
    Uses BrainSmash to generate spatial autocorrelation-preserving nulls

    Parameters:
    - map_a: numpy array, input map to permute
    - map_b: numpy array, comparison map
    - regions_ind: list, indices of regions to include in the correlation if map_b does not have data for all regions. Uses all regions if None.
    - dist: numpy array, Euclidean distance matrix
    - surrogates_filename: filename containing surrogate maps. If this does not exist then some are generated
    - lh: boolean, true if the correlation is to be conducted on the left hemisphere only
    - n_perm: number of permutations (default: 1000)
    - test_type: str, 'lower', 'upper', or 'two-tailed' (default: 'two-tailed')
    
    Returns:
    - r_emp: float, empirical r value
    - p_emp: float, spatial autocorrelation-preserving p value
    """

    # If surrogates do not exist, calculate them
    if not os.path.exists(surrogates_filename): 
        get_brainsmash_permuted_maps(map_a, dist, n_perm, surrogates_filename)

    # Load surrogates
    surrogates = pd.read_csv(surrogates_filename, delimiter=' ', header=None)
    surrogates = pd.concat([surrogates, surrogates], axis=1) if not lh else surrogates

    # Calculate spearman correlation between empirical maps
    map_a = map_a[regions_ind] if regions_ind is not None else map_a
    r_emp = spearmanr(map_a, map_b)[0]

    # Calculate null distribution of r values
    null_r_list = []
    for i in range(n_perm):
        null = surrogates.iloc[i,:].values
        null = null[regions_ind] if regions_ind is not None else null
        null_r_list.append(spearmanr(null, map_b)[0])
        
    # Calculate empirical p-value based on test type
    if test_type == 'two-tailed':
        p_emp = np.mean(np.abs(null_r_list) >= abs(r_emp))
    elif test_type == 'upper':
        p_emp = np.mean(null_r_list >= r_emp)
    elif test_type == 'lower':
        p_emp = np.mean(null_r_list <= r_emp)
    else:
        raise ValueError("Invalid test_type. Choose 'two-tailed', 'upper', or 'lower'.")

    return r_emp, p_emp

def get_brainsmashed_edge_correlation_p_val(mat_a, mat_b, regions_ind, map_a, dist, surrogates_filename, n_perm=1000, test_type='two-tailed'):
    """
    Calculate the p-value for correlation between two matrices (only use this for single hemisphere correlations)

    Parameters:
    - mat_a: numpy array, input matrix to permute 
    - mat_b: numpy array, comparison matrix (one hemisphere)
    - regions_ind: list, indices of regions to include in the correlation if map_b does not have data for all regions. Uses all regions if None.
    - map_a: numpy array, map used to generate spatially autocorrelated null permutations, i.e. mean T1w/T2w or mean MBP expression (one hemisphere)
    - dist: numpy array, euclidean distance matrix (one hemisphere)
    - surrogates_filename: filename containing surrogate maps. If this does not exist then some are generated
    - n_perm: number of permutations (default: 1000)
    - test_type: str, 'lower', 'upper', or 'two-tailed' (default: 'two-tailed')

    Returns:
    - r_emp: float, empirical r value
    - p_emp: float, spatial autocorrelation-preserving p value
    """

    # Calculate empirical correlation
    mat_a_filt = mat_a[regions_ind,:][:,regions_ind] if regions_ind is not None else mat_a
    triu_ind = np.triu_indices(mat_a_filt.shape[0], k=1)
    mat_a_triu = mat_a_filt[triu_ind]
    mat_b_triu = mat_b[triu_ind]
    r_emp = spearmanr(mat_a_triu, mat_b_triu)[0]

    # Get nulls
    mat_a_nulls, _ = get_brainsmash_permuted_matrices(mat_a, map_a, dist, n_perm, surrogates_filename)

    # Loop nulls, get correlation
    null_r_list = []
    print("Generating null statistics")
    for null in tqdm(mat_a_nulls):
        null = null[regions_ind,:][:,regions_ind] if regions_ind is not None else null
        null_triu = null[triu_ind]
        null_r_list.append(spearmanr(null_triu, mat_b_triu)[0])
    
    # Calculate empirical p-value based on test type
    if test_type == 'two-tailed':
        p_emp = np.mean(np.abs(null_r_list) >= abs(r_emp))
    elif test_type == 'upper':
        p_emp = np.mean(null_r_list >= r_emp)
    elif test_type == 'lower':
        p_emp = np.mean(null_r_list <= r_emp)
    else:
        raise ValueError("Invalid test_type. Choose 'two-tailed', 'upper', or 'lower'.")

    return r_emp, p_emp

def get_brainsmashed_edge_correlation_p_val_coarse(mat_a, 
                                                   mat_b, 
                                                   mat_a_full, 
                                                   map_a, 
                                                   dist, 
                                                   coarse_lut,
                                                   lut,
                                                   surrogates_filename, 
                                                   n_perm=1000, 
                                                   test_type='two-tailed'):
    """
    Calculate the p-value for correlation between two coarse matrices
    All parameters are LH only as gene expression data is from LH only

    Parameters:
    - mat_a: numpy array, COARSE input matrix to permute 
    - mat_b: numpy array, COARSE comparison matrix
    - mat_a_full: numpy array, FULL input matrix to permute 
    - map_a: numpy array, FULL map used to generate spatially autocorrelated null permutations, i.e. mean T1w/T2w or mean MBP expression
    - dist: numpy array, FULL euclidean distance matrix
    - surrogates_filename: filename containing surrogate maps. If this does not exist then some are generated
    - n_perm: number of permutations (default: 1000)
    - test_type: str, 'lower', 'upper', or 'two-tailed' (default: 'two-tailed')

    Returns:
    - r_emp: float, empirical r value
    - p_emp: float, spatial autocorrelation-preserving p value
    """

    # Calculate empirical correlation
    triu_ind = np.triu_indices(mat_a.shape[0], k=1)
    mat_a_triu = mat_a[triu_ind]
    mat_b_triu = mat_b[triu_ind]
    r_emp = spearmanr(mat_a_triu, mat_b_triu)[0]

    # Get nulls
    mat_a_full_nulls, _ = get_brainsmash_permuted_matrices(mat_a_full, map_a, dist, n_perm, surrogates_filename)

    # Loop nulls
    null_r_list = []
    print("Generating null statistics")
    for null in tqdm(mat_a_full_nulls):
        # Get coarse network from null
        null_coarse = get_coarse_matrix(null, coarse_lut, lut)

        # Calculate correlation
        null_coarse_triu = null_coarse[triu_ind]

        # Append correlation
        null_r_list.append(spearmanr(null_coarse_triu, mat_b_triu)[0])
    
    # Calculate empirical p-value based on test type
    if test_type == 'two-tailed':
        p_emp = np.mean(np.abs(null_r_list) >= abs(r_emp))
    elif test_type == 'upper':
        p_emp = np.mean(null_r_list >= r_emp)
    elif test_type == 'lower':
        p_emp = np.mean(null_r_list <= r_emp)
    else:
        raise ValueError("Invalid test_type. Choose 'two-tailed', 'upper', or 'lower'.")

    return r_emp, p_emp

def run_mlr(df, y, X):
    """
    Run a multiple linear regressions for each outcome (y) variable, output key values
    Interactions should be added as separate columns prior to function calling, e.g. df['Age_Sex'] = df['Age'] * df['Sex']

    Parameters:
    - df: pandas df, holding y and X
    - y: list, column names indicating outcome variables
    - X: list, column names indicating predictors to be used in each model

    Returns:
    - results_df: pandas df, containing intercept, b values, p values (corrected and uncorrected), and R2 for each precictor
    """

    # Initialise df to hold results
    results = []

    # Loop regions, compute linear model, save to results
    for outcome in tqdm(y):
        
        # Subset outcome
        y_df = df[outcome]

        # Subset predictors
        X_df = df[X]

        # Add intercept
        X_with_const = sm.add_constant(X_df)

        # Fit model
        mlr_sm = sm.OLS(y_df, X_with_const).fit()

        # Generate results dictionary
        current_results = {
            'outcome': outcome,
            'intercept': mlr_sm.params['const'],
            'r2':mlr_sm.rsquared
        }

        # Add betas and p-values for each predictor
        for predictor in X:
            current_results[f'b_{predictor}'] = mlr_sm.params[predictor]
            current_results[f'p_{predictor}'] = mlr_sm.pvalues[predictor]

        # Append current results to the full results dataframe
        results.append(current_results)

    # Convert to DataFrame
    result_df = pd.DataFrame(results)

    # Multiple comparisons correction for p-values (Benjamini-Hochberg)
    for predictor in X:
        result_df[f'p_{predictor}_corr'] = multipletests(result_df[f'p_{predictor}'], method='fdr_bh')[1]
    
    return result_df

def hierarchical_clustering(X, n_clusters):
    """
    Perform hierarchical clustering

    Parameters:
    - X: numpy array, matrix to be clustered
    - n_clusters: int, number of clusters

    Returns:
    - labels: numpy array, cluster assignment of each ROI
    """

    # Initialise clustering model
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage='ward'
    )

    # Fit model
    labels = model.fit_predict(X)

    return labels

def compute_overlap_matrix(df, col1='func_group', col2='cluster'):
    """
    Compute pairwise overlap between two categorical groupings of regions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least two categorical columns and a 'region' column.
    col1 : str, default='func_group'
        Column name for the first grouping (e.g. functional groups).
    col2 : str, default='cluster'
        Column name for the second grouping (e.g. data-driven clusters).

    Returns
    -------
    overlap_matrix : pandas.DataFrame
        Matrix (rows = groups in col1, columns = groups in col2) containing
        the proportion overlap, defined as:
            overlap = |set1 ∩ set2| / min(|set1|, |set2|)
        i.e. intersection normalized by the smaller set size.
        NaN if either set is empty.
    """

    # Get unique groups in each column
    groups1 = df[col1].unique()
    groups2 = df[col2].unique()

    # Initialise empty overlap matrix
    overlap_matrix = pd.DataFrame(index=groups1, columns=groups2, dtype=float)

    # Loop over all pairs of groups
    for g1, g2 in product(groups1, groups2):
        set1 = set(df[df[col1] == g1]['region'])
        set2 = set(df[df[col2] == g2]['region'])

        # Compute Szymkiewicz–Simpson overlap coefficient
        intersection = len(set1 & set2)
        min_size = min(len(set1), len(set2))
        if min_size == 0:
            overlap = np.nan
        else:
            overlap = intersection / min_size

        # Store in matrix
        overlap_matrix.loc[g1, g2] = overlap

    return overlap_matrix

def cluster_t_test_p_val(clusters, anat, map, dist, surrogates_filename, n_perm=1000, test_type='two-tailed'):
    """
    Perform t-test on clusters, extract p-value by comparing to spatial autocorrelation preserving nulls

    Parameters:
    - clusters: numpy array, cluster assignment per region
    - anat: numpy array, anatomical value per region (in this context: lamination or hierarchical level)
    - map: numpy array, map used to generate spatially autocorrelated null permutations, i.e. mean T1w/T2w or mean MBP expression (one hemisphere)
    - surrogates_filename: filename containing surrogate maps

    Returns:
    - t_emp: float, empirical t statistic
    - p_emp: float, spatial autocorrelation-preserving p value
    """

    # Compute empirical t-stat
    df = pd.DataFrame({'clust':clusters, 'anat':anat})
    df = df.dropna()
    groups = [df['anat'][df['clust'] == c] for c in np.unique(df['clust'])]
    t_emp, _ = ttest_ind(*groups, equal_var=False, nan_policy='omit')

    # Get nulls
    # If surrogates do not exist, calculate them
    if not os.path.exists(surrogates_filename): 
        get_brainsmash_permuted_maps(map, dist, n_perm, surrogates_filename)

    # Load surrogates
    surrogates = pd.read_csv(surrogates_filename, delimiter=' ', header=None)
    surrogates = surrogates.round(5)

    # Round map data to same numerical precision as brainsmashed data
    map = np.round(map, decimals=5)

    # Loop surrogates
    null_t_list = []
    for i in tqdm(range(n_perm)):

        # Permute cluster assignment
        perm_ind = [np.where(surrogates.iloc[i,:] == val)[0][0] for val in map]
        clust_perm = clusters[perm_ind]
        
        # Re-compute t-statistic
        df = pd.DataFrame({'clust':clust_perm, 'anat':anat})
        df = df.dropna()
        groups = [df['anat'][df['clust'] == c] for c in np.unique(df['clust'])]
        null_t_list.append(ttest_ind(*groups, equal_var=False, nan_policy='omit')[0])

    # Calculate empirical p-value based on test type
    if test_type == 'two-tailed':
        p_emp = np.mean(np.abs(null_t_list) >= abs(t_emp))
    elif test_type == 'upper':
        p_emp = np.mean(null_t_list >= t_emp)
    elif test_type == 'lower':
        p_emp = np.mean(null_t_list <= t_emp)
    else:
        raise ValueError("Invalid test_type. Choose 'two-tailed', 'upper', or 'lower'.")
        
    return t_emp, p_emp

def get_coph_dist_mat(X):
    """
    Compute the cophenetic distance matrix from an input matrix.

    Parameters:
    - X: numpy array, input matrix

    Returns:
    - coph_dists: numpy array, cophenetic distance matrix
    """

    # Pairwise Euclidean distances
    Y = pdist(X, metric='euclidean')

    # Hierarchical clustering with Ward's method
    Z = linkage(Y, method='ward')

    # Cophenetic correlation (c) and distances
    coph_dists = cophenet(Z)

    return coph_dists
