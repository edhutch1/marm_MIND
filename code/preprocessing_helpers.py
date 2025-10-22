# Import packages
import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

def extract_voxels(img_file, parc_file, lut, ROI, Side, Code):
    """
    Extracts voxels from fully parcellated image

    Parameters:
    - img_file: path to image to be extracted
    - parc_file: path to parcellation image
    - lut: full lookup table containing all regions in Paxinos parcellation
    - ROI (string): column name containing region names
    - Side (string): column name containing hemisphere
    - Code (string): column name containing region codes
    
    Returns:
    - vox: dataframe with a row per voxel, with voxel value, region, and side
    """

    # Load image
    img_data = nib.load(img_file).get_fdata()

    # Load parcellation
    parc_data = nib.load(parc_file).get_fdata()

    # Initialise df to hold voxel data
    vox = lut[[ROI, Side]].copy()

    # Loop regions, extract values
    vox['Value'] = [
        img_data[parc_data == code] for code in tqdm(lut[Code])
    ]

    vox = vox.explode('Value')
    vox['Label'] = vox[ROI] + '_' + vox[Side]

    return vox

def extract_voxels_coarse(img_file, parc_file, lut, bm_codes, pax_codes, coarse_lut, ROI, Side, pax_codes_coarse):
    """
    Extracts voxels from coarsely parcelled image, for comparison with single-cell data from Krienen et al. (2023)

    Parameters:
    - img_file: path to image to be extracted
    - parc_file: path to parcellation image

    - lut: full lookup table containing all regions in Paxinos parcellation
    - bm_codes (string): column name containing brain_MINDs parcellation codes (string)
    - pax_codes (string): column name containing Paxinos parcellation codes (string)

    - coarse_lut: lookup table containing coarse regions
    - ROI (string): column name containing coarse region names
    - Side (string): column name containing hemisphere
    - pax_codes_coarse (string): column name containing a string list of Paxinos codes in a region
    
    Returns:
    - vox: dataframe with a row per voxel, with voxel value, region, and side
    """

    # Load image
    img_data = nib.load(img_file).get_fdata()

    # Load parcellation
    parc_data = nib.load(parc_file).get_fdata()

    # Initialise df to hold voxel data
    vox = coarse_lut[[ROI, Side]].copy()

    # Loop regions, extract values
    #vox['Value'] = [
    #    np.concatenate([img_data[parc_data == lut[bm_codes][np.where(lut[pax_codes] == code)[0][0]]] for code in codes]) for codes in coarse_lut[pax_codes_coarse]
    #]

    # Initialise variable to hold voxels per coarse ROI
    values_list = []

    # Loop coarse ROIs
    for codes in tqdm(coarse_lut[pax_codes_coarse]):

        # Initialise df to hold voxel values for this coarse ROI
        region_values = [] 

        # Map string list to integer list
        codes = list(map(int, codes.split()))
        
        # Loop over each fine code in this coarse ROI's codes
        for code in codes:
            # Find index in the full lookup table corresponding to this code
            idx = np.where(lut[pax_codes] == code)[0][0]
            
            # Map Paxinos code to Brain/Marmoset (BM) code
            bm_code = lut[bm_codes][idx]
            
            # Extract voxel values where parc_data == bm_code
            vals = img_data[parc_data == bm_code]
            
            # Append to list for this coarse region
            region_values.append(vals)
        
        # Concatenate all voxels from this group into one array
        region_values = np.concatenate(region_values)
        
        # Append to master list
        values_list.append(region_values)

    # Explode
    vox['Value'] = values_list
    vox = vox.explode('Value')
    vox['Label'] = vox[ROI] + '_' + vox[Side]

    return vox