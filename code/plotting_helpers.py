# Import packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize

import seaborn as sns

from scipy.stats import spearmanr
from statannotations.Annotator import Annotator

# Import code
from stats_helpers import *

def matrix_plot(mat, edge_cmap, diverging, cbar_label, patches, boundaries, labels, save_path=None):
    """
    Plot matrix (left hemisphere) sorted by cortical zone

    Parameters:
    - mat: pandas df, matrix to plot
    - cmap: string, colour map to use
    - diverging: boolean, if True, uses diverging colour map
    - patches: boolean, if True, adds patches to left and top of matrix indicating cortical zone
    - labels: boolean, if True, adds labels to left of matrix indicating cortical zone
    - boundaries: boolean, if True, adds black boundaries separating cortical zone
    """

    # Define colour map for cortical zones
    cmap = {
        'MPC':  [0.5529, 0.8275, 0.7804],
        'OFC':  [1.0000, 1.0000, 0.7020],
        'DLP':  [0.7451, 0.7294, 0.8549],
        'VLP':  [0.9843, 0.5020, 0.4471],
        'INS':  [0.5020, 0.6941, 0.8275],
        'PCR':  [0.9922, 0.7059, 0.3843],
        'ACC':  [0.7020, 0.8706, 0.4118],
        'PirC': [0.9882, 0.8039, 0.8980],
        'PPC':  [0.8510, 0.8510, 0.8510],
        'PRM':  [0.7373, 0.5020, 0.7412],
        'SSC':  [0.8000, 0.9216, 0.7725],
        'AUD':  [1.0000, 0.9294, 0.4353],
        'VIS':  [1.0000, 0.7137, 0.7569],
        'LIT':  [0.7843, 0.7843, 0.9804],
        'VTC':  [1.0000, 0.8627, 0.7843],
    }

    # Load groups
    lut = pd.read_csv('data/lut_master.csv', index_col=0)
    lut = lut[lut['Side'] == 'L']
    lut['sortnum'] = range(lut.shape[0])
    lut['zone_label'] = lut['zone'] + '_' + lut['Side']

    # Define zonal order
    order_1h = ['MPC_L', 'OFC_L', 'DLP_L', 'VLP_L', 'INS_L', 'PCR_L', 'ACC_L', 'PirC_L', 
                'PPC_L', 'PRM_L', 'SSC_L', 'AUD_L', 'VIS_L', 'LIT_L', 'VTC_L']

    # Reorder labels by cortical zone
    lut['zone_label'] = pd.Categorical(lut['zone_label'], categories=order_1h, ordered=True)
    lut_reordered = lut.sort_values(by=['zone_label', 'Code']).reset_index(drop=True)

    # Re-order matrix
    mat_reordered = mat.iloc[lut_reordered.sortnum, lut_reordered.sortnum]

    # Map the cortical type to color
    row_colors = lut_reordered['zone'].map(lambda x: cmap[x])
    col_colors = lut_reordered['zone'].map(lambda x: cmap[x])

    # Define labels
    group_labs = ['MPC', 'OFC', 'DLP', 'VLP', 'INS', 'PCR', 'ACC', 'PirC', 'PPC', 'PRM', 'SSC', 'AUD', 'VIS', 'LIT', 'VTC']

    # Create the heatmap
    plt.figure(figsize=(8, 8))

    if diverging:
        ax = sns.heatmap(mat_reordered.iloc[0:115, 0:115], cmap=edge_cmap, cbar=False, 
                        xticklabels=False, yticklabels=False, cbar_kws={"shrink": 0.8},
                        vmin=-np.max(np.abs(mat.values)), vmax=np.max(np.abs(mat.values)), rasterized=True)
    elif not diverging:
        ax = sns.heatmap(mat_reordered.iloc[0:115, 0:115], cmap=edge_cmap, cbar=False, 
                        xticklabels=False, yticklabels=False, cbar_kws={"shrink": 0.8},
                        vmin=0, vmax=1, rasterized=True)

    # Add colourbar below
    cbar = plt.colorbar(
        ax.collections[0],
        ax=ax,
        orientation="horizontal",
        fraction=0.04,
        pad=0.05
    )

    # Customize color bar
    cbar.set_label(cbar_label, fontsize=30)
    cbar.ax.tick_params(labelsize=20)
    cbar.outline.set_visible(False)

    # Add patches
    if patches:
        for i in range(115):
            plt.gca().add_patch(mpatches.Rectangle((-3, i), 3, 1, color=row_colors[i], clip_on=False, lw=1, edgecolor='face'))  # Left
            plt.gca().add_patch(mpatches.Rectangle((i, -3), 1, 3, color=col_colors[i], clip_on=False, lw=1, edgecolor='face'))  # Top
    
    # Add boundaries
    if boundaries:
        group_boundaries = [5, 13, 20, 25, 33, 43, 47, 48, 58, 66, 73, 87, 100, 107]
        
        for boundary in group_boundaries:
            plt.axhline(y=boundary, color='black', linewidth=1)  # Horizontal line
            plt.axvline(x=boundary, color='black', linewidth=1)  # Vertical line

    # Add labels
    if labels:
        for i, (name, boundary) in enumerate(zip(group_labs, group_boundaries[:-1])):
            
            mid = (boundary + group_boundaries[i + 1]) / 2  # Midpoint of the section
            plt.text(-4, mid, name, fontsize=10, ha='right', va='center', color='black')  

    plt.xticks([])
    plt.yticks([])

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

def cortical_zone_boxplot(df, x_lab='', h_line=False, save_path=None):
    """
    Generates boxplot by cortical zone

    Parameters:
    - df: pandas df with "vals" (regional values to plot) and "groups" (cortical zones) columns
    - x_lab: string, x axis label
    - h_line: boolean, if true, plots a horizontal line at b_Age = 0 (default = None)
    - save_path: string, path to save (default = "None": no save)

    """

    # Your custom functional group colors
    cmap = {
        'MPC':  [0.5529, 0.8275, 0.7804],
        'OFC':  [1.0000, 1.0000, 0.7020],
        'DLP':  [0.7451, 0.7294, 0.8549],
        'VLP':  [0.9843, 0.5020, 0.4471],
        'INS':  [0.5020, 0.6941, 0.8275],
        'PCR':  [0.9922, 0.7059, 0.3843],
        'ACC':  [0.7020, 0.8706, 0.4118],
        'PirC': [0.9882, 0.8039, 0.8980],
        'PPC':  [0.8510, 0.8510, 0.8510],
        'PRM':  [0.7373, 0.5020, 0.7412],
        'SSC':  [0.8000, 0.9216, 0.7725],
        'AUD':  [1.0000, 0.9294, 0.4353],
        'VIS':  [1.0000, 0.7137, 0.7569],
        'LIT':  [0.7843, 0.7843, 0.9804],
        'VTC':  [1.0000, 0.8627, 0.7843],
    }

    # Prepare your DataFrame
    df.dropna(inplace=True)

    # Calculate medians and sort group order
    medians = df.groupby('groups')['vals'].median().sort_values().index

    # Now create a palette using your lookup
    palette = {group: cmap[group] for group in medians}

    # Plotting
    plt.figure(figsize=(3, 3))
    sns.boxplot(
        x='vals',
        y='groups',
        data=df,
        order=medians,
        palette=palette,   # Use your custom pastel palette
        showfliers=False,
        medianprops={"color": 'k', "linewidth": 1.5}
    )
    sns.stripplot(
        x='vals',
        y='groups',
        data=df,
        color="k",
        size=3,
        alpha=0.6,
        order=medians
    )

    plt.xlabel(x_lab, fontsize=16, fontfamily='Arial')
    plt.ylabel('', fontsize=16, fontfamily='Arial')
    plt.tight_layout()

    if h_line:
        plt.axvline(0, linestyle='dashed', color='k')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

def SA_position_correlation(to_plot, 
                            dist, 
                            surrogates_filename, 
                            n_perm=1000, 
                            test_type='two-tailed',
                            x_lab='', 
                            colour_points_by="y", 
                            cmap="Greys_r", 
                            title_stats=True, 
                            location=None, 
                            save_path=None):
    """
    Plots correlation with hierarchical level

    Parameters:
    - to_plot: numpy array, regional values to plot
    - x_lab: string, x-axis label
    - colour_points_by: 'string', colour points by x or y axis (default: "y", i.e. hierarchical level)
    - cmap: cmap. Default is "Greys_r"
    - title_stats: boolean, if True, adds stats to title, else adds to location specified
    - location: string, if title_stats is False, specify location of stats text. Options are "top right", "top left", "bottom left", "bottom right"
    - save_path: path to save, default = "None", no save

    """

    # Import lookup-table and filter
    lut = pd.read_csv('data/lut_master.csv', index_col=0)
    lut = lut[lut['ROI'] != 'APir'].reset_index(drop=True) 
    lut = lut[lut['Side'] == 'L']
    to_remove = ['Pir', 'Ent', 'A35', 'A29a-c', 'A24a'] # These regions do not have a hierarchical position assignment
    ind = ~lut.ROI.isin(to_remove)

    # Get hierarchical position assignments
    lvl = lut[~lut['ROI'].isin(to_remove)]['lvl'].values

    # Get r and p value
    lh=True
    r, p = get_brainsmashed_region_correlation_p_val(to_plot, 
                                                     lvl, 
                                                     ind, 
                                                     dist, 
                                                     surrogates_filename, 
                                                     lh,
                                                     n_perm, 
                                                     test_type)

    # Filter regions for plotting
    to_plot_filt = to_plot[ind]

    # Create plot
    df = pd.DataFrame({'x':to_plot_filt, 'y':lvl})
    plt.figure(figsize=(3, 3))
    plt.scatter(df["x"], df["y"], c=df[colour_points_by], cmap=cmap, edgecolors='black', s=50)
    sns.regplot(x="x", y="y", data=df, scatter=False, color='black')
    plt.ylabel('Hierarchical position', fontsize=16, fontfamily='Arial')
    plt.xlabel(x_lab, fontsize=16, fontfamily='Arial')

    if p < 0.001:
        p_text = '< 0.001'
    else:
        p_text = f'= {p:.3f}'

    if title_stats:
        plt.title(fr'$r = {r:.2f}$' + '\n' + fr'$p_{{\mathrm{{variogram}}}} = {p_text}$', fontsize=16, fontfamily='Arial')
    else:
        ax = plt.gca()
        x1, x2 = ax.get_xlim(); y1, y2 = ax.get_ylim()
        
        if location == "top right":
            xpos = x1 + 0.95*(x2-x1)
            ypos = y1 + 0.95*(y2-y1)
            ha, va = "right", "top"
        elif location == "top left":
            xpos = x1 + 0.05*(x2-x1)
            ypos = y1 + 0.95*(y2-y1)
            ha, va = "left", "top"
        elif location == "bottom left":
            xpos = x1 + 0.05*(x2-x1)
            ypos = y1 + 0.05*(y2-y1)
            ha, va = "left", "bottom"
        elif location == "bottom right":
            xpos = x1 + 0.95*(x2-x1)
            ypos = y1 + 0.05*(y2-y1)
            ha, va = "right", "bottom"

        ax.text(xpos, ypos, fr'$r = {r:.2f}$' + '\n' + fr'$p_{{\mathrm{{variogram}}}} {p_text}$', ha=ha, va=va, fontsize=12, fontfamily='Arial')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

def age_bin_correlations(df, age_bins, cbar_min):
    """
    Divides dataset into bins, calculates mean network for each age bin, cross-correlates mean networks
    
    Parameters:
    - df: pandas df, df containing edges per subject (first four columns are subject data, including ['Age'], then edge weights)
    - age_bins: np.arange(min age, bin width, max age)
    - cbar_min: float, minimum value to display on colourbar
    """

    # Get number of edges
    n_edges = df.iloc[:,4:].shape[1]

    # Initialise variables to hold number of subjects per bin and averaged network for each bin
    subject_counts = []
    bin_mean_networks = np.zeros((len(age_bins)-1, n_edges))

    # Loop each bin, extract number of subjects and mean network
    for i in range(len(age_bins)-1):

        # Subset the dataframe between age bin limits
        # Because there are only 3 subjects below 8 months (aged ~7.5 months), include them in the first bin
        if i == 0:
            binned_edges = df.loc[df['Age'] < age_bins[i+1], df.columns[4:]]
        else:
            binned_edges = df.loc[(df['Age'] > age_bins[i]) & (df['Age'] < age_bins[i+1]), df.columns[4:]]
        
        # Append number of subjects
        subject_counts.append(binned_edges.shape[0])

        # Append mean network
        bin_mean_networks[i,:] = binned_edges.mean(axis=0)


    # ----- Cross-correlate ----- #

    bin_corrs = np.corrcoef(bin_mean_networks)


    # ----- Plot ----- #

    # Initialize the JointGrid
    sns.set_theme(style="ticks")
    subj_counts_df = pd.DataFrame({'Bin': np.arange(len(subject_counts)), 'Count': subject_counts})
    g = sns.JointGrid(data=subj_counts_df, x='Bin', y="Count", marginal_ticks=True)
    g.fig.set_size_inches(3, 3)

    # Add heatmap to main axes
    cax = g.figure.add_axes([.87, .25, .05, 0.5])  # Position for the colorbar

    h = sns.heatmap(bin_corrs,
                    cmap = 'Reds_r',
                    cbar=True, cbar_ax=cax, 
                    ax=g.ax_joint,
                    xticklabels=[f"{age_bins[i]:.2f}" for i in range(len(age_bins) - 1)], 
                    yticklabels=[f"{age_bins[i]:.2f}" for i in range(len(age_bins) - 1)],
                    rasterized=True)

    # Colourbar
    cbar = h.figure.colorbar(h.collections[0], cax=cax)
    cbar_ticks = np.linspace(cbar_min, 1, 3)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{tick:.2f}" for tick in cbar_ticks], fontsize=14)
    cbar.set_label('Pearson R', fontsize=20, labelpad=10, fontfamily='Arial')

    # Add barplot to x marginal axes
    g.ax_marg_x.bar(
        subj_counts_df['Bin'] + 0.5,
        subj_counts_df['Count'],
        color="gray",
        width=0.8
    )

    # Display counts above bars in marginal plot
    for i, count in enumerate(subj_counts_df['Count']):
        g.ax_marg_x.text(
            x=i + 0.5,
            y=count + 1,
            s=str(count),
            ha='center',
            va='bottom',
            fontsize=14
        )

    # Housekeeping
    g.ax_marg_y.set_visible(False) # Remove y marginal plot
    g.ax_marg_x.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    g.ax_marg_x.set_title('Subjects per bin', fontsize=14, pad=20, fontfamily='Arial')
    tick_labels = [f"{int(x * 12)}" for x in age_bins[:-1]]; tick_labels[0] = 7
    g.ax_joint.set_xticklabels(tick_labels, rotation=90, fontsize=14, fontfamily='Arial')
    g.ax_joint.set_yticklabels(tick_labels, rotation=0, fontsize=14, fontfamily='Arial')
    g.ax_joint.set_xlabel('Age (months)', fontsize=20, fontfamily='Arial')
    g.ax_joint.set_ylabel('Age (months)', fontsize=20, fontfamily='Arial')
    g.ax_joint.yaxis.set_tick_params(rotation=0, labelsize=14)
    g.ax_joint.xaxis.set_tick_params(rotation=0, labelsize=14)

    plt.show()

def plot_dendrogram_and_matrix(X, n_clusters, cluster_cmap, cbar_label=None, save_path=None):
    """
    Performs agglomerative hierarchical clustering on input matrix, then plots clustered matrix with associated dendrogram
    
    Cluster rows of matrix X and plot dendrogram on top with row color annotations for cluster assignments.

    Parameters:
    - X: numpy array, matrix to be clustered
    - n_clusters: int, number of clusters
    - cluster_cmap: dictionary, cluster colour mapping
    - cbar_label: string, title for colourbar
    """

    # ----- Perform clustering and get labels ----- #

    cluster_labels = hierarchical_clustering(X, n_clusters)


    # ----- Define cluster color map ----- #

    row_colors = pd.Series(cluster_labels).map(cluster_cmap).to_numpy()


    # ----- Plot ----- #

    g = sns.clustermap(
        X,
        method='ward',
        metric='euclidean',
        row_cluster=True,
        col_cluster=True,
        cmap='coolwarm', 
        vmin = -np.max(np.abs(X)),
        vmax = np.max(np.abs(X)),
        figsize=(6, 5),
        xticklabels=[],
        yticklabels=[],
        row_colors=row_colors,
        col_colors=row_colors,
        dendrogram_ratio=(0.15, 0),
        cbar_kws={"orientation": "horizontal"},
        cbar_pos=(0.3, -0.05, 0.55, 0.03),
        rasterized=True
    )

    # Retain just column dendrogram
    g.ax_row_dendrogram.set_visible(True)
    g.ax_col_dendrogram.set_visible(True)

    # Colourbar
    g.cax.xaxis.set_label_position('bottom')
    g.cax.set_xlabel(cbar_label, fontsize=22, labelpad=10, fontfamily='Arial')
    g.cax.tick_params(labelsize=14, rotation=0)

    # Save
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

def plot_2_cluster_lamination(df, map, dist, surrogates_filename, n_perm=1000, test_type='two-tailed', save_path=None):
    """
    Performs t-test comparing degree of lamination in two clusters
    Calculates p value by comparing to 1000 spatial autocorrelation preserving nulls
    Plots result
    
    Parameters:
    - df: pandas dataframe, dataframe containing cluster assignments and anatomical data
    - anat
    - X: numpy array, matrix to be clustered
    - n_clusters: int, number of clusters
    - cluster_cmap: dictionary, cluster colour mapping
    - cbar_label: string, title for colourbar
    """

    # Order in which to plot clusters
    ordered_clusters = ['Association', 'Sensory'] 

    # Remove nans for plotting
    df_narm = df.dropna()

    # Plot setup
    plt.figure(figsize=(2.5, 3))
    ax = sns.violinplot(
        data=df_narm,
        x='cluster_name',
        y='lamination',
        order=ordered_clusters,
        inner='box',
        linewidth=1.5,
        cut=0
    )

    # Compute mean cortical type for coloring violins
    means = df_narm.groupby('cluster_name')['lamination'].mean().reindex(ordered_clusters)
    norm = Normalize(vmin=df_narm['lamination'].min(), vmax=df_narm['lamination'].max())
    cmap = get_cmap('viridis')

    # Color violins based on mean values
    for i, violin in enumerate(ax.collections[:len(ordered_clusters)]):
        color = cmap(norm(means.iloc[i]))
        violin.set_facecolor(color)
        violin.set_edgecolor('none')

    # Get t-statistic and p value
    t, p = cluster_t_test_p_val(clusters=df['cluster'].values, 
                                anat=df['lamination'].values, 
                                map = map,
                                dist=dist,
                                surrogates_filename=surrogates_filename,
                                n_perm=n_perm, 
                                test_type=test_type
                                )

    # Annotate plot with p value
    if p < 0.001:
        p_text = '***'
    elif p < 0.01:
        p_text = '**'
    elif p < 0.05:
        p_text = '*'
    else:
        p_text = 'ns'
    
    pairs = [("Association", "Sensory")]
    annot = Annotator(ax, pairs, data=df_narm, x='cluster_name', y='lamination', order=ordered_clusters)
    annot.set_custom_annotations([f"{p_text}"])
    annot.configure(
        test=None,
        loc='outside',
        text_offset=3,
        fontsize=10,
    )
    annot.annotate()

    # Format axes
    plt.xlabel('', fontsize=16)
    ax.set_xticklabels(['Assoc', 'Sensory'], fontsize=12, fontfamily='Arial')
    ax.tick_params(axis='x', length=0)
    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylim(1, 6)

    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.1, pad=0.05)
    cbar.set_label("Lamination", fontsize=16)
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels([1, 2, 3, 4, 5, 6])
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_visible(False)
    
    plt.tight_layout()
    plt.draw()
    
    pos_ax = ax.get_position()
    pos_cb = cbar.ax.get_position()
    cbar.ax.set_position([pos_cb.x0, pos_ax.y0, pos_cb.width, pos_cb.height])


    # Save
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

    # Print t-statistic and p value
    print(f"t = {t:.3f}, p = {p:.3f}")

def plot_2_cluster_SA_position(df, map, dist, surrogates_filename, n_perm=1000, test_type='two-tailed', save_path=None):
    """
    Performs t-test comparing mean hierarchical position of two clusters
    Calculates p value by comparing to 1000 spatial autocorrelation preserving nulls
    Plots result
    
    Parameters:
    - df: pandas dataframe, dataframe containing cluster assignments and anatomical data
    - anat
    - X: numpy array, matrix to be clustered
    - n_clusters: int, number of clusters
    - cluster_cmap: dictionary, cluster colour mapping
    - cbar_label: string, title for colourbar
    """

    # Order in which to plot clusters
    ordered_clusters = ['Association', 'Sensory'] 

    # Remove nans for plotting
    df_narm = df.dropna()

    # Plot setup
    plt.figure(figsize=(2.5, 3))
    ax = sns.violinplot(
        data=df_narm,
        x='cluster_name',
        y='lvl',
        order=ordered_clusters,
        inner='box',
        linewidth=1.5,
        cut=0
    )

    # Compute mean cortical type for coloring violins
    means = df_narm.groupby('cluster_name')['lvl'].mean().reindex(ordered_clusters)
    norm = Normalize(vmin=df_narm['lvl'].min(), vmax=df_narm['lvl'].max())
    cmap = get_cmap('Greys')

    # Color violins based on mean values
    for i, violin in enumerate(ax.collections[:len(ordered_clusters)]):
        color = cmap(norm(means.iloc[i]))
        violin.set_facecolor(color)
        violin.set_edgecolor('none')

    # Get t-statistic and p value
    t, p = cluster_t_test_p_val(clusters=df['cluster'].values, 
                                anat=df['lvl'].values, 
                                map = map,
                                dist=dist,
                                surrogates_filename=surrogates_filename,
                                n_perm=n_perm, 
                                test_type=test_type
                                )

    # Annotate plot with p value
    if p < 0.001:
        p_text = '***'
    elif p < 0.01:
        p_text = '**'
    elif p < 0.05:
        p_text = '*'
    else:
        p_text = 'ns'
    
    pairs = [("Association", "Sensory")]
    annot = Annotator(ax, pairs, data=df_narm, x='cluster_name', y='lvl', order=ordered_clusters)
    annot.set_custom_annotations([f"{p_text}"])
    annot.configure(
        test=None,
        loc='outside',
        text_offset=3,
        fontsize=10,
    )
    annot.annotate()

    # Format axes
    plt.xlabel('', fontsize=16)
    ax.set_xticklabels(['Assoc', 'Sensory'], fontsize=12, fontfamily='Arial')
    ax.tick_params(axis='x', length=0)
    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_yticklabels([])

    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.1, pad=0.05)
    cbar.set_label("Hierarchical position", fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_visible(False)
    
    plt.tight_layout()
    plt.draw()
    
    pos_ax = ax.get_position()
    pos_cb = cbar.ax.get_position()
    cbar.ax.set_position([pos_cb.x0, pos_ax.y0, pos_cb.width, pos_cb.height])

    # Save
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')


    plt.show()

    # Print t-statistic and p value
    print(f"t = {t:.3f}, p = {p:.3f}")

def cluster_zone_overlap(df, cluster_cmap, cluster_strings, overlap_threshold, save_path=None):
    """
    Visualize overlap between clusters and functional groups.

    Parameters
    ----------
    df : pandas.DataFrame, DataFrame with columns 'region', 'func_group', and 'cluster'.
    cluster_cmap : dict, Mapping from cluster label to RGB color (0–1 values).
    cluster_strings: string list, ordered names of clusters
    overlap_threshold: float, threshold above which overlap lines are drawn
    """

    # ----- Define colour maps ----- #
    zone_cmap = {
        'MPC':  [0.5529, 0.8275, 0.7804],
        'OFC':  [1.0000, 1.0000, 0.7020],
        'DLP':  [0.7451, 0.7294, 0.8549],
        'VLP':  [0.9843, 0.5020, 0.4471],
        'INS':  [0.5020, 0.6941, 0.8275],
        'PCR':  [0.9922, 0.7059, 0.3843],
        'ACC':  [0.7020, 0.8706, 0.4118],
        'PirC': [0.9882, 0.8039, 0.8980],
        'PPC':  [0.8510, 0.8510, 0.8510],
        'PRM':  [0.7373, 0.5020, 0.7412],
        'SSC':  [0.8000, 0.9216, 0.7725],
        'AUD':  [1.0000, 0.9294, 0.4353],
        'VIS':  [1.0000, 0.7137, 0.7569],
        'LIT':  [0.7843, 0.7843, 0.9804],
        'VTC':  [1.0000, 0.8627, 0.7843],
    }

    # Convert RGB values to hex for plotting
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % tuple(int(255 * x) for x in rgb)

    zone_cmap_rgb = {k: rgb_to_hex(v) for k, v in zone_cmap.items()}
    ordered_zones = ['MPC', 'OFC', 'DLP', 'VLP', 'ACC', 'INS', 'LIT', 'VTC', 'PirC','PRM', 'SSC', 'AUD', 'VIS', 'PCR', 'PPC']
    

    # ----- Compute overlap matrix ----- #

    overlap_df = compute_overlap_matrix(df, col1='zone', col2='cluster')


    # ----- Plot ----- #

    # Housekeeping
    clusters = overlap_df.columns.tolist()
    x_cluster, x_zone = 0.1, 0.9
    y_cluster = np.linspace(1, 0, len(clusters))
    y_zone = np.linspace(1, 0, len(ordered_zones))


    plt.figure(figsize=(3.75, 3.5))
    ax = plt.gca()

    # Colormap for overlap intensity
    cmap = get_cmap("Grays")
    norm = Normalize(vmin=0, vmax=1)

    # Plot edges
    for j, cl in enumerate(clusters):
        for i, fg in enumerate(ordered_zones):
            if fg in overlap_df.index and cl in overlap_df.columns:
                overlap = overlap_df.loc[fg, cl]
                if overlap > overlap_threshold:
                    ax.plot(
                        [x_cluster, x_zone],
                        [y_cluster[j], y_zone[i]],
                        color=cmap(norm(overlap)),
                        linewidth=3,
                        alpha=0.8
                    )

    # Plot cluster nodes (left)
    for j, cl in enumerate(clusters):
        ax.plot(x_cluster, y_cluster[j], 'o',
                color=list(cluster_cmap.values())[j], markersize=12)
        label = cluster_strings[j] if j < len(cluster_strings) else cl
        ax.text(x_cluster - 0.05, y_cluster[j], label,
                va='center', ha='right', fontsize=18, fontfamily='Arial')

    # Plot functional group nodes (right)
    for i, fg in enumerate(ordered_zones):
        ax.plot(x_zone, y_zone[i], 'o',
                color=zone_cmap_rgb.get(fg, 'black'), markersize=12)
        ax.text(x_zone + 0.05, y_zone[i], fg,
                va='center', ha='left', fontsize=12, fontfamily='Arial')

    # Final formatting
    ax.axis('off')
    plt.tight_layout()

    # Colorbar for overlap intensity
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label("Overlap Coefficient", fontsize=18, fontfamily='Arial')
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_visible(False)

    # Save
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
