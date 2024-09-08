#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gradient_utils.py
@Time    :   2024/09/03 09:47:17
@Author  :   Tao Zhou
@Version :   1.0
@Contact :   zhotoa@foxmail.com
'''
from typing import List,Optional, Union, Literal, Tuple
import statsmodels.api as sm
from anndata import AnnData
import pandas as pd
import numpy as np
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
from patsy import bs, cr, dmatrix
from scipy import stats
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from tqdm import tqdm
import numpy as np
import statsmodels.api as sm

def lrt(full: GLMResultsWrapper, restr: GLMResultsWrapper) -> np.float64:
    """Perform likelihood-ratio test on the full model and constrained model.
    Args:
        full: The regression model without constraint.
        restr: The regression model after constraint.
    Returns:
        The survival probability (1-cumulative probability) to observe the likelihood ratio for the constrained
        model to be true.
    """
    llf_full = full.llf
    llf_restr = restr.llf
    df_full = full.df_resid
    df_restr = restr.df_resid
    lrdf = df_restr - df_full
    lrstat = -2 * (llf_restr - llf_full)
    lr_pvalue = stats.chi2.sf(lrstat, df=lrdf)
    return lr_pvalue


def diff_test_helper(
    data: pd.DataFrame,
    fullModelFormulaStr: str = "~cr(time, df=3)",
    reducedModelFormulaStr: str = "~1",
) -> Union[Tuple[Literal["fail"], Literal["NB2"], Literal[1]], Tuple[Literal["ok"], Literal["NB2"], np.ndarray],]:
    """A helper function to generate required data fields for differential gene expression test.
    Args:
        data: The original dataframe containing expression data.
        fullModelFormulaStr: A formula string specifying the full model in differential expression tests (i.e.
            likelihood ratio tests) for each gene/feature. Defaults to "~cr(integral_time, df=3)".
        reducedModelFormulaStr: A formula string specifying the reduced model in differential expression tests (i.e.
            likelihood ratio tests) for each gene/feature. Defaults to "~1".
    Returns:
        A tuple [parseResult, family, pval], where `parseResult` should be "ok" or "fail", showing whether the provided
        dataframe is successfully parsed or not. `family` is the distribution family used for the expression responses
        in statsmodels, currently only "NB2" is supported. `pval` is the survival probability (1-cumulative probability)
        to observe the likelihood ratio for the constrained model to be true. If parsing dataframe failed, this value is
        set to be 1.
    """
    transformed_x = dmatrix(fullModelFormulaStr, data, return_type="dataframe")
    transformed_x_null = dmatrix(reducedModelFormulaStr, data, return_type="dataframe")
    expression = data["expression"]
    try:
        nb2_family = sm.families.NegativeBinomial(alpha = 1)  # (alpha=aux_olsr_results.params[0])
        nb2_full = sm.GLM(expression, transformed_x, family=nb2_family, ).fit()
        nb2_null = sm.GLM(expression, transformed_x_null, family=nb2_family).fit()
    except:
        return ("fail", "NB2", 1)
    pval = lrt(nb2_full, nb2_null)
    return ("ok", "NB2", pval)


def glm_degs(
    adata: AnnData,
    genes: Optional[List[str]] = None,
    factors: str = "gradient"
) -> None:
    """Differential genes expression tests using generalized linear regressions.
    The results would be stored in the adata's .uns["glm_degs"] annotation and the update is inplace.
    Tests each gene for differential expression as a function of integral time (the time estimated via the reconstructed
    vector field function) or pseudotime using generalized additive models with natural spline basis. This function can
    also use other covariates as specified in the full (i.e `~clusters`) and reduced model formula to identify
    differentially expression genes across different categories, group, etc.
    glm_degs relies on statsmodels package and is adapted from the `differentialGeneTest` function in Monocle. Note that
    glm_degs supports performing deg analysis for any layer or normalized data in your adata object. That is you can
    either use the total, new, unspliced or velocity, etc. for the differential expression analysis.
    Args:
        adata: An AnnData object.
        genes: The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is
            used. Defaults to None.
        The distribution family used for the expression responses in statsmodels. Currently, always uses `NB2`
            and this is ignored. NB model requires us to define a parameter alpha which it uses to express the
            variance in terms of the mean as follows: variance = mean + alpha mean^p. When p=2, it corresponds to
            the NB2 model. In order to obtain the correct parameter alpha (sm.genmod.families.family.NegativeBinomial
            (link=None, alpha=1.0), by default it is 1), we use the auxiliary OLS regression without a constant from
            Messrs Cameron and Trivedi. More details can be found here:
            https://towardsdatascience.com/negative-binomial-regression-f99031bb25b4. Defaults to "NB2".
    Raises:
        ValueError: `X_data` is provided but `genes` does not correspond to its columns.
        Exception: Factors from the model formula `fullModelFormulaStr` invalid.
    """
    if genes is None:
        genes = adata.var.index.tolist()
    df_factors = adata.obs[[factors]]
    df_factors['expression'] = 0
    sparse = issparse(adata.X)
    deg_df = pd.DataFrame(index=genes, columns=["status", "family", "pval"])
    for i, gene in tqdm(enumerate(genes),  "Detecting gradient dependent genes via Generalized Additive Models (GAMs)",):
        expression = adata[:, gene].X.toarray().flatten() if sparse else adata[:, gene].flatten()
        df_factors.loc[:,"expression"] = expression
        result = diff_test_helper(df_factors, '~grid_field', '~1')
        deg_df.loc[gene, ["status", "family", "pval"]] = result
    deg_df["qval"] = multipletests(deg_df["pval"], method="fdr_bh")[1]
    
    deg_df['pos_rate'] = 0.0
    for i in deg_df.index.to_list():
        deg_df.at[i, 'pos_rate'] = np.sum(adata[:,i].X.A != 0) / len(adata[:,i].X.A)
    deg_df['gene'] = deg_df.index.to_list()
    deg_df['pval'] = deg_df['pval'].astype(np.float32)
    
    return deg_df

def digitize_general(
    pc: np.ndarray,
    adj_mtx: np.ndarray,
    boundary_lower: np.ndarray,
    boundary_upper: np.ndarray,
    max_itr: int = 1e6,
    lh: float = 1,
    hh: float = 100,
) -> np.ndarray:
    """Calculate the "heat" for a general point cloud of interests by solving a PDE, partial differential equation,
    the heat equation. The two polar boundaries are given by their indices within the point cloud. The neighbor network
    of the point cloud is given as an adjacency matrix.

    Args:
        pc: An array of 3-D coordinates, representing the point cloud.
        adj: A 2-D adjacency matrix of the neighbor network.
        boundary_low: The indices of points selected as lower boundary in the point cloud.
        boundary_low: The indices of points selected as upper boundary in the point cloud.
        max_itr: Maximum number of iterations dedicated to solving the heat equation.
        lh: lowest digital-heat (temperature). Defaults to 1.
        hh: highest digital-heat (temperature). Defaults to 100.

    Returns:
        An array of "heat" values of each point in the point cloud.
    """

    mask_field = np.zeros(len(pc))
    mask_field[boundary_lower] = lh
    mask_field[boundary_upper] = hh

    max_err = 1e-5
    err = 1
    itr = 0
    grid_field = mask_field.copy()

    while (err > max_err) and (itr <= max_itr):
        grid_field_pre = grid_field.copy()

        grid_field = np.matmul(grid_field, adj_mtx)

        grid_field = np.where(mask_field != 0, mask_field, grid_field)
        err = np.sqrt(np.sum((grid_field - grid_field_pre) ** 2) / np.sum(grid_field**2))
        if itr >= max_itr:
            print("Max iteration reached, with L2 error at: " + str(err))
        itr = itr + 1

    print("Total iteration: " + str(itr))

    return grid_field


def polyfit_degs(

    adata: AnnData,
    genes: Optional[List[str]] = None,
    factors: str = "gradient",
    degree: int = 3,
    get_rid_of_zero: bool = False
) -> None:
    """Differential genes expression tests using polynomial regression.
    Args:
        adata (AnnData): Annotated data object containing gene expression data.
        genes (Optional[List[str]], optional): List of genes to perform differential expression tests on. If None, all genes in `adata` will be used. Defaults to None.
        factors (str, optional): Name of the column in `adata.obs` that contains the factors for regression analysis. Defaults to "gradient".
        degree (int, optional): Degree of the polynomial regression model. Defaults to 3.
        get_rid_of_zero (bool, optional): Flag indicating whether to exclude cells with zero expression values from the analysis. Defaults to False.
    Returns:
        deg_df (pd.DataFrame): DataFrame containing the results of the differential gene expression analysis.
            Columns:
                - gene: Gene names.
                - p_value: P-value of the F-test for the overall significance of the polynomial regression model.
                - rsquared_adj: Adjusted R-squared value of the polynomial regression model.
                - rsquared: R-squared value of the polynomial regression model.
                - degree_i: Coefficients of the polynomial regression model for each degree, where i ranges from 0 to the specified degree.
                - degree_i_pvalues: P-values of the coefficients of the polynomial regression model for each degree, where i ranges from 0 to the specified degree.
    Raises:
        ValueError: Raised if `genes` is provided but does not correspond to the columns in `adata`.
        Exception: Raised if factors from the model formula `fullModelFormulaStr` are invalid.
    """
    if genes is None:
        genes = adata.var.index.tolist()
    df_factors = adata.obs[[factors]].copy()
    df_factors['expression'] = 0
    deg_df = pd.DataFrame(genes, index = genes, columns =['gene'])
    deg_df['p_value'] = 0
    deg_df['rsquared_adj'] = 0
    deg_df['rsquared'] = 0
    deg_df[['degree_'+str(degree-i) for i in range(degree+1)]] = 0
    deg_df[['degree_'+str(degree-i)+'_pvalues' for i in range(degree+1)]] = 0
    for i in tqdm(genes):
        expression = adata[:, i].X.toarray().flatten()
        df_factors.loc[:,"expression"] = expression
        if get_rid_of_zero:
            df_factors_copy = df_factors.copy()
            df_factors_copy = df_factors_copy[df_factors_copy['expression']!=0]
            x = df_factors_copy[factors].values
            y = df_factors_copy['expression'].values
            
            X = np.vander(x.ravel(), degree + 1)
            model = sm.OLS(y, X)
            results = model.fit()
            deg_df.loc[i, 'p_value'] = results.f_pvalue
            deg_df.loc[i, ['degree_'+str(degree-i) for i in range(degree+1)]] = results.params
            deg_df.loc[i, ['degree_'+str(degree-i)+'_pvalues' for i in range(degree+1)]] = results.pvalues
            deg_df.loc[i, 'rsquared_adj'] = results.rsquared_adj
            deg_df.loc[i, 'rsquared'] = results.rsquared
            
        else:
            x = df_factors[factors].values
            y = df_factors['expression'].values
            X = np.vander(x.ravel(), degree + 1)
            model = sm.OLS(y, X)
            results = model.fit()
            deg_df.loc[i, 'p_value'] = results.f_pvalue
            deg_df.loc[i, ['degree_'+str(degree-i) for i in range(degree+1)]] = results.params
            deg_df.loc[i, ['degree_'+str(degree-i)+'_pvalues' for i in range(degree+1)]] = results.pvalues
            deg_df.loc[i, 'rsquared_adj'] = results.rsquared_adj
            deg_df.loc[i, 'rsquared'] = results.rsquared
        adata[:, i].X = df_factors['expression'].values

    return deg_df

def get_gradient_genes_matrix(adata, gradient_key, genes):
    '''
    Get the gene expression matrix for a given set of genes along a specified gradient.
    Parameters:
        adata (AnnData): Annotated data object containing gene expression data.
        gradient_key (str): Key of the gradient in the `adata.obs` dataframe.
        genes (list): List of genes to include in the gene expression matrix.
    Returns:
        cell_exp (pd.DataFrame): Gene expression matrix with cells as rows and genes as columns, sorted by the gradient values.
    '''
    gradient_genes_matrix = pd.DataFrame(index = adata.obs[gradient_key].tolist().copy())
    select_gene_matrix = adata[:,genes].X.toarray().copy()
    gradient_genes_matrix[genes] = select_gene_matrix
    gradient_genes_matrix = gradient_genes_matrix.sort_index()
    return gradient_genes_matrix

def get_gradient_genes_heatmap(gradient_genes_matrix, save_path = None):
    '''
    Generate a heatmap of the gradient genes based on the given cell expression data.
    Parameters:
    - cell_exp (numpy.ndarray): The cell expression data.
    - save_path (str, optional): The file path to save the heatmap image. If not provided, the heatmap will be displayed.
    Returns:
    - pandas.DataFrame: The smoothed and normalized expression data used for the heatmap.
    Note:
    - This function requires the following packages: statsmodels, scipy, seaborn, matplotlib.pyplot.
    - The cell expression data should be a 2D numpy array, where each row represents a gene and each column represents a cell.
    - The function performs z-score normalization on the cell expression data.
    - The function applies a Savitzky-Golay filter to smooth the expression data.
    - The function generates a heatmap using seaborn and matplotlib.pyplot.
    - If save_path is provided, the heatmap image will be saved as a file. Otherwise, the heatmap will be displayed.
    - The colormap used for the heatmap is "bwr".
    - If the heatmap is saved, the image will have a DPI of 600 and tight bounding box.
    '''
    from scipy import stats
    from scipy.signal import savgol_filter
    import seaborn as sns
    import matplotlib.pyplot as plt
    gradient_genes_matrix = gradient_genes_matrix.copy()
    gradient_genes_matrix_z = stats.zscore(gradient_genes_matrix, axis=0)
    gradient_genes_matrix_z = gradient_genes_matrix_z.T
    smooth_length = 100
    last_pd_smooth = savgol_filter(gradient_genes_matrix_z, smooth_length, 1)
    last_pd_smooth = pd.DataFrame(last_pd_smooth)
    last_pd_smooth.columns = gradient_genes_matrix_z.columns
    last_pd_smooth.index = gradient_genes_matrix_z.index
    n_rows, n_cols = 10,last_pd_smooth.shape[0]/2
    if save_path:
        fig, ax = plt.subplots(figsize=(n_rows, n_cols)) 
        sns.heatmap(last_pd_smooth, cmap = "bwr", xticklabels = False, ax=ax)
        plt.savefig(save_path, bbox_inches = 'tight', dpi = 600)
        return last_pd_smooth
    else:
        fig, ax = plt.subplots(figsize=(n_rows, n_cols)) 
        sns.heatmap(last_pd_smooth, cmap = "bwr", xticklabels = False, ax=ax)
        plt.show()
        return last_pd_smooth