import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
from scipy import stats
from sklearn import preprocessing
import multiprocessing as mp
from pygam import LinearGAM, s, f
import statsmodels.stats as stat
from scipy.signal import savgol_filter
import statsmodels.formula.api as smf
import scipy.stats
from scipy import sparse
import gc
from anndata import AnnData
from matplotlib.axes import Axes




def GAM_gene_fit(exp_gene_list):

    """
    Parameters
    ----------
    exp_gene_list : multi layer list

    exp_gene_list[0]: dataframe
                    columns : ptime,gene_expression
    exp_gene_list[1]: gene_name

    """

    r_list = list()
    trend_list = list()
    gene_list = list()
    pvalue_list = list()

    df_new = exp_gene_list[0]
    gene = exp_gene_list[1]
    x = df_new[["ptime"]].values
    y = df_new[gene]
    gam = LinearGAM(s(0, n_splines=8))
    gam_fit = gam.gridsearch(x, y, progress=False)
    grid_X = gam_fit.generate_X_grid(term=0)
    r_list.append(gam_fit.statistics_["pseudo_r2"]["explained_deviance"])
    pvalue_list.append(gam_fit.statistics_["p_values"][0])
    gene_list.append(gene)

    trend_list.append(js_score(gam_fit, grid_X))

    ## sort gene by fdr and R2
    df_batch_res = pd.DataFrame(
        {
            "gene": gene_list,
            "pvalue": pvalue_list,
            "model_fit": r_list,
            "pattern": trend_list,
        }
    )
    return df_batch_res

def ptime_gene_GAM(adata: AnnData, core_number: int = 3) -> pd.DataFrame:
    """
    Fit GAM model by formula gene_exp ~ Ptime.

    Call GAM_gene_fit() by multi-process computing to improve operational speed.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    core_number
        Number of processes for caculating.

    Returns
    -------
    :class:`~pandas.DataFrame`
        An :class:`~pandas.DataFrame` object, each column is one index.

        - pvalue: calculated from GAM
        - R2: a goodness-of-fit measure. larger value means better fit
        - pattern: increase or decrease. drection of gene expression changes across time
        - fdr: BH fdr

    """
    # perform GAM model on each gene
    gene_list_for_gam = adata.uns["gene_list_lm"]

    df_exp_filter = pd.DataFrame(data=adata.X,
                                 index=adata.obs.index,
                                 columns=adata.var.index)

    print("Genes number fitted by GAM model:  ", len(gene_list_for_gam))
    if core_number >= 1:
        para_list = list()
        for gene in gene_list_for_gam:
            df_new = pd.DataFrame({
                "ptime": list(adata.obs["ptime"]),
                gene: list(df_exp_filter[gene])
            })
            # df_new=df_new.loc[df_new[gene]>0]
            para_list.append((df_new, gene))
        p = mp.Pool(core_number)
        df_res = p.map(GAM_gene_fit, para_list)
        p.close()
        p.join()
        df_res = pd.concat(df_res)

        del para_list
        gc.collect()
    fdr = stat.multitest.fdrcorrection(np.array(df_res["pvalue"]))[1]
    df_res["fdr"] = fdr
    df_res.index = list(df_res["gene"])
    return df_res