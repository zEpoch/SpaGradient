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
    X_data: Optional[np.ndarray] = None,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
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
        X_data: The user supplied data that will be used for differential expression analysis directly. Defaults to
            None.
        genes: The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is
            used. Defaults to None.
        layer: The layer that will be used to retrieve data for dimension reduction and clustering. If `None`, .X is
            used. Defaults to None.
        fullModelFormulaStr: A formula string specifying the full model in differential expression tests (i.e.
            likelihood ratio tests) for each gene/feature. Defaults to "~cr(integral_time, df=3)".
        reducedModelFormulaStr: A formula string specifying the reduced model in differential expression tests (i.e.
            likelihood ratio tests) for each gene/feature. Defaults to "~1".
        family: The distribution family used for the expression responses in statsmodels. Currently, always uses `NB2`
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
    sparse = issparse(adata.X)
    deg_df = pd.DataFrame(index=genes, columns=["status", "family", "pval"])
    for i, gene in tqdm(enumerate(genes),  "Detecting gradient dependent genes via Generalized Additive Models (GAMs)",):
        expression = adata[:, gene].X.toarray().flatten() if sparse else adata[:, gene].flatten()
        df_factors.loc[:,"expression"] = expression
        result = diff_test_helper(df_factors, '~grid_field', '~1')
        deg_df.loc[gene, ["status", "family", "pval"]] = result
    deg_df["qval"] = multipletests(deg_df["pval"], method="fdr_bh")[1]
    return deg_df