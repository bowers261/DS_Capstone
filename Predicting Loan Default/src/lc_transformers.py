"""Custom scikit-learn transformers for the LendingClub credit-funnel project.

These live in a module (not inside a notebook) so that a pipeline pickled in
Notebook 02 can be reloaded in Notebook 03 or a deployment script: joblib stores
the class by its import path (`lc_transformers.CreditFeatureEngineer`), which any
process that can import this module will resolve. Classes defined in a notebook
cell pickle as `__main__.<Class>` and fail to load elsewhere.

Import in a notebook with:
    import sys; sys.path.insert(0, "src")
    from lc_transformers import (CreditFeatureEngineer, RareCategoryGrouper,
                                 RiskMissingFlag)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Shared feature-engineering column groups (imported by Notebooks 02 and 03 so
# both use identical definitions — edit here, not in a notebook cell).
# All are applied only if the column is present, so unknown names are no-ops.
# ---------------------------------------------------------------------------
LOG_COLS = [
    "annual_inc", "loan_amnt", "revol_bal", "tot_cur_bal", "total_bal_il",
    "total_bal_ex_mort", "tot_hi_cred_lim", "avg_cur_bal", "bc_open_to_buy",
]
MTHS_SINCE_COLS = [
    "mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
]
INSTALLMENT_BLOCK = [   # ~38% missing block -> add _missing flags, keep numeric
    "open_il_12m", "open_il_24m", "il_util", "all_util", "total_bal_il",
    "open_rv_12m", "open_rv_24m", "max_bal_bc", "open_acc_6m", "inq_last_12m",
    "total_cu_tl", "inq_fi", "mths_since_rcnt_il",
]
EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Application-time feature engineering for LendingClub loans.

    All operations are guarded by column presence, so the transformer is robust
    to the exact whitelist produced upstream. Nothing here learns from the target
    and no statistic crosses from test to train. Returns a pandas DataFrame.

    Parameters
    ----------
    mths_cap : int
        Cap (and NaN sentinel) for the ``mths_since_*`` fields, where NaN means
        the event never occurred and is treated as "a very long time ago".
    log_cols : list[str]
        Skewed monetary columns to ``log1p``-transform (raw column dropped).
    mths_cols : list[str]
        ``mths_since_*`` columns to convert to an ``_ever`` flag + capped numeric.
    installment_block : list[str]
        ~38%-missing installment/revolving columns to add ``_missing`` flags for.
    emp_length_map : dict
        Mapping from ``emp_length`` labels to an ordinal 0-10.
    """

    def __init__(self, mths_cap=180, log_cols=None, mths_cols=None,
                 installment_block=None, emp_length_map=None):
        self.mths_cap = mths_cap
        self.log_cols = log_cols
        self.mths_cols = mths_cols
        self.installment_block = installment_block
        self.emp_length_map = emp_length_map

    def fit(self, X, y=None):
        # Stateless w.r.t. data distribution; store config resolved to lists/dicts.
        self.log_cols_ = self.log_cols or []
        self.mths_cols_ = self.mths_cols or []
        self.installment_block_ = self.installment_block or []
        self.emp_length_map_ = self.emp_length_map or {}
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()

        # 1) Credit-history length in months from earliest_cr_line -> issue_d.
        if {"earliest_cr_line", "issue_d"}.issubset(X.columns):
            ecl = pd.to_datetime(X["earliest_cr_line"], format="%b-%Y", errors="coerce")
            iss = pd.to_datetime(X["issue_d"], format="%b-%Y", errors="coerce")
            X["credit_history_months"] = ((iss - ecl).dt.days / 30.44).clip(lower=0)
        X = X.drop(columns=[c for c in ["earliest_cr_line", "issue_d"] if c in X.columns])

        # 2) mths_since_* : NaN means the event never happened.
        for c in self.mths_cols_:
            if c in X.columns:
                X[c + "_ever"] = X[c].notna().astype("int8")
                X[c] = X[c].clip(upper=self.mths_cap).fillna(self.mths_cap)

        # 3) Employment length -> ordinal numeric.
        if "emp_length" in X.columns:
            X["emp_length_num"] = X["emp_length"].map(self.emp_length_map_)
            X = X.drop(columns=["emp_length"])

        # 4) Log-transform skewed monetary fields.
        for c in self.log_cols_:
            if c in X.columns:
                X[c + "_log"] = np.log1p(pd.to_numeric(X[c], errors="coerce").clip(lower=0))
                X = X.drop(columns=[c])

        # 5) Missing-block flags for the ~38%-missing installment/revolving block.
        for c in self.installment_block_:
            if c in X.columns:
                X[c + "_missing"] = X[c].isna().astype("int8")

        # Safety: never model on raw datetime columns.
        dt_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
        if dt_cols:
            X = X.drop(columns=dt_cols)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Keep the top-N levels per categorical column (learned on train); others -> 'Other'.

    Unseen levels at transform time also map to the ``other_label``.
    """

    def __init__(self, top_n=15, other_label="Other"):
        self.top_n = top_n
        self.other_label = other_label

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.top_levels_ = {}
        for c in self.columns_:
            vc = X[c].astype("object").value_counts()
            self.top_levels_[c] = set(vc.head(self.top_n).index)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.columns_:
            keep = self.top_levels_.get(c, set())
            X[c] = X[c].astype("object").where(X[c].isin(keep), self.other_label)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features if input_features is not None else self.columns_)


class RiskMissingFlag(BaseEstimator, TransformerMixin):
    """Add a ``credit_missing`` indicator for the policy model's harmonized credit
    score (LendingClub ``Risk_Score`` is ~67% missing in the rejected file, and its
    *absence* may itself be informative about the historical approval decision).
    """

    def __init__(self, add_flag=True, credit_col="credit"):
        self.add_flag = add_flag
        self.credit_col = credit_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.add_flag and self.credit_col in X.columns:
            X[self.credit_col + "_missing"] = X[self.credit_col].isna().astype("int8")
        return X
