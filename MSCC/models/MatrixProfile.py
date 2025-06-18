import stumpy
import numpy as np
from typing import Callable, Optional
class MatrixProfile():
    """
    Wrapper of the stympy implementation of the MatrixProfile algorithm

    Parameters
    ----------
    window : int,
        target subsequence length.
    
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples - m,)
        The anomaly score.
        The higher, the more abnormal. Anomalies tend to have higher
        scores. This value is available once the detector is
        fitted.
    """

    # def __init__(self, window):
    #     self.window = window
    #     self.model_name = 'MatrixProfile'

    # def fit(self, X, y=None):
    #     """Fit detector. y is ignored in unsupervised methods.
        
    #     Parameters
    #     ----------
    #     X : numpy array of shape (n_samples, )
    #         The input samples.
    #     y : Ignored
    #         Not used, present for API consistency by convention.
        
    #     Returns
    #     -------
    #     self : object
    #         Fitted estimator.
    #     """
    #     self.profile = stumpy.stump(X.ravel(),m=self.window)
    #     #self.profile = mp.compute(X, windows=self.window)
    #     res = np.zeros(len(X))
    #     res.fill(self.profile[:, 0].min())
    #     res[self.window//2:-self.window//2+1] = self.profile[:, 0]
    #     self.decision_scores_ = res
    #     return self
    
    def __init__(
        self,
        window: int,
        agg_func: Callable[[np.ndarray], np.ndarray] = np.max,
        drop_nan: bool = True,
    ):
        self.window = 64
        self.agg_func = agg_func
        self.drop_nan = drop_nan

    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        print("Using MatrixProfile algorithm for anomaly detection.")
        # 1️⃣ 基础检查
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (time × features 或 features × time)")

        # 2️⃣ **始终** 构造 (d, n) 形状给 stumpy
        if X.shape[0] >= X.shape[1]:
            # 行数多 ⇒ 行是时间 ⇒ 需要转置
            T = X.T
        else:
            # 行本来就是维度
            T = X

        d, n = T.shape
        if n < self.window:
            raise ValueError(f"window ({self.window}) larger than series length ({n})")

        # 3️⃣ 计算 multivariate matrix profile
        P, _ = stumpy.mstump(T, m=self.window)

        # 4️⃣ 聚合 + 边缘填充（同之前）
        profile_1d = self.agg_func(P, axis=0)
        if self.drop_nan:
            mask = ~np.isfinite(profile_1d)
            if mask.any():
                finite_vals = profile_1d[~mask]
                fill_val = finite_vals.min() if finite_vals.size else 0.0
                profile_1d[mask] = fill_val

        res = np.full(n, profile_1d.min(), dtype=float)
        start = self.window // 2
        end = n - self.window // 2 + 1
        res[start:end] = profile_1d

        self.decision_scores_ = res
        self.profile_per_dim_ = P
        return self

