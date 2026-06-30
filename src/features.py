"""数据特征工程模块 — 添加时间周期性、滞后和滚动统计特征。

提供 add_features() 函数，为时序数据添加 sin/cos 时间编码（避免边界跳变）、
滞后特征（上周同日、上月同日）和 7/30 天滚动均值。
"""
import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加时间周期性、滞后和滚动统计特征。

    输入 DataFrame 需包含 'Date' 和 'Global_active_power' 列。
    对输入不做修改，返回新 DataFrame。

    Args:
        df: 原始 DataFrame，包含 'Date' 列和其他特征列

    Returns:
        添加了以下 11 列新特征的新 DataFrame（NaN 行已丢弃并从索引 0 开始）:
          - doy_sin / doy_cos: 一年中第几天的 sin/cos 编码（周期 365）
          - month_sin / month_cos: 月份的 sin/cos 编码（周期 12）
          - dow_sin / dow_cos: 星期几的 sin/cos 编码（周期 7）
          - is_weekend: 周末标记（0/1）
          - lag_7: Target 变量的 7 天滞后
          - lag_30: Target 变量的 30 天滞后
          - roll_mean_7: Target 变量的 7 天滚动均值
          - roll_mean_30: Target 变量的 30 天滚动均值
    """
    date = pd.to_datetime(df["Date"])

    out = df.copy()

    # --- 时间周期性特征 ---
    # 使用 sin/cos 编码避免数值边界跳变（如 12 月 31 日→1 月 1 日）
    out["doy_sin"] = np.sin(2 * np.pi * date.dt.dayofyear / 365)
    out["doy_cos"] = np.cos(2 * np.pi * date.dt.dayofyear / 365)
    out["month_sin"] = np.sin(2 * np.pi * date.dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * date.dt.month / 12)
    out["dow_sin"] = np.sin(2 * np.pi * date.dt.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * date.dt.dayofweek / 7)
    out["is_weekend"] = (date.dt.dayofweek >= 5).astype(float)

    # --- 滞后特征 ---
    # 仅在 target 变量上做滞后（Global_active_power）
    out["lag_7"] = out["Global_active_power"].shift(7)
    out["lag_30"] = out["Global_active_power"].shift(30)

    # --- 滚动统计特征 ---
    out["roll_mean_7"] = out["Global_active_power"].rolling(7).mean()
    out["roll_mean_30"] = out["Global_active_power"].rolling(30).mean()

    # 丢弃由 shift/rolling 产生的 NaN 行，重置索引
    return out.dropna().reset_index(drop=True)
