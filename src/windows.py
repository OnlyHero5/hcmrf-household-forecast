from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .features import add_features


@dataclass(frozen=True, slots=True)
class WindowSplit:
    input_len: int
    horizon: int
    train_starts: list[int]
    val_starts: list[int]
    test_starts: list[int]
    scaler_fit_end: int
    test_target_start: int


@dataclass(frozen=True, slots=True)
class PreparedWindows:
    values: np.ndarray
    dates: pd.Series
    scaler: MinMaxScaler
    split: WindowSplit


def load_daily_frame(data_dir: str) -> pd.DataFrame:
    data_path = Path(data_dir)
    daily_path = data_path / "daily.csv"
    if daily_path.exists():
        frame = pd.read_csv(daily_path)
    else:
        train = pd.read_csv(data_path / "train.csv")
        test = pd.read_csv(data_path / "test.csv")
        frame = pd.concat([train, test], ignore_index=True)

    frame = frame.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return frame


def make_window_split(
    total_days: int,
    input_len: int,
    horizon: int,
    step_size: int,
    development_end: int | None = None,
) -> WindowSplit:
    """构造带时间隔离的训练、验证和测试窗口。

    测试窗口固定为序列末端。验证窗口紧邻测试输入段之前；训练窗口的目标段
    必须在验证目标段开始前结束，因此训练标签与验证标签完全不重叠。
    """
    test_start = total_days - input_len - horizon
    test_target_start = total_days - horizon
    # 验证目标固定在官方训练段末端，避免把官方 test.csv 用于模型选择。
    if development_end is None:
        development_end = test_start
    val_start = development_end - input_len - horizon
    last_train_start = val_start - horizon

    if last_train_start < 0:
        raise ValueError(
            "时间序列不足以构造无标签重叠的 train/val/test 窗口："
            f"total_days={total_days}, input_len={input_len}, horizon={horizon}"
        )

    train_starts = list(range(0, last_train_start + 1, step_size))
    if not train_starts:
        raise ValueError("训练窗口为空，请缩短 input_len/horizon 或增加历史数据。")

    # scaler 只能使用验证目标开始之前已经观测到的数据。
    scaler_fit_end = val_start + input_len
    return WindowSplit(
        input_len=input_len,
        horizon=horizon,
        train_starts=train_starts,
        val_starts=[val_start],
        test_starts=[test_start],
        scaler_fit_end=scaler_fit_end,
        test_target_start=test_target_start,
    )


def prepare_windows(data_dir: str, input_len: int, horizon: int, step_size: int) -> PreparedWindows:
    frame = add_features(load_daily_frame(data_dir))
    test_dates = pd.read_csv(Path(data_dir) / "test.csv", usecols=["Date"])
    official_test_start = pd.to_datetime(test_dates["Date"]).min()
    frame_dates = pd.to_datetime(frame["Date"])
    development_end_candidates = np.flatnonzero(frame_dates.to_numpy() >= np.datetime64(official_test_start))
    if development_end_candidates.size == 0:
        raise ValueError("无法在完整日级序列中定位官方 test.csv 起始日期。")
    official_development_end = int(development_end_candidates[0])
    test_target_start = len(frame) - horizon
    # 365 天测试目标会向前跨入官方训练段；验证目标必须在测试目标开始前结束。
    development_end = min(official_development_end, test_target_start)
    split = make_window_split(
        len(frame), input_len, horizon, step_size, development_end=development_end
    )

    scaler = MinMaxScaler()
    feature_values = frame.iloc[:, 1:]
    scaler.fit(feature_values.iloc[: split.scaler_fit_end])
    values = scaler.transform(feature_values)
    dates = pd.to_datetime(frame["Date"])
    return PreparedWindows(values=values, dates=dates, scaler=scaler, split=split)
