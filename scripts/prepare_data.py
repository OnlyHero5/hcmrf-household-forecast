"""数据处理脚本 — 将原始分钟级数据聚合为日级并融合天气数据。

处理流程：
  1. 加载原始分钟级 household_power_consumption.txt
  2. 按天聚合：有功功率/无功功率/分表 → 求和，电压/电流 → 平均
  3. 计算 sub_metering_remainder（剩余功率）
  4. 融合 ST QUENTIN 气象站月度天气数据（RR, NBJRR1/5/10, NBJBROU）
  5. 缺失值处理（线性插值 + 前向/后向填充）
  6. 按时间切分 train/test（2010-01-01 为界）
  7. 保存 data/processed/train.csv 和 test.csv
"""
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

print("=" * 60)
print("Step 1: 加载原始分钟级数据...")
print("=" * 60)

# 原始数据列名
cols = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
        'Sub_metering_3']

df = pd.read_csv('data/raw/household_power_consumption.txt', sep=';', na_values=['?'],
                 low_memory=False)
df.columns = cols

# 解析日期时间
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
df['Date_only'] = df['Datetime'].dt.date

# 转换数值列（处理 '?' 等非法值）
numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

print(f"Loaded {len(df):,} rows from {df['Date_only'].min()} to {df['Date_only'].max()}")

print("\n" + "=" * 60)
print("Step 2: 按天聚合...")
print("=" * 60)

# 按 PDF 要求聚合：
# global_active_power, global_reactive_power, sub_metering_1/2/3 → 按天求和
# voltage, global_intensity → 按天平均
daily_sum_cols = ['Global_active_power', 'Global_reactive_power',
                  'Sub_metering_1', 'Sub_metering_2']
daily_mean_cols = ['Voltage', 'Global_intensity']

daily_agg = {}
for c in daily_sum_cols:
    daily_agg[c] = 'sum'
for c in daily_mean_cols:
    daily_agg[c] = 'mean'
daily_agg['Sub_metering_3'] = 'sum'

daily = df.groupby('Date_only').agg(daily_agg).reset_index()
daily.rename(columns={'Date_only': 'Date'}, inplace=True)

# 计算剩余功率：remainder = (global_active_power * 1000 / 60) - (sub1 + sub2 + sub3)
daily['Sub_metering_remainder'] = (
    (daily['Global_active_power'] * 1000 / 60) -
    (daily['Sub_metering_1'] + daily['Sub_metering_2'] + daily['Sub_metering_3'])
)

# 统计每天的缺失值比例
na_counts = df.groupby('Date_only').apply(
    lambda x: pd.Series({'na_count': x[numeric_cols].isna().any(axis=1).sum(),
                         'total_count': len(x)})
).reset_index()
daily = daily.merge(na_counts, left_on='Date', right_on='Date_only', how='left')
daily.drop('Date_only', axis=1, inplace=True)

daily['na_ratio'] = daily['na_count'] / daily['total_count']

print(f"Aggregated to {len(daily)} daily records")
print(f"Columns: {list(daily.columns)}")
print(f"Date range: {daily['Date'].min()} to {daily['Date'].max()}")
print(f"\nMissing data summary:")
print(f"  Days with any NA: {(daily['na_count'] > 0).sum()}")
print(f"  Max NA ratio in a day: {daily['na_ratio'].max():.2%}")

print("\n" + "=" * 60)
print("Step 3: 融合天气数据...")
print("=" * 60)

# 加载天气数据（ST QUENTIN 气象站月度数据）
weather_file = 'data/weather/test_c4dc2289-2451-482c-a566-857ab34165a7.csv.gz'
print(f"Loading weather data from {weather_file}...")
weather_data = pd.read_csv(weather_file, sep=';', low_memory=False)

print(f"Weather data: {len(weather_data)} rows from {len(weather_data['NUM_POSTE'].unique())} stations")

# 筛选 ST QUENTIN 站（编号 02320001）
station_data = weather_data[weather_data['NUM_POSTE'] == '02320001'].copy()
if len(station_data) == 0:
    # 如果找不到 ST QUENTIN，找一个有 NBJBROU 数据的站
    print("ST QUENTIN not found, searching for suitable station...")
    has_brou = weather_data.dropna(subset=['NBJBROU'])
    station_ids = has_brou['NUM_POSTE'].value_counts().head(5)
    print(f"Top stations with NBJBROU data: {dict(station_ids)}")
    best_station = station_ids.index[0]
    station_data = weather_data[weather_data['NUM_POSTE'] == best_station].copy()
    print(f"Using station: {station_data['NOM_USUEL'].iloc[0]} ({best_station})")
else:
    print(f"Using ST QUENTIN station")

# 只保留需要的列
weather_cols = ['NUM_POSTE', 'NOM_USUEL', 'AAAAMM', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
station_data = station_data[weather_cols].copy()

for c in ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']:
    station_data[c] = pd.to_numeric(station_data[c], errors='coerce')

# 只保留 2006-2010 年的数据
station_data = station_data[
    station_data['AAAAMM'].astype(str).str.match(r'^(200[6-9]|2010)')
].reset_index(drop=True)

# 解析 AAAAMM 为年月
station_data['YearMonth'] = pd.to_datetime(
    station_data['AAAAMM'].astype(str) + '01', format='%Y%m%d'
)

# 在日级数据中创建年月列
daily['YearMonth'] = pd.to_datetime(daily['Date']).dt.to_period('M').dt.to_timestamp()

# 按年月合并天气数据
daily = daily.merge(station_data[['YearMonth', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']],
                     on='YearMonth', how='left')
daily.drop('YearMonth', axis=1, inplace=True)

print(f"Weather data merged! RR/ NBJRR1/5/10 non-null: {daily[['RR','NBJRR1','NBJRR5','NBJRR10']].notna().all(axis=1).sum()}")
print(f"NBJBROU non-null: {daily['NBJBROU'].notna().sum()}")

print("\n" + "=" * 60)
print("Step 4: 处理缺失值...")
print("=" * 60)

print(f"Missing values before fill:\n{daily.isna().sum()}")

# 按日期排序
daily = daily.sort_values('Date').reset_index(drop=True)
# 线性插值（最多连续补 5 个）
daily[numeric_cols] = daily[numeric_cols].interpolate(method='linear', limit=5)
# 前向/后向填充剩余缺失值
daily[numeric_cols] = daily[numeric_cols].ffill().bfill()
# 天气列也用前向/后向填充
daily[['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']] = \
    daily[['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']].ffill().bfill()
# 丢弃关键列仍有 NaN 的行
daily = daily.dropna(subset=['Global_active_power', 'Voltage']).reset_index(drop=True)

print(f"Missing values after fill:\n{daily.isna().sum()}")
print(f"Final daily records: {len(daily)}")

print("\n" + "=" * 60)
print("Step 5: 切分训练集/测试集...")
print("=" * 60)

daily = daily.sort_values('Date').reset_index(drop=True)

# 时序切分：2010-01-01 之前为训练集，之后为测试集
split_date = pd.Timestamp('2010-01-01')

daily['Date_parsed'] = pd.to_datetime(daily['Date'])
train = daily[daily['Date_parsed'] < split_date].copy()
test = daily[daily['Date_parsed'] >= split_date].copy()

# 如果训练集数据不足（< 365 天），改用 80/20 切分
if len(train) < 365:
    split_idx = int(len(daily) * 0.8)
    train = daily.iloc[:split_idx].copy()
    test = daily.iloc[split_idx:].copy()

print(f"Train: {len(train)} days ({train['Date_parsed'].min().date()} to {train['Date_parsed'].max().date()})")
print(f"Test:  {len(test)} days ({test['Date_parsed'].min().date()} to {test['Date_parsed'].max().date()})")

# 输出列定义
output_cols = ['Date',
               'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
               'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_remainder',
               'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

# 确保所有列都存在
for c in output_cols:
    if c not in train.columns:
        train[c] = np.nan
    if c not in test.columns:
        test[c] = np.nan

# 保存
os.makedirs('data/processed', exist_ok=True)
train[output_cols].to_csv('data/processed/train.csv', index=False)
test[output_cols].to_csv('data/processed/test.csv', index=False)

print(f"\nSaved train.csv ({len(train)} rows) and test.csv ({len(test)} rows)")
print(f"Columns: {output_cols}")

print("\n" + "=" * 60)
print("数据汇总")
print("=" * 60)
print(f"\nTrain head:")
print(train[output_cols].head())
print(f"\nTrain stats:")
print(train[output_cols].describe())
print(f"\nTest head:")
print(test[output_cols].head())
print(f"\nTest stats:")
print(test[output_cols].describe())

print(f"\nMissing values in train output: {train[output_cols].isna().sum().sum()}")
print(f"Missing values in test output: {test[output_cols].isna().sum().sum()}")
