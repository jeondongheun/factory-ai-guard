"""
모드별 z-score 계산 및 최적 임계값 분석
운영 모드별로 baseline 분리 후 z-score 계산
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ── 경로 설정 ────────────────────────────────────────────────────
SKAB_DIR = '/Users/mac/Desktop/Code/factory-ai-guard/backend/ml/data/SKAB/data'
SAVE_DIR = '/Users/mac/Desktop/Code/factory-ai-guard/backend/ml/data/SKAB/img/zscore_mode'
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

TARGET_FEATURES = ['Accelerometer1RMS', 'Temperature', 'Volume Flow RateRMS']

# ── 운영 모드 판별 함수 ──────────────────────────────────────────
def get_mode(flow):
    if flow > 100:
        return 'high_flow'
    elif flow > 50:
        return 'mid_flow'
    else:
        return 'low_flow'

MODE_LABELS = {
    'high_flow': '고유량 (>100)',
    'mid_flow':  '중유량 (50~100)',
    'low_flow':  '저유량 (<50)'
}

# ================================================================
# 1. 데이터 로딩 및 전처리
# ================================================================
print('[ 1 ] 데이터 로딩...')

all_files = []
for root, dirs, files in os.walk(SKAB_DIR):
    for file in files:
        if file.endswith('.csv'):
            all_files.append(os.path.join(root, file))
all_files.sort()

def current_voltage_correction(data):
    data = data.copy()
    mask = (data['Current'] > 100) & (data['Voltage'] < 100)
    idx = mask[mask].index
    voltage = data['Current'][mask]
    current = data['Voltage'][mask]
    data.loc[idx, 'Current'] = current
    data.loc[idx, 'Voltage'] = voltage
    return data

def filter_data(data):
    mask = ((data['Accelerometer1RMS'] > 0) &
            (data['Accelerometer2RMS'] > 0) &
            (data['Volume_Flow_RateRMS'] > 0) &
            (data['Current'] < 100) &
            (data['Voltage'] > 100))
    return data[mask]

dfs = []
for f in all_files:
    df = pd.read_csv(f, sep=';', index_col='datetime', parse_dates=True)
    df.columns = df.columns.str.replace(' ', '_')
    df = current_voltage_correction(df)
    df = filter_data(df)
    dfs.append(df)

alldata = pd.concat(dfs).sort_index()

# anomaly 컬럼 없는 행 제거 (anomaly-free는 이상 없음으로 처리)
if 'anomaly' not in alldata.columns:
    alldata['anomaly'] = 0
alldata['anomaly'] = alldata['anomaly'].fillna(0)

# 운영 모드 컬럼 추가
alldata['mode'] = alldata['Volume_Flow_RateRMS'].apply(get_mode)

print(f'  전체 데이터: {len(alldata)}행')
print(f'  모드별 분포:')
print(alldata['mode'].value_counts().to_string())


# ================================================================
# 2. 모드별 baseline 계산 (정상 데이터만 사용)
# ================================================================
print('\n[ 2 ] 모드별 baseline 계산...')

modes = ['high_flow', 'mid_flow', 'low_flow']
baselines = {}  # {mode: {feature: {mean, std}}}

for mode in modes:
    mode_normal = alldata[(alldata['mode'] == mode) & (alldata['anomaly'] == 0)]
    baselines[mode] = {}
    for feat in TARGET_FEATURES:
        col = feat.replace(' ', '_')
        if col in mode_normal.columns:
            mu  = mode_normal[col].mean()
            std = mode_normal[col].std()
            baselines[mode][feat] = {'mean': mu, 'std': std}
            print(f'  [{MODE_LABELS[mode]}] {feat}: mean={mu:.4f}, std={std:.4f}')

print()


# ================================================================
# 3. 모드별 z-score 계산
# ================================================================
print('[ 3 ] 모드별 z-score 계산...')

for feat in TARGET_FEATURES:
    col = feat.replace(' ', '_')
    z_col = f'z_{col}'
    alldata[z_col] = np.nan

    for mode in modes:
        mask = alldata['mode'] == mode
        mu   = baselines[mode][feat]['mean']
        std  = baselines[mode][feat]['std']
        if std > 0:
            alldata.loc[mask, z_col] = (
                (alldata.loc[mask, col] - mu) / std
            ).abs()

print('  완료')
z_cols = [f"z_{f.replace(' ', '_')}" for f in TARGET_FEATURES]
print(f'  z-score 컬럼: {z_cols}')


# ================================================================
# 4. z-score 분포 비교 (변수별 1장, 모드 3개)
# ================================================================
print('\n[ 4 ] z-score 분포 시각화...')

for feat in TARGET_FEATURES:
    z_col = f'z_{feat.replace(" ", "_")}'
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{feat} - 모드별 z-score 분포 (정상 vs 이상)',
                 fontsize=13, fontweight='bold')

    for col_idx, mode in enumerate(modes):
        ax = axes[col_idx]
        mode_data = alldata[alldata['mode'] == mode]
        normal = mode_data[mode_data['anomaly'] == 0][z_col].dropna()
        anomal = mode_data[mode_data['anomaly'] == 1][z_col].dropna()

        p95 = normal.quantile(0.95)
        p99 = normal.quantile(0.99)

        ax.hist(normal, bins=50, density=True, alpha=0.5,
                color='#3498DB', label=f'정상 (n={len(normal):,})')
        ax.hist(anomal, bins=50, density=True, alpha=0.5,
                color='#E74C3C', label=f'이상 (n={len(anomal):,})')
        ax.axvline(p95, color='blue', linestyle='--', linewidth=1.5,
                   label=f'정상 95%: {p95:.2f}')
        ax.axvline(p99, color='blue', linestyle=':', linewidth=1.5,
                   label=f'정상 99%: {p99:.2f}')
        ax.set_xlim(0, 10)
        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_xlabel('|z-score|', fontsize=10)
        ax.set_ylabel('밀도', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'01_zscore_dist_{feat.replace(" ", "_")}.png'
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  저장: {fname}')


# ================================================================
# 5. ROC Curve (변수별 1장, 모드 3개)
# ================================================================
print('\n[ 5 ] ROC Curve 분석...')

roc_results = {}

for feat in TARGET_FEATURES:
    z_col = f'z_{feat.replace(" ", "_")}'
    roc_results[feat] = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{feat} - ROC Curve (모드별 z-score)',
                 fontsize=13, fontweight='bold')

    for col_idx, mode in enumerate(modes):
        ax = axes[col_idx]
        mode_data = alldata[alldata['mode'] == mode].dropna(subset=[z_col, 'anomaly'])

        y_true  = mode_data['anomaly'].values
        y_score = mode_data[z_col].values

        if len(np.unique(y_true)) < 2:
            ax.text(0.5, 0.5, '데이터 부족', ha='center', va='center')
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        roc_results[feat][mode] = roc_auc

        ax.plot(fpr, tpr, color='#E74C3C', linewidth=2,
                label=f'AUC={roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_xlabel('FAR (오탐율)', fontsize=10)
        ax.set_ylabel('TPR (탐지율)', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'02_roc_{feat.replace(" ", "_")}.png'
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  저장: {fname}')

print('\n  AUC 결과:')
for feat in TARGET_FEATURES:
    for mode in modes:
        val = roc_results.get(feat, {}).get(mode, 'N/A')
        if val != 'N/A':
            print(f'    {feat} [{MODE_LABELS[mode]}]: {val:.3f}')
        else:
            print(f'    {feat} [{mode}]: N/A')


# ================================================================
# 6. 임계값별 F1/FAR/MAR (변수별 1장, 모드 3개)
# ================================================================
print('\n[ 6 ] 임계값별 F1/FAR/MAR 분석...')

optimal_thresholds = {}
thresholds = np.linspace(0, 5, 200)

for feat in TARGET_FEATURES:
    z_col = f'z_{feat.replace(" ", "_")}'
    optimal_thresholds[feat] = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{feat} - 임계값별 F1 / FAR / MAR (모드별 z-score)',
                 fontsize=13, fontweight='bold')

    for col_idx, mode in enumerate(modes):
        ax = axes[col_idx]
        mode_data = alldata[alldata['mode'] == mode].dropna(subset=[z_col, 'anomaly'])

        y_true  = mode_data['anomaly'].values
        y_score = mode_data[z_col].values

        f1s, fars, mars = [], [], []

        for th in thresholds:
            y_pred = (y_score >= th).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()

            far  = fp / (fp + tn) if (fp + tn) > 0 else 0
            mar  = fn / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            f1s.append(f1)
            fars.append(far)
            mars.append(mar)

        best_idx = np.argmax(f1s)
        best_th  = thresholds[best_idx]
        optimal_thresholds[feat][mode] = {
            'threshold': round(best_th, 3),
            'f1':        round(f1s[best_idx], 3),
            'far':       round(fars[best_idx], 3),
            'mar':       round(mars[best_idx], 3),
        }

        ax.plot(thresholds, f1s,  color='#2ECC71', linewidth=2, label='F1')
        ax.plot(thresholds, fars, color='#E74C3C', linewidth=2, label='FAR')
        ax.plot(thresholds, mars, color='#3498DB', linewidth=2, label='MAR')
        ax.axvline(best_th, color='black', linestyle='--', linewidth=1.5,
                   label=f'최적: {best_th:.2f}')
        ax.set_title(MODE_LABELS[mode], fontsize=11)
        ax.set_xlabel('z-score 임계값', fontsize=10)
        ax.set_ylabel('비율', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'03_threshold_{feat.replace(" ", "_")}.png'
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  저장: {fname}')


# ================================================================
# 7. 최종 결과 요약
# ================================================================
print('\n' + '=' * 60)
print('  최적 임계값 요약 (모드별)')
print('=' * 60)

for feat in TARGET_FEATURES:
    print(f'\n{feat}:')
    for mode in modes:
        r = optimal_thresholds[feat][mode]
        print(f'  [{MODE_LABELS[mode]}]')
        print(f'    임계값: {r["threshold"]} | F1: {r["f1"]} | FAR: {r["far"]} | MAR: {r["mar"]}')

print('\n' + '=' * 60)
print('  전체 대비 개선 여부 (이전 분석과 비교)')
print('=' * 60)
print('''
이전 (모드 혼합):
  Accelerometer1RMS: AUC 0.543 (랜덤 수준)
  Temperature:       AUC 0.506 (랜덤보다 못함)
  Volume Flow Rate:  AUC 0.798

모드 분리 후:
  → 위 그래프에서 확인
  → Temperature, Accelerometer1RMS AUC 개선 기대
''')

print(f'\n완료. 저장 경로: {SAVE_DIR}')