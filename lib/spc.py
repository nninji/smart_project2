"""
SPC 관리도 (Statistical Process Control)
강의록 09_통계적공정관리.pdf

계량형 5종: X̄-R, X̄-s, I-MR, CUSUM, EWMA
계수형 4종: NP, P, C, U
"""
import numpy as np
import pandas as pd
from .constants import unbiased_const


# ────────────────────────────────────────────────────────────────────
# 계량형 관리도 (강의록 09 page 5 공식표)
# ────────────────────────────────────────────────────────────────────

def xbar_r_chart(df, sg_col, var_col):
    """X̄-R 관리도 (강의록 09 page 5, n=2~10)"""
    grouped = df.groupby(sg_col)[var_col]
    sg = pd.DataFrame({
        'Xbar': grouped.mean(),
        'R':    grouped.max() - grouped.min(),
        'n':    grouped.count(),
    })
    X_bar_bar = float(sg['Xbar'].mean())
    R_bar = float(sg['R'].mean())
    n_mode = int(sg['n'].mode().iloc[0])

    A2 = unbiased_const('A2', n_mode)
    D3 = unbiased_const('D3', n_mode)
    D4 = unbiased_const('D4', n_mode)
    d2 = unbiased_const('d2', n_mode)

    # σ_within (강의록 page 6의 R̄/d2 사용)
    sigma_within = R_bar / d2

    xbar_chart = pd.DataFrame({
        'sg':    sg.index.astype(str),
        'point': sg['Xbar'].values,
        'CL':    X_bar_bar,
        'UCL':   X_bar_bar + A2 * R_bar,
        'LCL':   X_bar_bar - A2 * R_bar,
    })
    r_chart = pd.DataFrame({
        'sg':    sg.index.astype(str),
        'point': sg['R'].values,
        'CL':    R_bar,
        'UCL':   D4 * R_bar,
        'LCL':   D3 * R_bar,
    })
    return {
        'type': 'X̄-R',
        'primary': xbar_chart,
        'secondary': r_chart,
        'primaryLabel': 'X̄',
        'secondaryLabel': 'R',
        'sigmaWithin': sigma_within,
        'n_mode': n_mode,
        'constants': {'A2': A2, 'D3': D3, 'D4': D4, 'd2': d2},
    }


def xbar_s_chart(df, sg_col, var_col):
    """X̄-s 관리도 (강의록 09 page 5, n≥10)"""
    grouped = df.groupby(sg_col)[var_col]
    sg = pd.DataFrame({
        'Xbar': grouped.mean(),
        's':    grouped.std(ddof=1),
        'n':    grouped.count(),
    })
    X_bar_bar = float(sg['Xbar'].mean())
    s_bar = float(sg['s'].mean())
    n_mode = int(sg['n'].mode().iloc[0])

    A3 = unbiased_const('A3', n_mode)
    B3 = unbiased_const('B3', n_mode)
    B4 = unbiased_const('B4', n_mode)
    c4 = unbiased_const('c4', n_mode)

    sigma_within = s_bar / c4

    xbar_chart = pd.DataFrame({
        'sg':    sg.index.astype(str),
        'point': sg['Xbar'].values,
        'CL':    X_bar_bar,
        'UCL':   X_bar_bar + A3 * s_bar,
        'LCL':   X_bar_bar - A3 * s_bar,
    })
    s_chart = pd.DataFrame({
        'sg':    sg.index.astype(str),
        'point': sg['s'].values,
        'CL':    s_bar,
        'UCL':   B4 * s_bar,
        'LCL':   B3 * s_bar,
    })
    return {
        'type': 'X̄-s',
        'primary': xbar_chart,
        'secondary': s_chart,
        'primaryLabel': 'X̄',
        'secondaryLabel': 's',
        'sigmaWithin': sigma_within,
        'n_mode': n_mode,
        'constants': {'A3': A3, 'B3': B3, 'B4': B4, 'c4': c4},
    }


def imr_chart(df, sg_col, var_col, window=2):
    """I-MR 관리도 (강의록 09 page 5-11, n=1)"""
    # 순서 보존
    df_sorted = df.sort_values(sg_col).reset_index(drop=True)
    values = df_sorted[var_col].values
    sgs = df_sorted[sg_col].astype(str).values
    X_bar = float(np.mean(values))

    # 이동범위 (rolling window)
    mr = np.full(len(values), np.nan)
    for i in range(window-1, len(values)):
        seg = values[i-window+1:i+1]
        mr[i] = float(np.max(seg) - np.min(seg))
    mr_valid = mr[~np.isnan(mr)]
    MR_bar = float(np.mean(mr_valid)) if len(mr_valid) > 0 else 0

    d2 = unbiased_const('d2', window)
    D3 = unbiased_const('D3', window)
    D4 = unbiased_const('D4', window)
    sigma_within = MR_bar / d2

    i_chart = pd.DataFrame({
        'sg':    sgs,
        'point': values,
        'CL':    X_bar,
        'UCL':   X_bar + 3 * MR_bar / d2,
        'LCL':   X_bar - 3 * MR_bar / d2,
    })
    mr_chart = pd.DataFrame({
        'sg':    sgs,
        'point': mr,
        'CL':    MR_bar,
        'UCL':   D4 * MR_bar,
        'LCL':   D3 * MR_bar,
    })
    return {
        'type': 'I-MR',
        'primary': i_chart,
        'secondary': mr_chart,
        'primaryLabel': 'I',
        'secondaryLabel': 'MR',
        'sigmaWithin': sigma_within,
        'window': window,
        'constants': {'d2': d2, 'D3': D3, 'D4': D4},
    }


def cusum_chart(df, sg_col, var_col, mu0=None, K_factor=0.5, H_factor=5):
    """
    CUSUM 누적합 관리도 (강의록 09 page 2)
    - K = 0.5σ (참조값)
    - H = 5σ (결정구간)
    """
    df_sorted = df.sort_values(sg_col).reset_index(drop=True)
    values = df_sorted[var_col].values
    sgs = df_sorted[sg_col].astype(str).values

    if mu0 is None:
        mu0 = float(np.mean(values))

    # σ 추정: 이동범위 평균/d2(2)
    mr = np.abs(np.diff(values))
    sigma = float(np.mean(mr) / unbiased_const('d2', 2)) if len(mr) > 0 else 1.0
    K = K_factor * sigma
    H = H_factor * sigma

    s_plus = np.zeros(len(values))
    s_minus = np.zeros(len(values))
    for i, v in enumerate(values):
        prev_p = s_plus[i-1] if i > 0 else 0
        prev_m = s_minus[i-1] if i > 0 else 0
        s_plus[i] = max(0, prev_p + (v - mu0) - K)
        s_minus[i] = max(0, prev_m - (v - mu0) - K)

    return {
        'type': 'CUSUM',
        'primary': pd.DataFrame({
            'sg':    sgs,
            'point': s_plus,
            'pointNeg': -s_minus,
            'CL':    0,
            'UCL':   H,
            'LCL':   -H,
        }),
        'primaryLabel': 'CUSUM',
        'mu0': mu0, 'sigma': sigma, 'K': K, 'H': H,
    }


def ewma_chart(df, sg_col, var_col, mu0=None, lam=0.2, L=3):
    """EWMA 지수가중이동평균 관리도 (강의록 09 page 2)"""
    df_sorted = df.sort_values(sg_col).reset_index(drop=True)
    values = df_sorted[var_col].values
    sgs = df_sorted[sg_col].astype(str).values

    if mu0 is None:
        mu0 = float(np.mean(values))

    mr = np.abs(np.diff(values))
    sigma = float(np.mean(mr) / unbiased_const('d2', 2)) if len(mr) > 0 else 1.0

    z = np.zeros(len(values))
    z[0] = lam * values[0] + (1 - lam) * mu0
    for i in range(1, len(values)):
        z[i] = lam * values[i] + (1 - lam) * z[i-1]

    ucl = np.zeros(len(values))
    lcl = np.zeros(len(values))
    for i in range(len(values)):
        factor = np.sqrt((lam / (2 - lam)) * (1 - (1 - lam)**(2 * (i + 1))))
        margin = L * sigma * factor
        ucl[i] = mu0 + margin
        lcl[i] = mu0 - margin

    return {
        'type': 'EWMA',
        'primary': pd.DataFrame({
            'sg':    sgs,
            'point': z,
            'CL':    mu0,
            'UCL':   ucl,
            'LCL':   lcl,
        }),
        'primaryLabel': 'EWMA',
        'mu0': mu0, 'sigma': sigma, 'lambda': lam, 'L': L,
    }


# ────────────────────────────────────────────────────────────────────
# 계수형 관리도 (강의록 09 page 5)
# ────────────────────────────────────────────────────────────────────

def np_chart(df, sg_col, size_col, defect_col):
    """NP 관리도 (불량개수, n 동일, 강의록 09 page 12)"""
    df_s = df.sort_values(sg_col).reset_index(drop=True)
    total_d = float(df_s[defect_col].sum())
    k = len(df_s)
    np_bar = total_d / k
    p_bar = total_d / float(df_s[size_col].sum())
    se = np.sqrt(np_bar * (1 - p_bar))

    return {
        'type': 'NP',
        'primary': pd.DataFrame({
            'sg':    df_s[sg_col].astype(str).values,
            'point': df_s[defect_col].values,
            'CL':    np_bar,
            'UCL':   np_bar + 3 * se,
            'LCL':   max(0, np_bar - 3 * se),
        }),
        'primaryLabel': 'np',
        'p_bar': p_bar,
    }


def p_chart(df, sg_col, size_col, defect_col):
    """P 관리도 (불량률, n 가변, 강의록 09 page 14)"""
    df_s = df.sort_values(sg_col).reset_index(drop=True)
    p_bar = float(df_s[defect_col].sum()) / float(df_s[size_col].sum())

    n_arr = df_s[size_col].values
    se = np.sqrt(p_bar * (1 - p_bar) / n_arr)

    return {
        'type': 'P',
        'primary': pd.DataFrame({
            'sg':    df_s[sg_col].astype(str).values,
            'point': df_s[defect_col].values / n_arr,
            'CL':    p_bar,
            'UCL':   p_bar + 3 * se,
            'LCL':   np.maximum(0, p_bar - 3 * se),
            'n_i':   n_arr,
        }),
        'primaryLabel': 'p',
        'p_bar': p_bar,
    }


def c_chart(df, sg_col, defect_col):
    """C 관리도 (결점수, n 동일, 강의록 09 page 14)"""
    df_s = df.sort_values(sg_col).reset_index(drop=True)
    c_bar = float(df_s[defect_col].mean())
    se = np.sqrt(c_bar)

    return {
        'type': 'C',
        'primary': pd.DataFrame({
            'sg':    df_s[sg_col].astype(str).values,
            'point': df_s[defect_col].values,
            'CL':    c_bar,
            'UCL':   c_bar + 3 * se,
            'LCL':   max(0, c_bar - 3 * se),
        }),
        'primaryLabel': 'c',
    }


def u_chart(df, sg_col, size_col, defect_col):
    """U 관리도 (단위당 결점수, n 가변, 강의록 09 page 14)"""
    df_s = df.sort_values(sg_col).reset_index(drop=True)
    u_bar = float(df_s[defect_col].sum()) / float(df_s[size_col].sum())
    n_arr = df_s[size_col].values
    se = np.sqrt(u_bar / n_arr)

    return {
        'type': 'U',
        'primary': pd.DataFrame({
            'sg':    df_s[sg_col].astype(str).values,
            'point': df_s[defect_col].values / n_arr,
            'CL':    u_bar,
            'UCL':   u_bar + 3 * se,
            'LCL':   np.maximum(0, u_bar - 3 * se),
            'n_i':   n_arr,
        }),
        'primaryLabel': 'u',
    }


# ────────────────────────────────────────────────────────────────────
# Nelson's Rule 8 (강의록 09 page 4)
# ────────────────────────────────────────────────────────────────────

NELSON_RULES_TEXT = {
    1: '1개 이상의 관측치가 ±3σ를 벗어난 경우',
    2: '연속된 9개 이상의 관측치가 평균선의 같은 쪽에 존재',
    3: '6개 이상의 관측치가 연속적으로 증가하거나 감소(추세)',
    4: '연속된 14개 이상의 관측치가 번갈아 증감(진동)',
    5: '연속된 3개 중 2개 이상이 ±2σ를 동일한 방향으로 벗어남',
    6: '연속된 5개 중 4개 이상이 ±1σ를 동일한 방향으로 벗어남',
    7: '연속된 15개 관측치가 ±1σ 안에 존재(층화)',
    8: '연속된 8개 관측치가 ±1σ 밖에 양쪽으로 존재(혼합)',
}


def apply_nelson_rules(chart_df):
    """
    Nelson's Rule 8가지 적용 (강의록 09 page 4)

    Returns:
        chart_df에 'violations' 컬럼이 추가된 DataFrame (각 점의 위반 규칙 목록)
    """
    df = chart_df.copy().reset_index(drop=True)
    points = df['point'].values
    UCL = df['UCL'].values if 'UCL' in df.columns else None
    CL = df['CL'].values if 'CL' in df.columns else None
    LCL = df['LCL'].values if 'LCL' in df.columns else None

    if UCL is None or CL is None or LCL is None:
        df['violations'] = [[] for _ in range(len(df))]
        return df

    # ±1σ, ±2σ, ±3σ 영역 (CL은 스칼라이거나 배열일 수 있음)
    CL_s = float(np.mean(CL)) if hasattr(CL, '__len__') else float(CL)
    UCL_s = float(np.mean(UCL)) if hasattr(UCL, '__len__') else float(UCL)
    LCL_s = float(np.mean(LCL)) if hasattr(LCL, '__len__') else float(LCL)
    sigma = (UCL_s - CL_s) / 3
    if sigma <= 0:
        df['violations'] = [[] for _ in range(len(df))]
        return df

    violations = [[] for _ in range(len(points))]
    for i, v in enumerate(points):
        if np.isnan(v):
            continue
        # Rule 1: ±3σ 벗어남
        if v > UCL_s or v < LCL_s:
            violations[i].append(1)

    # Rule 2: 9점 연속 한쪽
    for i in range(8, len(points)):
        seg = points[i-8:i+1]
        if all(s > CL_s for s in seg) or all(s < CL_s for s in seg):
            violations[i].append(2)

    # Rule 3: 6점 연속 증가/감소
    for i in range(5, len(points)):
        seg = points[i-5:i+1]
        if all(seg[j] < seg[j+1] for j in range(5)) or all(seg[j] > seg[j+1] for j in range(5)):
            violations[i].append(3)

    # Rule 4: 14점 교대
    for i in range(13, len(points)):
        seg = points[i-13:i+1]
        alt = all((seg[j] - CL_s) * (seg[j+1] - CL_s) < 0 for j in range(13))
        if alt:
            violations[i].append(4)

    # Rule 5: 3점 중 2점 2σ 외 동일 방향
    for i in range(2, len(points)):
        seg = points[i-2:i+1]
        above = sum(1 for s in seg if s > CL_s + 2*sigma)
        below = sum(1 for s in seg if s < CL_s - 2*sigma)
        if above >= 2 or below >= 2:
            violations[i].append(5)

    # Rule 6: 5점 중 4점 1σ 외 동일 방향
    for i in range(4, len(points)):
        seg = points[i-4:i+1]
        above = sum(1 for s in seg if s > CL_s + sigma)
        below = sum(1 for s in seg if s < CL_s - sigma)
        if above >= 4 or below >= 4:
            violations[i].append(6)

    # Rule 7: 15점 1σ 내
    for i in range(14, len(points)):
        seg = points[i-14:i+1]
        if all(abs(s - CL_s) < sigma for s in seg):
            violations[i].append(7)

    # Rule 8: 8점 1σ 밖 (양쪽)
    for i in range(7, len(points)):
        seg = points[i-7:i+1]
        if all(abs(s - CL_s) > sigma for s in seg):
            violations[i].append(8)

    df['violations'] = violations
    return df
