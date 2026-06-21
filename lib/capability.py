"""
공정능력 지수 계산 (Process Capability Index)
강의록 08_공정능력분석.pdf

σ_within 계산법 (강의록 page 6-7):
  A. n=1: 이동범위 MR̄/d2(w) 또는 median(MR)/d2(w)
  B(1). n≥2: 합동표준편차 sp/c4(d)  ← 기본 (강의록 PVC 예제 사용)
  B(2). n≥2: 부분군 범위 평균 R̄/d2(ni)
  B(3). n≥2: 부분군 표준편차 평균 s̄/c4(ni)
"""
import numpy as np
from scipy import stats as sp_stats
from .constants import unbiased_const, grade_capability


SIGMA_WITHIN_METHODS = {
    'pooled':    {'label': '합동표준편차 (sp / c4(d))',           'formula': 'σ_w = sp / c4(d)',           'since': '08 p.6 (1)'},
    'range':     {'label': '부분군 범위 평균 (R̄ / d2(n_i))',     'formula': 'σ_w = R̄ / d2(n_i)',        'since': '08 p.6 (2)'},
    'std':       {'label': '부분군 표준편차 평균 (s̄ / c4(n_i))', 'formula': 'σ_w = s̄ / c4(n_i)',        'since': '08 p.7 (3)'},
    'mr-mean':   {'label': '이동범위 평균 (MR̄ / d2(w))',          'formula': 'σ_w = MR̄ / d2(w)',          'since': '08 p.6 A'},
    'mr-median': {'label': '이동범위 중앙값 (median(MR) / d2(w))','formula': 'σ_w = median(MR) / d2(w)',  'since': '08 p.6 A'},
}


def _compute_sigma_within(sg_stats, all_values, method='pooled', window=2):
    """σ_within 계산 (3가지 방법 + n=1 모드 2가지)"""
    all_values = np.asarray(all_values, dtype=float)
    all_values = all_values[~np.isnan(all_values)]
    is_single = (sg_stats['n'] == 1).all()

    # n=1 또는 명시적 MR 방법
    if is_single or method in ('mr-mean', 'mr-median'):
        mr = np.abs(np.diff(all_values))
        if len(mr) == 0:
            return np.nan
        d2_w = unbiased_const('d2', window)
        if method == 'mr-median':
            return float(np.median(mr) / d2_w)
        return float(np.mean(mr) / d2_w)

    # n >= 2
    if method == 'range':
        # 가장 빈도 높은 부분군 크기
        mode_n = int(sg_stats['n'].mode().iloc[0])
        r_bar = float(sg_stats['range'].mean())
        return r_bar / unbiased_const('d2', mode_n)

    if method == 'std':
        mode_n = int(sg_stats['n'].mode().iloc[0])
        s_bar = float(sg_stats['std'].dropna().mean())
        return s_bar / unbiased_const('c4', mode_n)

    # 기본: pooled (강의록 PVC 예제와 일치)
    valid = sg_stats[sg_stats['n'] > 1].dropna(subset=['std'])
    if len(valid) == 0:
        return np.nan
    numer = ((valid['n'] - 1) * valid['std']**2).sum()
    denom = (valid['n'] - 1).sum()
    if denom == 0:
        return np.nan
    sp = float(np.sqrt(numer / denom))
    n_total = len(all_values)
    k = len(sg_stats)
    d = n_total - k + 1
    return sp / unbiased_const('c4', d)


def compute_capability(df, sg_col, var_col, LSL, USL, method='pooled', window=2):
    """
    공정능력 종합 계산

    Returns:
        dict with: xBar, sigmaWithin, sigmaOverall, Cp, Cpk, Pp, Ppk,
                  ppmWithin, ppmOverall, method, methodInfo, grade, gradeCpk, gradePpk
    """
    from .stats_lib import subgroup_stats

    if LSL is None or USL is None or LSL >= USL:
        return {'error': f'LSL({LSL})은 USL({USL})보다 작아야 합니다'}

    # 부분군 통계
    sg = subgroup_stats(df, sg_col, var_col)
    if len(sg) == 0:
        return {'error': '유효한 부분군이 없습니다'}

    all_values = pd.to_numeric(df[var_col], errors='coerce').dropna().values
    if len(all_values) < 2:
        return {'error': '관측치가 2개 이상 필요합니다'}

    x_bar = float(np.mean(all_values))
    n_total = len(all_values)
    k = len(sg)

    # 1) σ_overall (강의록 08 p.8: s/c4(n))
    s_hat = float(np.std(all_values, ddof=1))
    sigma_overall = s_hat / unbiased_const('c4', n_total)

    # 2) σ_within (3가지 방법 중 선택)
    sigma_within = _compute_sigma_within(sg, all_values, method, window)
    if np.isnan(sigma_within) or sigma_within <= 0:
        sigma_within = sigma_overall

    # 3) 공정능력지수
    Cp = (USL - LSL) / (6 * sigma_within)
    Cpk = float(min((USL - x_bar) / (3 * sigma_within), (x_bar - LSL) / (3 * sigma_within)))
    Pp = (USL - LSL) / (6 * sigma_overall)
    Ppk = float(min((USL - x_bar) / (3 * sigma_overall), (x_bar - LSL) / (3 * sigma_overall)))

    # 4) PPM 추정
    ppm_within = float((sp_stats.norm.cdf(LSL, x_bar, sigma_within) +
                       (1 - sp_stats.norm.cdf(USL, x_bar, sigma_within))) * 1e6)
    ppm_overall = float((sp_stats.norm.cdf(LSL, x_bar, sigma_overall) +
                        (1 - sp_stats.norm.cdf(USL, x_bar, sigma_overall))) * 1e6)

    return {
        'xBar': x_bar,
        'sigmaWithin': sigma_within,
        'sigmaOverall': sigma_overall,
        'Cp': float(Cp), 'Cpk': Cpk, 'Pp': float(Pp), 'Ppk': Ppk,
        'ppmWithin': ppm_within, 'ppmOverall': ppm_overall,
        'LSL': float(LSL), 'USL': float(USL),
        'n': n_total, 'k': k,
        'method': method,
        'methodInfo': SIGMA_WITHIN_METHODS[method],
        'grade': grade_capability(Cp),
        'gradeCpk': grade_capability(Cpk),
        'gradePpk': grade_capability(Ppk),
        'subgroupStats': sg,
    }


# pandas는 capability.py 안에서 사용되므로 명시적 import
import pandas as pd
