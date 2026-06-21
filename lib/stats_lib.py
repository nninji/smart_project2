"""
통계 함수
강의록 08_공정능력분석.pdf 기반
"""
import numpy as np
import pandas as pd
from scipy import stats


def subgroup_stats(df, sg_col, var_col):
    """부분군 통계 계산"""
    # NaN/null 제거
    clean = df[[sg_col, var_col]].copy()
    clean[var_col] = pd.to_numeric(clean[var_col], errors='coerce')
    clean = clean.dropna(subset=[var_col])

    grouped = clean.groupby(sg_col)[var_col]
    result = pd.DataFrame({
        'n': grouped.count(),
        'mean': grouped.mean(),
        'std': grouped.std(ddof=1),
        'range': grouped.max() - grouped.min(),
        'min': grouped.min(),
        'max': grouped.max(),
    })
    # 빈 부분군 제거
    result = result[result['n'] > 0]
    return result


def normality_check(values, label=''):
    """
    정규성 검정 3종 (강의록 08 page 13)
    - Shapiro-Wilk
    - Anderson-Darling
    - Kolmogorov-Smirnov
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) < 3:
        return {'passed': True, 'pValue': 1.0, 'tests': [], 'note': '표본 부족'}

    tests = []

    # Shapiro-Wilk
    try:
        if len(values) <= 5000:
            stat, p = stats.shapiro(values)
            tests.append({'name': 'Shapiro-Wilk', 'pValue': float(p), 'passed': p > 0.05, 'stat': float(stat)})
    except Exception:
        pass

    # Anderson-Darling (정규분포 기준) + 근사 p-value (D'Agostino 1986)
    try:
        result = stats.anderson(values, dist='norm')
        a2 = float(result.statistic)
        n = len(values)
        a2_star = a2 * (1 + 0.75/n + 2.25/n**2)
        if a2_star >= 0.6:
            p_ad = float(np.exp(1.2937 - 5.709*a2_star + 0.0186*a2_star**2))
        elif a2_star > 0.34:
            p_ad = float(np.exp(0.9177 - 4.279*a2_star - 1.38*a2_star**2))
        elif a2_star > 0.2:
            p_ad = float(1 - np.exp(-8.318 + 42.796*a2_star - 59.938*a2_star**2))
        else:
            p_ad = float(1 - np.exp(-13.436 + 101.14*a2_star - 223.73*a2_star**2))
        p_ad = max(0.0, min(1.0, p_ad))
        tests.append({'name': 'Anderson-Darling', 'pValue': p_ad, 'passed': p_ad > 0.05, 'stat': a2})
    except Exception:
        pass

    # Kolmogorov-Smirnov (정규분포 기준, μ̂/σ̂ 표본 추정값 사용)
    try:
        mu, sigma = float(np.mean(values)), float(np.std(values, ddof=1))
        if sigma > 0:
            ks_stat, ks_p = stats.kstest(values, 'norm', args=(mu, sigma))
            tests.append({'name': 'Kolmogorov-Smirnov', 'pValue': float(ks_p),
                         'passed': ks_p > 0.05, 'stat': float(ks_stat)})
    except Exception:
        pass

    if not tests:
        return {'passed': True, 'pValue': 1.0, 'tests': [], 'note': '검정 실패'}

    min_p = min(t['pValue'] for t in tests)
    return {
        'passed': bool(min_p > 0.05),
        'pValue': min_p,
        'tests': tests,
        'skewness': float(stats.skew(values)),
        'kurtosis': float(stats.kurtosis(values)),
        'note': '정규분포로 가정 가능' if min_p > 0.05 else '정규분포로 가정하기 어려움 (Box-Cox 변환 권장)',
    }


def qq_plot_data(values):
    """Q-Q plot 데이터 생성 (z값 vs 이론분위수)"""
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    z = stats.zscore(values)
    (osm, osr), (slope, intercept, r) = stats.probplot(z, dist='norm')
    return pd.DataFrame({'theoretical': osm, 'sample': osr})


def histogram_bins(values, n_bins=20):
    """히스토그램 bin 계산"""
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    counts, edges = np.histogram(values, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({'x': centers, 'count': counts}), edges


def fit_normal_pdf(values, x_range=None, n_points=200):
    """정규분포 적합 곡선"""
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    mu, sigma = float(np.mean(values)), float(np.std(values, ddof=1))
    if x_range is None:
        lo = float(np.min(values)) - 0.5 * sigma
        hi = float(np.max(values)) + 0.5 * sigma
    else:
        lo, hi = x_range
    x = np.linspace(lo, hi, n_points)
    y = stats.norm.pdf(x, mu, sigma)
    return pd.DataFrame({'x': x, 'pdf': y, 'count_scale': y * len(values) * (hi - lo) / 20})
