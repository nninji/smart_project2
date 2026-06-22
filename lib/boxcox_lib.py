"""
Box-Cox 변환 (강의록 08 page 27-29)
scipy.stats.boxcox 사용 — 강의록 코드와 동일
"""
import numpy as np
from scipy import stats


def boxcox_transform(values, lam=None):
    """
    Box-Cox 변환
    @param values: 양수 데이터
    @param lam: 람다 (None이면 MLE로 자동 추정)
    @returns: dict with transformed, lambda, error
    """
    values = np.asarray(values)
    if (values <= 0).any():
        return {'error': 'Box-Cox는 모든 값이 양수여야 합니다. 음수/0 포함 시 Yeo-Johnson 변환이 필요합니다.'}
    if len(values) < 3:
        return {'error': '표본이 너무 작습니다 (n≥3 필요)'}

    if lam is None:
        # MLE로 자동 추정 (강의록과 동일: scipy.stats.boxcox)
        transformed, fitted_lam = stats.boxcox(values)
        return {
            'transformed': transformed,
            'lambda': float(fitted_lam),
            'interp': interpret_lambda(float(fitted_lam)),
        }
    else:
        # 명시된 람다로 변환
        transformed = stats.boxcox(values, lmbda=lam)
        return {
            'transformed': transformed,
            'lambda': float(lam),
            'interp': interpret_lambda(float(lam)),
        }


def boxcox_value(v, lam):
    """단일 값 변환"""
    if v <= 0:
        return np.nan
    if abs(lam) < 1e-7:
        return np.log(v)
    return (v**lam - 1) / lam


def interpret_lambda(lam):
    """람다 의미 해석 (강의록 08 p.27)"""
    if abs(lam) < 0.1:
        return {'name': '로그 변환 (ln(y))', 'desc': '강한 우편향 (지수적 증가)'}
    if abs(lam - 0.5) < 0.1:
        return {'name': '제곱근 변환 (√y)', 'desc': '중간 우편향 (Poisson 분포 유사)'}
    if abs(lam - 1) < 0.15:
        return {'name': '변환 없음 (원본)', 'desc': '이미 정규분포에 가까움'}
    if abs(lam - 2) < 0.2:
        return {'name': '제곱 변환 (y²)', 'desc': '좌편향 (꼬리가 왼쪽)'}
    if lam < 0:
        return {'name': '역수형 변환', 'desc': '매우 강한 우편향'}
    return {'name': f'거듭제곱 변환 (y^{lam:.2f})', 'desc': '데이터 특성에 맞춘 일반 변환'}
