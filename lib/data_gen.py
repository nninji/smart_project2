"""
시뮬레이션 데이터 생성 함수
강의록 09 page 6-7의 generate_value_data, generate_count_data와 정확히 동일
"""
import numpy as np
import pandas as pd


def generate_value_data(var_name='value', target=0, sg_name='subgroup',
                        num_sg=1, sg_size=2, sg_std=1,
                        mean_shift=0, sg_size_variation=1, seed=None):
    """
    계량형 데이터 생성 (강의록 09 page 6과 동일)

    @param target: 목표값
    @param num_sg: 부분군 수
    @param sg_size: 부분군당 표본 크기
    @param sg_std: 부분군 내 표준편차
    @param mean_shift: 부분군 평균이 이동할 수 있는 범위 (uniform(-shift, +shift))
    @param sg_size_variation: 부분군 크기 변동 (randint(-var, var+1))
    """
    if seed is not None:
        np.random.seed(seed)
    df = pd.DataFrame()
    for i in range(num_sg):
        shift = np.random.uniform(-mean_shift, mean_shift) if mean_shift > 0 else 0
        n_i = sg_size + (np.random.randint(-sg_size_variation, sg_size_variation+1) if sg_size_variation > 0 else 0)
        sampled = pd.DataFrame({
            sg_name: i + 1,
            var_name: np.random.normal(loc=target+shift, scale=sg_std, size=n_i)
        })
        df = pd.concat([df, sampled], axis=0, ignore_index=True)
    return df


def generate_count_data(var_name='count', sg_name='lot',
                       num_sg=20, sg_size=200, p=0.01,
                       sg_size_variation=1, seed=None):
    """
    계수형 데이터 생성 (강의록 09 page 7과 동일)

    @param sg_size: 부분군 표본 크기 (기준)
    @param p: 불량률/결함률
    @param sg_size_variation: 부분군 크기 변동
    """
    if seed is not None:
        np.random.seed(seed)
    df = pd.DataFrame()
    for i in range(num_sg):
        n_i = sg_size + (np.random.randint(-sg_size_variation, sg_size_variation+1) if sg_size_variation > 0 else 0)
        sampled = pd.DataFrame({
            sg_name:       [i + 1],
            'sample_size': [n_i],
            var_name:      [np.random.binomial(n=n_i, p=p)],
        })
        df = pd.concat([df, sampled], axis=0, ignore_index=True)
    return df
