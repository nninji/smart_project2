"""
강의록 예제 데이터셋 모음
- PVC 점도 (08 p.9) — 검증용 (Cp=1.3047 등)
- D램 웨이퍼 두께 (09 p.19)
- LED 전구 (09 p.22)
- 충전기 결함 C 관리도 (09 p.24)
- U 관리도 가변 (09 p.25)
- 비정규 데이터 - Box-Cox 시연 (08 p.27-29)
- 이상치 포함 (09 p.26-28)
- P 관리도 가변 표본
"""
import numpy as np
import pandas as pd
from .data_gen import generate_value_data, generate_count_data


def pvc_example():
    """PVC 점도 예제 — Cp=1.3047 검증용"""
    data = np.array([
        [3576.27, 3630.12, 3576.27, 3630.12, 3355.69, 3363.62],
        [3504.17, 3514.52, 3747.43, 3666.15, 3709.25, 3317.28],
        [3440.11, 3494.35, 3962.93, 3514.30, 3273.57, 3336.20],
        [3638.33, 3719.84, 3617.47, 3450.17, 3378.70, 3475.50],
        [3661.94, 3485.53, 3499.43, 3605.53, 3390.29, 3519.26],
    ])
    df_wide = pd.DataFrame(data, columns=['pl_1','pl_2','pl_3','pl_4','pl_5','pl_6'])
    df_long = df_wide.melt(var_name='prod_line', value_name='viscosity')
    return {
        'name': 'PVC 점도',
        'description': 'PVC 점도 측정 - 규격 3500±500, 6 lots × 5 obs. 기대: Cp=1.305, Cpk=1.213',
        'mode': 'variable',
        'df': df_long,
        'sg_col': 'prod_line', 'var_col': 'viscosity',
        'LSL': 3000, 'USL': 4000, 'target': 3500,
    }


def wafer_thickness_example():
    """3D 적층형 D램 웨이퍼 두께"""
    df = generate_value_data(var_name='Thickness', target=40, sg_name='Lot',
                             num_sg=30, sg_size=4, sg_std=2,
                             mean_shift=2, sg_size_variation=0, seed=42)
    return {
        'name': '3D D램 웨이퍼 두께',
        'description': '3D 적층형 D램 웨이퍼 두께 - 30 lots × 4 obs, target=40μm ±2μm',
        'mode': 'variable',
        'df': df,
        'sg_col': 'Lot', 'var_col': 'Thickness',
        'LSL': 38, 'USL': 42, 'target': 40,
    }


def led_np_example():
    """LED 전구 NP 관리도 예제"""
    df = generate_count_data(var_name='Defectives', sg_name='Lot',
                             num_sg=50, sg_size=300, p=0.02, sg_size_variation=0, seed=123)
    return {
        'name': 'LED 전구 불량개수',
        'description': 'LED 전구 50 lots × 300개 표본, NP 관리도용 (n 동일)',
        'mode': 'attribute',
        'df': df,
        'sg_col': 'Lot', 'size_col': 'sample_size', 'defect_col': 'Defectives',
    }


def p_chart_example():
    """P 관리도 가변 표본"""
    df = generate_count_data(var_name='Defectives', sg_name='Lot',
                             num_sg=50, sg_size=300, p=0.02, sg_size_variation=50, seed=234)
    return {
        'name': 'P 관리도 가변 표본',
        'description': '50 lots, 표본크기 250~350 가변, p≈0.02',
        'mode': 'attribute',
        'df': df,
        'sg_col': 'Lot', 'size_col': 'sample_size', 'defect_col': 'Defectives',
    }


def charger_c_example():
    """충전기 결함수 C 관리도"""
    df = generate_count_data(var_name='Defects', sg_name='Lot',
                             num_sg=20, sg_size=500, p=0.1, sg_size_variation=0, seed=345)
    return {
        'name': '충전기 결함수',
        'description': '20일 × 500개 충전기, 5종 결함 합계, C 관리도용',
        'mode': 'attribute',
        'df': df,
        'sg_col': 'Lot', 'size_col': 'sample_size', 'defect_col': 'Defects',
    }


def u_chart_example():
    """U 관리도 가변 결함률"""
    df = generate_count_data(var_name='Defects', sg_name='Lot',
                             num_sg=20, sg_size=500, p=0.1, sg_size_variation=100, seed=456)
    return {
        'name': 'U 관리도 가변 크기',
        'description': '20 lots × 400~600개, 결함률 0.1, U 관리도용',
        'mode': 'attribute',
        'df': df,
        'sg_col': 'Lot', 'size_col': 'sample_size', 'defect_col': 'Defects',
    }


def nonnormal_lognormal_example():
    """비정규 로그정규 데이터 - Box-Cox 시연"""
    np.random.seed(42)
    data = np.random.lognormal(mean=2.5, sigma=0.4, size=150)
    df = pd.DataFrame({
        'Lot': np.repeat(range(1, 31), 5)[:150],
        'Strength': data,
    })
    return {
        'name': '비정규 데이터 - Box-Cox 시연',
        'description': '로그정규분포 (mean=2.5, sigma=0.4) - Box-Cox 변환 후 정규성 만족',
        'mode': 'variable',
        'df': df,
        'sg_col': 'Lot', 'var_col': 'Strength',
        'LSL': 2, 'USL': 30, 'target': 12,
    }


def outlier_included_example():
    """이상치 포함 데이터 - 재작성 시연"""
    np.random.seed(100)
    df = generate_value_data(var_name='Weight', target=50, sg_name='Lot',
                             num_sg=25, sg_size=5, sg_std=2,
                             mean_shift=3, sg_size_variation=0, seed=100)
    # lot 19에 의도적 이상치 추가
    df.loc[df['Lot'] == 19, 'Weight'] += 5
    return {
        'name': '이상치 포함 - 재작성 시연',
        'description': 'lot 19에 이상치 추가 - "이상치 제거 후 재작성" 기능 시연용',
        'mode': 'variable',
        'df': df,
        'sg_col': 'Lot', 'var_col': 'Weight',
        'LSL': 45, 'USL': 55, 'target': 50,
    }


EXAMPLE_DATASETS = [
    {'id': 'pvc',      'label': 'PVC 점도 (검증용)',          'source': '계량형 · 6 lots × 5 obs',  'loader': pvc_example,             'badge': '검증'},
    {'id': 'wafer',    'label': '3D 웨이퍼 두께',              'source': '계량형 · 30 lots × 4 obs', 'loader': wafer_thickness_example, 'badge': '계량'},
    {'id': 'outlier',  'label': '이상치 포함 데이터',         'source': '재작성 절차 시연용',         'loader': outlier_included_example,'badge': '재작성'},
    {'id': 'nonnorm',  'label': '비정규 데이터 (Box-Cox 시연)','source': '로그정규분포 · 150 obs',    'loader': nonnormal_lognormal_example, 'badge': '변환'},
    {'id': 'led',      'label': 'LED 전구 불량 (NP)',         'source': '계수형 · 50 lots × 300개', 'loader': led_np_example,          'badge': '계수'},
    {'id': 'pchart',   'label': 'P 관리도 가변 표본',          'source': '계수형 · 가변 크기',       'loader': p_chart_example,         'badge': '계수'},
    {'id': 'charger',  'label': '충전기 결함 (C)',            'source': '계수형 · 20 lots × 500개',  'loader': charger_c_example,       'badge': '계수'},
    {'id': 'ucharter', 'label': 'U 관리도 가변 표본',          'source': '계수형 · 가변 크기',       'loader': u_chart_example,         'badge': '계수'},
]
