"""
Process Capability & SPC Studio — Streamlit Edition
강의록 08_공정능력분석.pdf + 09_통계적공정관리.pdf 기반

7개 페이지: 데이터 / 종합 / 공정능력 / SPC / 비교 / What-if / LIVE
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

from lib.constants import unbiased_const, grade_capability
from lib.stats_lib import subgroup_stats, normality_check, qq_plot_data, fit_normal_pdf
from lib.capability import compute_capability, SIGMA_WITHIN_METHODS
from lib.spc import (xbar_r_chart, xbar_s_chart, imr_chart, cusum_chart, ewma_chart,
                     np_chart, p_chart, c_chart, u_chart,
                     apply_nelson_rules, NELSON_RULES_TEXT)
from lib.boxcox_lib import boxcox_transform, boxcox_value, interpret_lambda
from lib.data_gen import generate_value_data, generate_count_data
from lib.datasets import EXAMPLE_DATASETS
from lib.insights import interpret_capability, interpret_spc


# ────────────────────────────────────────────────────────────────────
# 앱 설정
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Process Capability & SPC Studio',
    page_icon='σ',
    layout='wide',
    initial_sidebar_state='expanded',
)

# 커스텀 CSS — Light 테마 유지하면서 임팩트 강화
st.markdown("""
<style>
    /* 헤더 영역 */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1300px; }

    /* 페이지 hero 헤더 (첨부 이미지 스타일) */
    .page-hero {
        margin: 0.5rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .page-hero h1 {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        color: #0f172a;
        margin: 0 0 0.5rem 0 !important;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .page-hero .hint {
        font-size: 1rem;
        color: #475569;
        margin: 0;
        line-height: 1.5;
    }
    .page-hero .hint .emoji {
        display: inline-block;
        margin-right: 6px;
    }
    .page-hero .badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        background: #dbeafe;
        color: #1d4ed8;
        margin-left: 8px;
        vertical-align: middle;
        letter-spacing: 0.02em;
    }

    /* 사이드바 디자인 강화 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fafbfc 0%, #f1f5f9 100%);
    }
    [data-testid="stSidebar"] .sidebar-brand {
        padding: 0.5rem 0 1rem 0;
        text-align: center;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    [data-testid="stSidebar"] .sidebar-brand .logo {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px; height: 42px;
        background: #0f172a;
        color: white;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 8px;
    }
    [data-testid="stSidebar"] .sidebar-brand .title {
        font-size: 0.95rem; font-weight: 700; color: #0f172a;
    }
    [data-testid="stSidebar"] .sidebar-brand .subtitle {
        font-size: 0.72rem; color: #64748b; margin-top: 2px;
    }
    [data-testid="stSidebar"] .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin: 1.2rem 0 0.5rem 0;
    }

    /* 메트릭 카드 스타일 */
    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
    /* mono 클래스 */
    .mono { font-family: 'JetBrains Mono', monospace; }

    /* 액션 강조 박스 (메인 영역) */
    .action-banner {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #93c5fd;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 0.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .action-banner .icon { font-size: 22px; }
    .action-banner .text { color: #1e3a8a; font-size: 14px; }
    .action-banner .text strong { font-weight: 700; }

    /* 인사이트 카드 */
    .insight-card {
        padding: 12px 14px;
        border-radius: 10px;
        margin: 6px 0;
        border: 1px solid;
    }
    .insight-success { background: #ecfdf5; border-color: #6ee7b7; color: #064e3b; }
    .insight-warn    { background: #fefce8; border-color: #fde047; color: #713f12; }
    .insight-bad     { background: #fef2f2; border-color: #fca5a5; color: #7f1d1d; }
    .insight-info    { background: #eff6ff; border-color: #93c5fd; color: #1e3a8a; }

    /* LIVE 점 깜빡임 */
    .live-dot {
        display: inline-block; width: 12px; height: 12px;
        background: #ef4444; border-radius: 50%;
        animation: pulse 1.2s ease-in-out infinite;
        margin-right: 8px; vertical-align: middle;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
        50% { opacity: 0.6; box-shadow: 0 0 0 8px rgba(239,68,68,0); }
    }

    /* 데이터 상태 카드 */
    .data-status-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 0.5rem 0;
    }
    .data-status-card.has-data { border-left: 3px solid #10b981; }
    .data-status-card.no-data  { border-left: 3px solid #94a3b8; }
    .data-status-card .label { font-size: 0.7rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .data-status-card .value { font-size: 0.85rem; color: #0f172a; font-weight: 600; margin-top: 4px; }
    .data-status-card .meta  { font-size: 0.72rem; color: #64748b; margin-top: 2px; }

    /* 큰 액션 버튼 영역 */
    .stButton button[kind="primary"] {
        font-weight: 600;
        letter-spacing: -0.01em;
    }
</style>
""", unsafe_allow_html=True)


def page_header(title, hint=None, badge=None, live=False):
    """임팩트 있는 페이지 헤더 (첨부 이미지 스타일)"""
    badge_html = f'<span class="badge">{badge}</span>' if badge else ''
    live_html = '<span class="live-dot"></span>' if live else ''
    hint_html = f'<p class="hint"><span class="emoji">💡</span>{hint}</p>' if hint else ''
    st.markdown(f"""
<div class="page-hero">
  <h1>{live_html}{title}{badge_html}</h1>
  {hint_html}
</div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# 세션 상태 초기화
# ────────────────────────────────────────────────────────────────────
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'saved_dataset' not in st.session_state:
    st.session_state.saved_dataset = None
if 'live_points' not in st.session_state:
    st.session_state.live_points = []
if 'outlier_preview' not in st.session_state:
    st.session_state.outlier_preview = []
if 'sigma_method' not in st.session_state:
    st.session_state.sigma_method = 'pooled'


# ────────────────────────────────────────────────────────────────────
# 사이드바 — 그룹화 + 브랜드 강조
# ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # 브랜드 영역
    st.markdown("""
<div class="sidebar-brand">
  <div class="logo">σ</div>
  <div class="title">Process Capability<br>&amp; SPC Studio</div>
  <div class="subtitle">공정능력 + SPC 분석 도구</div>
</div>
    """, unsafe_allow_html=True)

    # 페이지 네비
    st.markdown('<div class="section-label">📑 페이지</div>', unsafe_allow_html=True)
    page = st.radio(
        "페이지 선택",
        ['① 데이터', '② 종합 대시보드', '③ 공정능력분석', '④ SPC 관리도',
         '⑤ 데이터셋 비교', '⑥ What-if 시뮬', '⑦ 🔴 LIVE 모드'],
        label_visibility='collapsed',
    )

    # 현재 데이터 상태 카드
    st.markdown('<div class="section-label">📦 현재 데이터</div>', unsafe_allow_html=True)
    if st.session_state.dataset is not None:
        d = st.session_state.dataset
        spec_meta = ''
        if d['mode'] == 'variable' and d.get('LSL') is not None:
            spec_meta = f"LSL={d['LSL']} · USL={d['USL']}"
        st.markdown(f"""
<div class="data-status-card has-data">
  <div class="label">{d['mode'].upper()}</div>
  <div class="value">{d['meta']['source']}</div>
  <div class="meta">{len(d['df'])} rows · {spec_meta}</div>
</div>
        """, unsafe_allow_html=True)
        if st.session_state.saved_dataset and len(d['df']) != len(st.session_state.saved_dataset['df']):
            if st.button("↩ 원본 데이터 복원", use_container_width=True):
                st.session_state.dataset = st.session_state.saved_dataset.copy()
                st.session_state.dataset['df'] = st.session_state.saved_dataset['df'].copy()
                st.rerun()
    else:
        st.markdown("""
<div class="data-status-card no-data">
  <div class="label">데이터 없음</div>
  <div class="meta">⬆ 위에서 '① 데이터' 선택 후 불러오기</div>
</div>
        """, unsafe_allow_html=True)

    # 구현 범위 안내
    st.markdown('<div class="section-label">🛠️ 구현 범위</div>', unsafe_allow_html=True)
    st.caption(
        "• Cp / Cpk / Pp / Ppk\n\n"
        "• Box-Cox 변환 (scipy)\n\n"
        "• 9종 관리도 (계량 5 + 계수 4)\n\n"
        "• Nelson Rules 8가지\n\n"
        "• What-if + LIVE 시뮬"
    )

    st.markdown('<div class="section-label">📐 분석 기법</div>', unsafe_allow_html=True)
    st.caption(
        "08_공정능력분석.pdf\n\n"
        "09_통계적공정관리.pdf"
    )


# ────────────────────────────────────────────────────────────────────
# 차트 헬퍼
# ────────────────────────────────────────────────────────────────────

def draw_control_chart(chart_result, checked=None, title=''):
    """관리도 그리기 (Plotly)"""
    primary = chart_result['primary']
    has_secondary = 'secondary' in chart_result

    if has_secondary:
        secondary = chart_result['secondary']
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           subplot_titles=(chart_result['primaryLabel'], chart_result['secondaryLabel']))
    else:
        fig = make_subplots(rows=1, cols=1)

    # 위반점 색 결정용
    def get_marker_color(violations):
        if not violations:
            return '#3b82f6'  # 정상
        if 1 in violations:
            return '#ef4444'  # 한계이탈
        return '#f59e0b'  # 패턴위반

    def get_marker_size(violations):
        return 10 if violations else 6

    # primary 차트
    colors = []
    sizes = []
    if checked is not None:
        for v in checked['violations']:
            colors.append(get_marker_color(v))
            sizes.append(get_marker_size(v))
    else:
        colors = ['#3b82f6'] * len(primary)
        sizes = [6] * len(primary)

    fig.add_trace(go.Scatter(
        x=primary['sg'], y=primary['point'],
        mode='lines+markers', name=chart_result['primaryLabel'],
        line=dict(color='#3b82f6', width=1.5),
        marker=dict(color=colors, size=sizes, line=dict(color='white', width=1)),
        hovertemplate='%{x}<br>값: %{y:.4f}<extra></extra>',
    ), row=1, col=1)

    # 관리한계
    UCL_val = primary['UCL'].iloc[0] if hasattr(primary['UCL'], 'iloc') else primary['UCL'].values[0] if len(primary['UCL'])>0 else None
    CL_val  = primary['CL'].iloc[0]  if hasattr(primary['CL'], 'iloc')  else primary['CL'].values[0]
    LCL_val = primary['LCL'].iloc[0] if hasattr(primary['LCL'], 'iloc') else primary['LCL'].values[0]

    # P, U 같이 가변 관리한계는 line으로
    if 'n_i' in primary.columns or chart_result.get('type') == 'EWMA':
        fig.add_trace(go.Scatter(x=primary['sg'], y=primary['UCL'], mode='lines',
                                 line=dict(color='#ec4899', width=1, dash='dot'), name='UCL'), row=1, col=1)
        fig.add_trace(go.Scatter(x=primary['sg'], y=primary['LCL'], mode='lines',
                                 line=dict(color='#ef4444', width=1, dash='dot'), name='LCL'), row=1, col=1)
    else:
        fig.add_hline(y=UCL_val, line_color='#ec4899', line_dash='dot', line_width=1,
                     annotation_text=f"UCL={UCL_val:.4f}", annotation_position="right", row=1, col=1)
        fig.add_hline(y=LCL_val, line_color='#ef4444', line_dash='dot', line_width=1,
                     annotation_text=f"LCL={LCL_val:.4f}", annotation_position="right", row=1, col=1)

    fig.add_hline(y=CL_val, line_color='#10b981', line_dash='dashdot', line_width=1.2,
                 annotation_text=f"CL={CL_val:.4f}", annotation_position="right", row=1, col=1)

    # secondary 차트
    if has_secondary:
        fig.add_trace(go.Scatter(
            x=secondary['sg'], y=secondary['point'],
            mode='lines+markers', name=chart_result['secondaryLabel'],
            line=dict(color='#a855f7', width=1.5),
            marker=dict(color='#a855f7', size=6),
            hovertemplate='%{x}<br>값: %{y:.4f}<extra></extra>',
        ), row=2, col=1)
        UCL2 = secondary['UCL'].iloc[0] if hasattr(secondary['UCL'], 'iloc') else secondary['UCL'].values[0]
        CL2  = secondary['CL'].iloc[0]  if hasattr(secondary['CL'], 'iloc')  else secondary['CL'].values[0]
        LCL2 = secondary['LCL'].iloc[0] if hasattr(secondary['LCL'], 'iloc') else secondary['LCL'].values[0]
        fig.add_hline(y=UCL2, line_color='#ec4899', line_dash='dot', line_width=1,
                     annotation_text=f"UCL={UCL2:.4f}", annotation_position="right", row=2, col=1)
        fig.add_hline(y=CL2, line_color='#10b981', line_dash='dashdot', line_width=1.2,
                     annotation_text=f"CL={CL2:.4f}", annotation_position="right", row=2, col=1)
        fig.add_hline(y=LCL2, line_color='#ef4444', line_dash='dot', line_width=1,
                     annotation_text=f"LCL={LCL2:.4f}", annotation_position="right", row=2, col=1)

    fig.update_layout(
        title=title or f'{chart_result["type"]} 관리도',
        height=600 if has_secondary else 400,
        showlegend=False, margin=dict(l=40, r=80, t=50, b=30),
        plot_bgcolor='rgba(240,240,250,0.3)',
    )
    return fig


def render_insights(insights):
    """인사이트 카드 렌더링"""
    for ins in insights:
        tone_class = f"insight-{ins['tone']}"
        st.markdown(f"""
<div class="insight-card {tone_class}">
  <div style="display: flex; align-items: start; gap: 12px;">
    <div style="font-size: 22px;">{ins['icon']}</div>
    <div style="flex: 1;">
      <div style="font-weight: 600; font-size: 14px;">{ins['title']}</div>
      <div style="font-size: 13px; margin-top: 4px; opacity: 0.9;">{ins['body']}</div>
    </div>
  </div>
</div>
        """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# 페이지 1: 데이터
# ────────────────────────────────────────────────────────────────────
if page == '① 데이터':
    page_header(
        "데이터 입력",
        hint="CSV를 업로드하거나 예제 데이터셋 8가지 중 하나를 선택해 시작하세요. 시뮬레이션 데이터도 만들 수 있습니다."
    )

    # 첫 방문자 안내 배너
    if st.session_state.dataset is None:
        st.markdown("""
<div class="action-banner">
  <div class="icon">🚀</div>
  <div class="text">
    <strong>처음이신가요?</strong> 아래 <strong>"📚 예제 데이터셋"</strong>에서 <strong>PVC 점도</strong>를 불러오면
    가장 빠르게 모든 기능을 체험할 수 있습니다 (Cp=1.3047 검증 확인 가능).
  </div>
</div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📂 CSV 업로드")
        st.caption("Long format(권장) 또는 Wide format(자동 변환)")
        uploaded = st.file_uploader("CSV 파일", type=['csv'], label_visibility='collapsed')
        if uploaded:
            df = pd.read_csv(uploaded)
            # Wide format 감지
            all_numeric = all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
            if all_numeric and len(df.columns) >= 2:
                if st.checkbox(f"Wide format으로 감지됨 ({len(df.columns)}개 컬럼) — Long으로 변환?", value=True):
                    df = df.melt(var_name='subgroup', value_name='value')
            st.session_state.dataset = {
                'mode': 'variable',
                'df': df,
                'sg_col': df.columns[0],
                'var_col': df.columns[1] if len(df.columns) > 1 else df.columns[0],
                'LSL': None, 'USL': None, 'target': None,
                'meta': {'source': uploaded.name, 'description': ''},
            }
            st.session_state.saved_dataset = st.session_state.dataset.copy()
            st.session_state.saved_dataset['df'] = df.copy()
            st.success("✅ 업로드 완료")
            st.rerun()

    with col2:
        st.subheader("📚 예제 데이터셋")
        st.caption("실무 시나리오 8가지")
        ex_labels = [f"{e['label']} · {e['source']}" for e in EXAMPLE_DATASETS]
        selected = st.selectbox("예제 선택", range(len(ex_labels)), format_func=lambda i: ex_labels[i])
        if st.button("📥 예제 불러오기", use_container_width=True):
            ex = EXAMPLE_DATASETS[selected]
            data = ex['loader']()
            ds = {
                'mode': data['mode'],
                'df': data['df'],
                'sg_col': data['sg_col'],
                'var_col': data.get('var_col'),
                'size_col': data.get('size_col'),
                'defect_col': data.get('defect_col'),
                'LSL': data.get('LSL'),
                'USL': data.get('USL'),
                'target': data.get('target'),
                'meta': {'source': data['name'], 'description': data['description']},
            }
            st.session_state.dataset = ds
            st.session_state.saved_dataset = {**ds, 'df': data['df'].copy()}
            st.rerun()

    with col3:
        st.subheader("🎲 시뮬레이션 생성")
        st.caption("계량형 / 계수형 데이터를 다양한 파라미터로 생성")
        sim_mode = st.radio("타입", ['계량형 (X̄-R)', '계수형 (NP/P/C/U)'], horizontal=True)
        if sim_mode == '계량형 (X̄-R)':
            target = st.number_input("목표값", value=100.0)
            tol    = st.number_input("허용오차 (±)", value=5.0, min_value=0.1)
            num_sg = st.number_input("부분군 수", value=20, min_value=2, max_value=100)
            sg_sz  = st.number_input("부분군 크기", value=5, min_value=1, max_value=30)
            sg_std = st.number_input("부분군 내 표준편차", value=1.0, min_value=0.01)
            shift  = st.number_input("평균 시프트 (±)", value=0.5, min_value=0.0)
            if st.button("🎲 생성", use_container_width=True):
                df_sim = generate_value_data(var_name='value', target=target, sg_name='Lot',
                                            num_sg=num_sg, sg_size=sg_sz, sg_std=sg_std,
                                            mean_shift=shift, sg_size_variation=0)
                ds = {
                    'mode': 'variable', 'df': df_sim,
                    'sg_col': 'Lot', 'var_col': 'value',
                    'LSL': target - tol, 'USL': target + tol, 'target': target,
                    'meta': {'source': '시뮬레이션 데이터', 'description': f'target={target}±{tol}, n={sg_sz}×{num_sg}'},
                }
                st.session_state.dataset = ds
                st.session_state.saved_dataset = {**ds, 'df': df_sim.copy()}
                st.rerun()
        else:
            num_sg = st.number_input("부분군 수", value=30, min_value=5, max_value=100, key='cnt_n')
            sg_sz  = st.number_input("부분군 크기", value=200, min_value=10, max_value=1000, key='cnt_s')
            p_rate = st.number_input("불량률 p", value=0.02, min_value=0.001, max_value=0.5, step=0.005, format='%.3f', key='cnt_p')
            sg_var = st.number_input("크기 변동 (±)", value=0, min_value=0, max_value=100, key='cnt_v')
            if st.button("🎲 생성", use_container_width=True, key='gen_cnt'):
                df_sim = generate_count_data(var_name='Defects', sg_name='Lot',
                                            num_sg=num_sg, sg_size=sg_sz, p=p_rate, sg_size_variation=sg_var)
                ds = {
                    'mode': 'attribute', 'df': df_sim,
                    'sg_col': 'Lot', 'size_col': 'sample_size', 'defect_col': 'Defects',
                    'meta': {'source': '계수형 시뮬레이션', 'description': f'p={p_rate}, n={sg_sz}±{sg_var}'},
                }
                st.session_state.dataset = ds
                st.session_state.saved_dataset = {**ds, 'df': df_sim.copy()}
                st.rerun()

    # 데이터 설정 + 미리보기
    if st.session_state.dataset is not None:
        st.divider()
        st.subheader("⚙️ 컬럼 · 규격 설정")
        d = st.session_state.dataset
        if d['meta'].get('description'):
            st.info(f"💡 {d['meta']['description']}")

        ds_changed = False
        cols = st.columns(4)
        with cols[0]:
            new_sg = st.selectbox("부분군 컬럼", d['df'].columns.tolist(),
                                  index=list(d['df'].columns).index(d['sg_col']) if d['sg_col'] in d['df'].columns else 0)
            if new_sg != d['sg_col']:
                d['sg_col'] = new_sg; ds_changed = True

        with cols[1]:
            mode = st.radio("모드", ['variable', 'attribute'],
                           index=0 if d['mode']=='variable' else 1, horizontal=True)
            if mode != d['mode']:
                d['mode'] = mode; ds_changed = True

        if d['mode'] == 'variable':
            with cols[2]:
                var_options = [c for c in d['df'].columns if pd.api.types.is_numeric_dtype(d['df'][c])]
                if d.get('var_col') not in var_options and var_options:
                    d['var_col'] = var_options[0]
                if var_options:
                    new_var = st.selectbox("측정값 컬럼", var_options,
                                          index=var_options.index(d['var_col']) if d['var_col'] in var_options else 0)
                    if new_var != d['var_col']:
                        d['var_col'] = new_var; ds_changed = True

            with cols[3]:
                pass

            c1, c2, c3 = st.columns(3)
            with c1:
                lsl = st.number_input("LSL (규격하한)", value=float(d['LSL']) if d['LSL'] is not None else 0.0)
                if lsl != d['LSL']: d['LSL'] = lsl; ds_changed = True
            with c2:
                usl = st.number_input("USL (규격상한)", value=float(d['USL']) if d['USL'] is not None else 1.0)
                if usl != d['USL']: d['USL'] = usl; ds_changed = True
            with c3:
                tgt = st.number_input("목표값 (선택)", value=float(d['target']) if d['target'] is not None else 0.0)
                if tgt != d['target']: d['target'] = tgt; ds_changed = True

            if d['LSL'] is not None and d['USL'] is not None and d['LSL'] >= d['USL']:
                st.error(f"⚠️ LSL({d['LSL']})은 USL({d['USL']})보다 작아야 합니다.")
        else:
            with cols[2]:
                num_options = [c for c in d['df'].columns if pd.api.types.is_numeric_dtype(d['df'][c])]
                if d.get('size_col') and d['size_col'] in num_options:
                    new_size = st.selectbox("표본크기 컬럼", num_options,
                                           index=num_options.index(d['size_col']))
                    if new_size != d['size_col']: d['size_col'] = new_size; ds_changed = True
            with cols[3]:
                num_options = [c for c in d['df'].columns if pd.api.types.is_numeric_dtype(d['df'][c])]
                if d.get('defect_col') and d['defect_col'] in num_options:
                    new_def = st.selectbox("불량/결점 컬럼", num_options,
                                          index=num_options.index(d['defect_col']))
                    if new_def != d['defect_col']: d['defect_col'] = new_def; ds_changed = True

        # 미리보기
        st.divider()
        st.subheader(f"📋 데이터 미리보기 ({len(d['df'])} rows × {len(d['df'].columns)} cols)")
        st.dataframe(d['df'].head(10), use_container_width=True, height=320)
        csv = d['df'].to_csv(index=False).encode('utf-8')
        st.download_button("⬇ CSV 다운로드", csv, file_name=f"spc-data-{int(time.time())}.csv", mime='text/csv')


# ────────────────────────────────────────────────────────────────────
# 페이지 2: 종합 대시보드
# ────────────────────────────────────────────────────────────────────
elif page == '② 종합 대시보드':
    page_header(
        "종합 대시보드",
        hint="현재 공정의 상태를 한눈에. 신호등 · 핵심 KPI · 분석 절차 안내."
    )
    d = st.session_state.dataset
    if d is None:
        st.markdown("""
<div class="action-banner">
  <div class="icon">📂</div>
  <div class="text">
    <strong>데이터가 아직 없습니다.</strong> 좌측 메뉴에서 <strong>'① 데이터'</strong>를 선택해 예제 또는 CSV를 불러와 주세요.
  </div>
</div>
        """, unsafe_allow_html=True)
        st.stop()
    if d['mode'] != 'variable':
        st.info("계수형 데이터는 SPC 관리도 페이지에서 확인하세요"); st.stop()
    if d.get('LSL') is None or d.get('USL') is None:
        st.warning("LSL/USL을 데이터 페이지에서 먼저 설정하세요"); st.stop()

    cap = compute_capability(d['df'], d['sg_col'], d['var_col'], d['LSL'], d['USL'])
    if 'error' in cap:
        st.error(cap['error']); st.stop()

    # 상태 신호등
    cpk = cap['Cpk']
    if cpk >= 1.67:   status_color, status = '#10b981', '우수'
    elif cpk >= 1.33: status_color, status = '#22c55e', '양호'
    elif cpk >= 1.00: status_color, status = '#eab308', '주의'
    else:             status_color, status = '#ef4444', '심각'

    st.markdown(f"""
<div style="display: flex; align-items: center; gap: 12px; padding: 12px;
            background: {status_color}15; border-left: 4px solid {status_color};
            border-radius: 6px; margin-bottom: 16px;">
  <div style="font-size: 24px;">●</div>
  <div>
    <div style="font-size: 18px; font-weight: 600; color: {status_color};">공정상태: {status}</div>
    <div style="font-size: 12px; opacity: 0.8;">{cap['gradeCpk']['action']}</div>
  </div>
</div>
    """, unsafe_allow_html=True)

    # KPI 카드
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 평균 X̄", f"{cap['xBar']:.4f}",
              delta=f"{cap['xBar']-d['target']:+.3f}" if d.get('target') else None)
    c2.metric("Cpk", f"{cap['Cpk']:.3f}", help=cap['gradeCpk']['level'])
    c3.metric("Ppk", f"{cap['Ppk']:.3f}", help=cap['gradePpk']['level'])
    c4.metric("PPM (장기)", f"{cap['ppmOverall']:,.0f}")

    # 분석 절차 안내
    st.divider()
    st.subheader("📋 분석 절차")
    p1, p2, p3, p4 = st.columns(4)
    p1.info("**① 품질 특성 선정**\n관리할 변수 결정\n(예: 점도, 두께)")
    p2.info("**② 부분군 크기 결정**\n5개 이상, 전체 30+\n(통계적 권장)")
    p3.info("**③ 관리도 종류 결정**\nn=1→I-MR\nn=2~10→X̄-R\nn=10+→X̄-s")
    p4.info("**④ 통계량 계산 & 플롯**\nCL/UCL/LCL + \nNelson Rules 8")

    # 부분군별 트렌드
    st.divider()
    sg = cap['subgroupStats'].reset_index()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=('부분군별 평균', '부분군별 표준편차'))
    fig.add_trace(go.Scatter(x=sg[d['sg_col']].astype(str), y=sg['mean'],
                            mode='lines+markers', line=dict(color='#3b82f6')), row=1, col=1)
    fig.add_hline(y=cap['xBar'], line_dash='dashdot', line_color='#10b981', row=1, col=1,
                  annotation_text=f"X̄̄={cap['xBar']:.3f}")
    fig.add_hline(y=d['USL'], line_dash='dash', line_color='#ef4444', row=1, col=1,
                  annotation_text=f"USL={d['USL']}")
    fig.add_hline(y=d['LSL'], line_dash='dash', line_color='#ef4444', row=1, col=1,
                  annotation_text=f"LSL={d['LSL']}")
    fig.add_trace(go.Scatter(x=sg[d['sg_col']].astype(str), y=sg['std'],
                            mode='lines+markers', line=dict(color='#a855f7')), row=2, col=1)
    fig.update_layout(height=500, showlegend=False, margin=dict(l=40, r=80, t=60, b=30))
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────
# 페이지 3: 공정능력분석
# ────────────────────────────────────────────────────────────────────
elif page == '③ 공정능력분석':
    page_header(
        "공정능력 분석",
        hint="Cp · Cpk · Pp · Ppk 계산과 5등급 판정. 정규성 불만족 시 Box-Cox 변환 자동 적용 가능."
    )
    d = st.session_state.dataset
    if d is None:
        st.markdown("""
<div class="action-banner">
  <div class="icon">📂</div>
  <div class="text">
    <strong>데이터가 아직 없습니다.</strong> 좌측 메뉴에서 <strong>'① 데이터'</strong>를 선택해 예제 또는 CSV를 불러와 주세요.
  </div>
</div>
        """, unsafe_allow_html=True)
        st.stop()
    if d['mode'] != 'variable':
        st.info("공정능력분석은 계량형 데이터에서만 가능합니다"); st.stop()
    if d.get('LSL') is None or d.get('USL') is None:
        st.warning("LSL/USL을 데이터 페이지에서 먼저 설정하세요"); st.stop()
    if d['LSL'] >= d['USL']:
        st.error(f"⚠️ LSL({d['LSL']})은 USL({d['USL']})보다 작아야 합니다"); st.stop()

    # σ_within 방법 선택
    method_options = {k: f"{v['label']} · {v['since']}"
                     for k, v in SIGMA_WITHIN_METHODS.items() if not k.startswith('mr-')}
    col_m, col_dl = st.columns([3, 1])
    with col_m:
        st.session_state.sigma_method = st.selectbox(
            "σ_within 계산법",
            options=list(method_options.keys()),
            format_func=lambda k: method_options[k],
            index=list(method_options.keys()).index(st.session_state.sigma_method),
        )

    cap = compute_capability(d['df'], d['sg_col'], d['var_col'], d['LSL'], d['USL'],
                            method=st.session_state.sigma_method)
    if 'error' in cap:
        st.error(cap['error']); st.stop()

    all_values = pd.to_numeric(d['df'][d['var_col']], errors='coerce').dropna().values
    norm = normality_check(all_values)

    with col_dl:
        st.write("")
        # 보고서 다운로드
        report = f"""
═══════════════════════════════════════════════
       공정능력 분석 보고서
═══════════════════════════════════════════════
생성일시: {time.strftime('%Y-%m-%d %H:%M:%S')}
데이터: {d['meta']['source']}
관측치: {len(all_values)}개 / 부분군: {cap['k']}개

─── 입력 정보 ────────────────────────────────
부분군 컬럼:  {d['sg_col']}
측정값 컬럼:  {d['var_col']}
LSL:          {d['LSL']}
USL:          {d['USL']}
목표값:       {d.get('target', '-')}
σ_within 계산법: {cap['methodInfo']['label']} ({cap['methodInfo']['since']})

─── 기초 통계량 ──────────────────────────────
전체 평균 X̄̄:  {cap['xBar']:.4f}
σ_within:     {cap['sigmaWithin']:.4f}  (군내변동)
σ_overall:    {cap['sigmaOverall']:.4f}  (군내+군간)
비율(o/w):    {cap['sigmaOverall']/cap['sigmaWithin']:.2f}

─── 공정능력 지수 ────────────────────────────
Cp  = {cap['Cp']:.4f}    ({cap['grade']['level']})
Cpk = {cap['Cpk']:.4f}    ({cap['gradeCpk']['level']})
Pp  = {cap['Pp']:.4f}
Ppk = {cap['Ppk']:.4f}    ({cap['gradePpk']['level']})

─── 판정 및 시정 조치 ────────────────────────
등급: {cap['grade']['grade']}등급 ({cap['grade']['level']}, {cap['grade']['sigma']})
시정 조치: {cap['grade']['action']}

─── 불량률 / PPM 추정 (정규분포 가정) ────────
PPM (단기, within):  {cap['ppmWithin']:,.0f}
PPM (장기, overall): {cap['ppmOverall']:,.0f}
불량률 (장기):       {cap['ppmOverall']/1e4:.4f}%

─── 정규성 검정 결과 ────────────────────────
{chr(10).join(f"  {t['name']:20s} p={t['pValue']:.4f}  {'✓ 만족' if t['passed'] else '✗ 불만족'}" for t in norm['tests'])}
종합 판단: {norm['note']}

═══════════════════════════════════════════════
Generated by Process Capability & SPC Studio
"""
        st.download_button("📄 보고서 다운로드", report,
                          file_name=f"capability-report-{int(time.time())}.txt",
                          mime='text/plain')

    # 정규성 경고 + Box-Cox 옵션
    if not norm['passed'] and (all_values > 0).all():
        st.warning(f"⚠️ 정규성 검정 불만족 (최소 p={norm['pValue']:.3f}). 정규성을 만족시키기 위해 **Box-Cox 변환** 후 분석이 정확합니다.")
        if st.button("📐 Box-Cox 변환 적용"):
            st.session_state.boxcox_on = True
        if st.session_state.get('boxcox_on'):
            bc = boxcox_transform(all_values)
            if 'error' not in bc:
                t_lsl = boxcox_value(d['LSL'], bc['lambda'])
                t_usl = boxcox_value(d['USL'], bc['lambda'])
                df_t = d['df'].copy()
                df_t[d['var_col']] = df_t[d['var_col']].apply(lambda v: boxcox_value(v, bc['lambda']))
                cap_t = compute_capability(df_t, d['sg_col'], d['var_col'], t_lsl, t_usl)
                norm_t = normality_check(bc['transformed'])

                st.success(f"✅ Box-Cox 적용: **λ = {bc['lambda']:.4f}** · {bc['interp']['name']} · {bc['interp']['desc']}")
                bc_cols = st.columns(4)
                bc_cols[0].metric("Cp 변환후", f"{cap_t['Cp']:.4f}", delta=f"{cap_t['Cp']-cap['Cp']:+.4f}")
                bc_cols[1].metric("Cpk 변환후", f"{cap_t['Cpk']:.4f}", delta=f"{cap_t['Cpk']-cap['Cpk']:+.4f}")
                bc_cols[2].metric("Pp 변환후", f"{cap_t['Pp']:.4f}", delta=f"{cap_t['Pp']-cap['Pp']:+.4f}")
                bc_cols[3].metric("Ppk 변환후", f"{cap_t['Ppk']:.4f}", delta=f"{cap_t['Ppk']-cap['Ppk']:+.4f}")
                st.caption(f"변환 후 정규성: {'만족 ✅' if norm_t['passed'] else '불만족'} (최소 p={norm_t['pValue']:.3f})")

                # 변환 전후 히스토그램 나란히 비교
                fig_bc = make_subplots(rows=1, cols=2,
                                       subplot_titles=(f'원본 데이터 (비대칭)',
                                                       f'Box-Cox 변환 후 (λ={bc["lambda"]:.2f})'))
                fig_bc.add_trace(go.Histogram(x=all_values, nbinsx=20, name='원본',
                                              marker_color='#ef4444', opacity=0.7), row=1, col=1)
                fig_bc.add_trace(go.Histogram(x=bc['transformed'], nbinsx=20, name='변환',
                                              marker_color='#3b82f6', opacity=0.7), row=1, col=2)
                fig_bc.update_layout(height=320, showlegend=False, title='Box-Cox 변환 전후 비교',
                                     margin=dict(l=40, r=20, t=60, b=30),
                                     bargap=0.05)
                st.plotly_chart(fig_bc, use_container_width=True)

                st.info("💡 공정능력지수는 무차원 비율이므로, 변환 후 Cpk = 1.33이면 원본에서도 동일하게 '공정능력이 양호'로 해석합니다.")
                if st.button("↩ Box-Cox 원본 복원"):
                    st.session_state.boxcox_on = False
                    st.rerun()

    # KPI 4개
    st.divider()
    st.subheader(f"공정능력지수 ({cap['methodInfo']['label']})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cp",  f"{cap['Cp']:.4f}",  help=cap['grade']['level'])
    c2.metric("Cpk", f"{cap['Cpk']:.4f}", help=cap['gradeCpk']['level'])
    c3.metric("Pp",  f"{cap['Pp']:.4f}")
    c4.metric("Ppk", f"{cap['Ppk']:.4f}", help=cap['gradePpk']['level'])

    # 5등급 판정
    g = cap['grade']
    st.markdown(f"""
<div style="padding: 12px; background: {g['color']}15; border-left: 4px solid {g['color']}; border-radius: 6px;">
<b style="color: {g['color']};">{g['grade']}등급 · {g['level']} ({g['sigma']})</b><br>
<span style="font-size: 13px; opacity: 0.9;">{g['action']}</span>
</div>
    """, unsafe_allow_html=True)

    # 자동 인사이트
    st.divider()
    st.subheader("🤖 자동 분석 인사이트")
    insights = interpret_capability(cap, norm, d.get('target'))
    render_insights(insights)

    # 히스토그램 + 정규분포 적합
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("히스토그램 + 정규분포 적합")
        bin_df, edges = (lambda: __import__('lib.stats_lib').stats_lib.histogram_bins(all_values, 20))()
        from lib.stats_lib import fit_normal_pdf, histogram_bins
        bin_df, _ = histogram_bins(all_values, 20)
        bin_w = bin_df['x'].diff().mean() if len(bin_df) > 1 else 1.0
        pdf_df = fit_normal_pdf(all_values, x_range=(float(min(d['LSL'], all_values.min())*0.99),
                                                     float(max(d['USL'], all_values.max())*1.01)))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_df['x'], y=bin_df['count'], name='관측값', marker_color='#93c5fd', opacity=0.7,
                            width=bin_w))
        fig.add_trace(go.Scatter(x=pdf_df['x'], y=pdf_df['pdf']*len(all_values)*bin_w,
                                mode='lines', name='정규분포 적합', line=dict(color='#3b82f6', width=2)))
        fig.add_vline(x=d['LSL'], line_color='#ef4444', line_dash='dash',
                     annotation_text=f"LSL={d['LSL']}", annotation_position="top")
        fig.add_vline(x=d['USL'], line_color='#ef4444', line_dash='dash',
                     annotation_text=f"USL={d['USL']}", annotation_position="top")
        if d.get('target'):
            fig.add_vline(x=d['target'], line_color='#10b981', annotation_text=f"target={d['target']}",
                         annotation_position="top")
        fig.update_layout(height=350, margin=dict(l=40, r=20, t=10, b=30), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Q-Q Plot · 정규성 검정 3종")
        qq = qq_plot_data(all_values)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=qq['theoretical'], y=qq['sample'], mode='markers',
                                marker=dict(color='#3b82f6', size=6)))
        fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines', line=dict(color='#ef4444', width=1.5)))
        fig.update_layout(height=250, margin=dict(l=40, r=20, t=10, b=30), showlegend=False,
                         xaxis_title='이론분위수', yaxis_title='표본분위수')
        st.plotly_chart(fig, use_container_width=True)
        for t in norm['tests']:
            badge = "✅ 만족" if t['passed'] else "⚠️ 불만족"
            st.write(f"**{t['name']}**: p = `{t['pValue']:.4f}` · {badge}")
        st.caption(norm['note'])

    # 단기 vs 장기 분포 비교
    st.divider()
    st.subheader("단기 vs 장기 공정능력 분포 비교")
    sigma_max = max(cap['sigmaWithin'], cap['sigmaOverall'])
    x_lo = min(d['LSL'], cap['xBar'] - 4*sigma_max)
    x_hi = max(d['USL'], cap['xBar'] + 4*sigma_max)
    x_pts = np.linspace(x_lo, x_hi, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_pts, y=sp_stats.norm.pdf(x_pts, cap['xBar'], cap['sigmaOverall']),
                            fill='tozeroy', line=dict(color='#a855f7'), name=f'장기 (Ppk={cap["Ppk"]:.3f})'))
    fig.add_trace(go.Scatter(x=x_pts, y=sp_stats.norm.pdf(x_pts, cap['xBar'], cap['sigmaWithin']),
                            fill='tozeroy', line=dict(color='#3b82f6'), name=f'단기 (Cpk={cap["Cpk"]:.3f})'))
    fig.add_vline(x=d['LSL'], line_color='#ef4444', line_dash='dash',
                 annotation_text=f"LSL={d['LSL']}")
    fig.add_vline(x=d['USL'], line_color='#ef4444', line_dash='dash',
                 annotation_text=f"USL={d['USL']}")
    fig.add_vline(x=cap['xBar'], line_color='#10b981', annotation_text='X̄̄')
    fig.update_layout(height=320, margin=dict(l=40, r=20, t=10, b=30))
    st.plotly_chart(fig, use_container_width=True)

    ratio = cap['sigmaOverall'] / cap['sigmaWithin']
    if ratio > 1.5:   st.error(f"σ_overall / σ_within = **{ratio:.2f}** → 군간변동이 매우 큼 (외부 요인 강함)")
    elif ratio > 1.2: st.warning(f"σ_overall / σ_within = **{ratio:.2f}** → 군간변동 다소 큼")
    else:             st.success(f"σ_overall / σ_within = **{ratio:.2f}** → 일관성 있는 공정")

    # ── 공정능력 종합 시각화 (stacked histogram + marginal box + Cp/Cpk 어노테이션)
    st.divider()
    st.subheader("공정능력 분석 종합 시각화")
    st.caption("부분군별 색 구분된 히스토그램 + 위쪽 박스플롯 + LSL/USL/Cp/Cpk 표시 — 모든 분석을 한 차트에서")

    fig_combo = px.histogram(
        d['df'],
        x=d['var_col'],
        color=d['sg_col'],
        nbins=20,
        marginal='box',
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f"{d['var_col'].capitalize()} Process Capability Analysis",
    )
    fig_combo.add_vline(x=d['LSL'], line_width=1.5, line_dash='dash', line_color='#ef4444',
                        annotation_text=f"LSL={d['LSL']}", annotation_position='top')
    fig_combo.add_vline(x=d['USL'], line_width=1.5, line_dash='dash', line_color='#ef4444',
                        annotation_text=f"USL={d['USL']}", annotation_position='top')
    if d.get('target'):
        fig_combo.add_vline(x=d['target'], line_width=1, line_color='#10b981',
                            annotation_text=f"target={d['target']}", annotation_position='top')

    # 우측 하단에 Cp/Cpk/Pp/Ppk 어노테이션 (강의록 스타일)
    fig_combo.add_annotation(
        xref='paper', yref='paper',
        x=1.0, y=0.0, xanchor='right', yanchor='bottom',
        text=(f"<b>Cp  = {cap['Cp']:.4f}</b><br>"
              f"<b>Cpk = {cap['Cpk']:.4f}</b><br>"
              f"Pp  = {cap['Pp']:.4f}<br>"
              f"Ppk = {cap['Ppk']:.4f}"),
        showarrow=False,
        bordercolor='#475569', borderwidth=1, borderpad=8,
        bgcolor='rgba(255,255,255,0.95)', font=dict(size=12, family='JetBrains Mono'),
    )
    fig_combo.update_layout(height=520, margin=dict(l=40, r=80, t=60, b=30),
                            legend_title_text=d['sg_col'])
    st.plotly_chart(fig_combo, use_container_width=True)

    # ── 부분군별 박스플롯 (별도)
    st.divider()
    st.subheader("부분군별 박스플롯")
    fig = px.box(d['df'], x=d['sg_col'], y=d['var_col'], points='all',
                color=d['sg_col'], color_discrete_sequence=px.colors.qualitative.Set2)
    fig.add_hline(y=d['LSL'], line_color='#ef4444', line_dash='dash', annotation_text='LSL')
    fig.add_hline(y=d['USL'], line_color='#ef4444', line_dash='dash', annotation_text='USL')
    fig.update_layout(height=400, showlegend=False, margin=dict(l=40, r=20, t=10, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # ── 부분군별 facet 히스토그램
    st.divider()
    st.subheader("부분군별 히스토그램 (facet)")
    st.caption("각 부분군을 분리해 보기. 부분군별 분포 차이 / 규격 이탈 부분군을 즉시 파악")
    n_sg = d['df'][d['sg_col']].nunique()
    facet_height = max(120 * n_sg, 360)
    fig_facet = px.histogram(
        d['df'], x=d['var_col'], facet_row=d['sg_col'],
        nbins=20, opacity=0.75,
        color_discrete_sequence=['#6366f1'],
        title=f"{d['var_col'].capitalize()} Histogram by {d['sg_col']}",
    )
    fig_facet.add_vline(x=d['LSL'], line_dash='dash', line_color='#ef4444',
                        annotation_text='LSL', line_width=1)
    fig_facet.add_vline(x=d['USL'], line_dash='dash', line_color='#ef4444',
                        annotation_text='USL', line_width=1)
    fig_facet.update_layout(height=facet_height, showlegend=False,
                            margin=dict(l=40, r=20, t=50, b=30))
    fig_facet.update_yaxes(matches=None)  # 각 facet별 y축 독립
    st.plotly_chart(fig_facet, use_container_width=True)

    # 부분군별 통계표
    st.subheader("부분군별 통계")
    sg_table = cap['subgroupStats'].reset_index()
    sg_table['규격 외'] = sg_table.apply(lambda r: '', axis=1)
    for i, row in sg_table.iterrows():
        sg_vals = d['df'][d['df'][d['sg_col']] == row[d['sg_col']]][d['var_col']]
        oob = ((sg_vals < d['LSL']) | (sg_vals > d['USL'])).sum()
        sg_table.at[i, '규격 외'] = f"⚠️ {oob}개" if oob > 0 else "✅ 0"
    st.dataframe(sg_table.round(4), use_container_width=True, height=300)


# ────────────────────────────────────────────────────────────────────
# 페이지 4: SPC 관리도
# ────────────────────────────────────────────────────────────────────
elif page == '④ SPC 관리도':
    page_header(
        "SPC 관리도",
        hint="계량형 5종 (X̄-R · X̄-s · I-MR · CUSUM · EWMA) + 계수형 4종 (NP · P · C · U). Nelson Rules 8가지 자동 감지."
    )
    d = st.session_state.dataset
    if d is None:
        st.markdown("""
<div class="action-banner">
  <div class="icon">📂</div>
  <div class="text">
    <strong>데이터가 아직 없습니다.</strong> 좌측 메뉴에서 <strong>'① 데이터'</strong>를 선택해 예제 또는 CSV를 불러와 주세요.
  </div>
</div>
        """, unsafe_allow_html=True)
        st.stop()

    # 관리도 종류 선택
    if d['mode'] == 'variable':
        chart_options = ['X̄-R', 'X̄-s', 'I-MR', 'CUSUM', 'EWMA']
    else:
        chart_options = ['NP', 'P', 'C', 'U']

    col1, col2, col3 = st.columns([3, 1, 2])
    with col1:
        chart_type = st.selectbox("관리도 종류", chart_options)
    with col2:
        window = 2
        if chart_type == 'I-MR':
            window = st.number_input("윈도우 w", value=2, min_value=2, max_value=10)

    # 관리도 가이드
    with st.expander("📖 관리도 선택 가이드"):
        gc1, gc2 = st.columns(2)
        gc1.info("**계량형 (continuous)**\n- 부분군 1 → I-MR\n- 부분군 2~10 → X̄-R\n- 부분군 10+ → X̄-s\n- 시간 가중 → CUSUM/EWMA")
        gc2.info("**계수형 (attribute)**\n- 불량개수, n 동일 → NP\n- 불량률, n 가변 → P\n- 결점수, n 동일 → C\n- 결점률, n 가변 → U")

    # 관리도 계산
    try:
        if d['mode'] == 'variable':
            if chart_type == 'X̄-R':
                chart_result = xbar_r_chart(d['df'], d['sg_col'], d['var_col'])
            elif chart_type == 'X̄-s':
                chart_result = xbar_s_chart(d['df'], d['sg_col'], d['var_col'])
            elif chart_type == 'I-MR':
                chart_result = imr_chart(d['df'], d['sg_col'], d['var_col'], window=window)
            elif chart_type == 'CUSUM':
                chart_result = cusum_chart(d['df'], d['sg_col'], d['var_col'], mu0=d.get('target'))
            elif chart_type == 'EWMA':
                chart_result = ewma_chart(d['df'], d['sg_col'], d['var_col'], mu0=d.get('target'))
        else:
            if chart_type == 'NP':
                chart_result = np_chart(d['df'], d['sg_col'], d['size_col'], d['defect_col'])
            elif chart_type == 'P':
                chart_result = p_chart(d['df'], d['sg_col'], d['size_col'], d['defect_col'])
            elif chart_type == 'C':
                chart_result = c_chart(d['df'], d['sg_col'], d['defect_col'])
            elif chart_type == 'U':
                chart_result = u_chart(d['df'], d['sg_col'], d['size_col'], d['defect_col'])
    except Exception as e:
        st.error(f"관리도 생성 실패: {e}"); st.stop()

    # Nelson Rules 적용
    checked = apply_nelson_rules(chart_result['primary'])
    violations_df = checked[checked['violations'].apply(lambda v: len(v) > 0)]

    # 이상치 제거 액션 — Before/After 비교 + 명시적 채택
    with col3:
        st.write("")
        if len(violations_df) > 0:
            outlier_sgs = violations_df['sg'].tolist()
            if st.button(f"🔍 이상치 제거 시뮬레이션 ({len(outlier_sgs)}개)", type='primary',
                        help="이상 부분군을 제외하고 재계산하여 한계 변화 비교"):
                st.session_state.outlier_preview = outlier_sgs

    # Before/After 비교 모드
    preview_outliers = st.session_state.get('outlier_preview', [])
    if preview_outliers and len(preview_outliers) > 0:
        st.divider()
        st.subheader(f"🔄 1단계 → 2단계: 이상 부분군 제거 후 재작성")
        st.caption(f"제외할 부분군: {', '.join(map(str, preview_outliers))}  ·  Shewhart 표준 절차 (이상원인 제거 → 한계 재계산)")

        # 제거 후 데이터로 다시 계산
        df_cleaned = d['df'][~d['df'][d['sg_col']].astype(str).isin(preview_outliers)].copy()

        try:
            if chart_type == 'X̄-R':
                chart_after = xbar_r_chart(df_cleaned, d['sg_col'], d['var_col'])
            elif chart_type == 'X̄-s':
                chart_after = xbar_s_chart(df_cleaned, d['sg_col'], d['var_col'])
            elif chart_type == 'I-MR':
                chart_after = imr_chart(df_cleaned, d['sg_col'], d['var_col'], window=window)
            elif chart_type == 'CUSUM':
                chart_after = cusum_chart(df_cleaned, d['sg_col'], d['var_col'], mu0=d.get('target'))
            elif chart_type == 'EWMA':
                chart_after = ewma_chart(df_cleaned, d['sg_col'], d['var_col'], mu0=d.get('target'))
            elif chart_type == 'NP':
                chart_after = np_chart(df_cleaned, d['sg_col'], d['size_col'], d['defect_col'])
            elif chart_type == 'P':
                chart_after = p_chart(df_cleaned, d['sg_col'], d['size_col'], d['defect_col'])
            elif chart_type == 'C':
                chart_after = c_chart(df_cleaned, d['sg_col'], d['defect_col'])
            elif chart_type == 'U':
                chart_after = u_chart(df_cleaned, d['sg_col'], d['size_col'], d['defect_col'])

            checked_after = apply_nelson_rules(chart_after['primary'])

            # 관리한계 변화 표
            UCL_before = float(chart_result['primary']['UCL'].iloc[0])
            LCL_before = float(chart_result['primary']['LCL'].iloc[0])
            CL_before  = float(chart_result['primary']['CL'].iloc[0])
            UCL_after  = float(chart_after['primary']['UCL'].iloc[0])
            LCL_after  = float(chart_after['primary']['LCL'].iloc[0])
            CL_after   = float(chart_after['primary']['CL'].iloc[0])

            limit_df = pd.DataFrame({
                '한계': ['UCL (상한)', 'CL (중심선)', 'LCL (하한)', '한계 폭 (UCL-LCL)'],
                '초기 (Before)': [f"{UCL_before:.4f}", f"{CL_before:.4f}", f"{LCL_before:.4f}", f"{UCL_before-LCL_before:.4f}"],
                '재작성 (After)': [f"{UCL_after:.4f}", f"{CL_after:.4f}", f"{LCL_after:.4f}", f"{UCL_after-LCL_after:.4f}"],
                '변화': [f"{UCL_after-UCL_before:+.4f}", f"{CL_after-CL_before:+.4f}", f"{LCL_after-LCL_before:+.4f}",
                       f"{(UCL_after-LCL_after) - (UCL_before-LCL_before):+.4f}"],
            })
            st.markdown("**📊 관리한계 변화 비교**")
            st.dataframe(limit_df, use_container_width=True, hide_index=True)

            width_change = (UCL_after - LCL_after) - (UCL_before - LCL_before)
            if width_change < 0:
                st.success(f"✅ 관리한계 폭이 **{abs(width_change):.4f}만큼 좁아졌습니다**. 이상원인 제거 효과가 확인되었습니다 (기존 정상점이 새 이상점이 될 수 있으므로 추가 점검 필요).")
            else:
                st.info(f"관리한계 폭 변화: {width_change:+.4f}")

            # 두 차트를 위/아래로 나란히 표시
            st.markdown("**📈 1단계 (Before)** — 모든 부분군 포함")
            fig_before = draw_control_chart(chart_result, checked=checked, title='')
            fig_before.update_layout(height=400, title='')
            st.plotly_chart(fig_before, use_container_width=True)

            st.markdown(f"**📈 2단계 (After)** — 이상 부분군 {len(preview_outliers)}개 제거")
            fig_after = draw_control_chart(chart_after, checked=checked_after, title='')
            fig_after.update_layout(height=400, title='')
            st.plotly_chart(fig_after, use_container_width=True)

            # 액션 버튼
            ac1, ac2 = st.columns(2)
            with ac1:
                if st.button("✅ 재작성 결과 채택 (데이터 영구 제거)", type='primary', use_container_width=True):
                    d['df'] = df_cleaned
                    st.session_state.outlier_preview = []
                    st.success(f"✅ {len(preview_outliers)}개 부분군 영구 제거됨")
                    st.rerun()
            with ac2:
                if st.button("↩ 취소", use_container_width=True):
                    st.session_state.outlier_preview = []
                    st.rerun()

        except Exception as e:
            st.error(f"재작성 실패: {e}")
            st.session_state.outlier_preview = []

    # KPI
    st.divider()
    primary = chart_result['primary']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CL", f"{primary['CL'].iloc[0]:.4f}")
    c2.metric("UCL", f"{primary['UCL'].iloc[0]:.4f}")
    c3.metric("LCL", f"{primary['LCL'].iloc[0]:.4f}")
    c4.metric("위반점", len(violations_df), delta=None,
              delta_color='inverse' if len(violations_df) > 0 else 'off')

    # 자동 인사이트
    st.subheader("🤖 자동 분석 인사이트")
    insights = interpret_spc(checked, chart_type)
    render_insights(insights)

    # 차트
    st.divider()
    fig = draw_control_chart(chart_result, checked=checked, title=f'{chart_type} 관리도')
    st.plotly_chart(fig, use_container_width=True)

    # Nelson Rule 8가지 (always 표시)
    with st.expander("📜 Nelson's Rule 8가지"):
        for r, txt in NELSON_RULES_TEXT.items():
            in_use = "🔴" if (r in [v for vs in checked['violations'] for v in vs]) else "⚪"
            st.write(f"{in_use} **Rule {r}**: {txt}")

    # 위반점 상세 표
    if len(violations_df) > 0:
        st.divider()
        st.subheader(f"위반점 상세 ({len(violations_df)}개)")
        show_df = violations_df[['sg', 'point', 'violations']].copy()
        show_df['violations'] = show_df['violations'].apply(lambda v: ', '.join(f"R{r}" for r in v))
        st.dataframe(show_df, use_container_width=True)


# ────────────────────────────────────────────────────────────────────
# 페이지 5: 데이터셋 비교
# ────────────────────────────────────────────────────────────────────
elif page == '⑤ 데이터셋 비교':
    page_header(
        "데이터셋 비교",
        hint="두 데이터셋의 공정능력을 나란히 비교합니다. \"개선 전 vs 개선 후\" 또는 \"라인 A vs 라인 B\" 시나리오에 활용.",
        badge="NEW"
    )

    # 데이터셋 옵션
    options = []
    if st.session_state.dataset is not None:
        options.append(('current', f"★ 현재 분석 중 ({st.session_state.dataset['meta']['source']})"))
    for ex in EXAMPLE_DATASETS:
        options.append((ex['id'], f"{ex['label']} · {ex['source']}"))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔵 데이터셋 A (왼쪽)")
        left_id = st.selectbox("A 선택", [o[0] for o in options],
                              format_func=lambda i: dict(options)[i], key='left_id')
    with c2:
        st.markdown("### 🟣 데이터셋 B (오른쪽)")
        right_id = st.selectbox("B 선택", [o[0] for o in options],
                               format_func=lambda i: dict(options)[i], key='right_id',
                               index=1 if len(options) > 1 else 0)

    def get_dataset(idx):
        if idx == 'current':
            return st.session_state.dataset
        ex = next((e for e in EXAMPLE_DATASETS if e['id'] == idx), None)
        if ex:
            data = ex['loader']()
            return {
                'mode': data['mode'], 'df': data['df'],
                'sg_col': data['sg_col'], 'var_col': data.get('var_col'),
                'LSL': data.get('LSL'), 'USL': data.get('USL'),
                'meta': {'source': data['name']},
            }
        return None

    left = get_dataset(left_id)
    right = get_dataset(right_id)

    if left is None or right is None or left['mode'] != 'variable' or right['mode'] != 'variable':
        st.warning("⚠️ 양쪽 모두 계량형 데이터여야 합니다"); st.stop()
    if left['LSL'] is None or right['LSL'] is None:
        st.warning("⚠️ 양쪽 모두 LSL/USL이 설정되어야 합니다"); st.stop()

    left_cap = compute_capability(left['df'], left['sg_col'], left['var_col'], left['LSL'], left['USL'])
    right_cap = compute_capability(right['df'], right['sg_col'], right['var_col'], right['LSL'], right['USL'])

    left_vals = pd.to_numeric(left['df'][left['var_col']], errors='coerce').dropna().values
    right_vals = pd.to_numeric(right['df'][right['var_col']], errors='coerce').dropna().values
    left_norm = normality_check(left_vals)
    right_norm = normality_check(right_vals)

    # 헤더 카드
    st.divider()
    c1, c2 = st.columns(2)
    c1.info(f"**🔵 A: {left['meta']['source']}**\n\n"
           f"{len(left_vals)} obs · {left_cap['k']} subgroup\n\n"
           f"LSL={left['LSL']}, USL={left['USL']}")
    c2.info(f"**🟣 B: {right['meta']['source']}**\n\n"
           f"{len(right_vals)} obs · {right_cap['k']} subgroup\n\n"
           f"LSL={right['LSL']}, USL={right['USL']}")

    # 4개 지수 가로 막대 비교
    st.subheader("공정능력 지수 비교")
    for key in ['Cp', 'Cpk', 'Pp', 'Ppk']:
        l = left_cap[key]; r = right_cap[key]
        cl, cc, cr = st.columns([2, 5, 2])
        l_bold = "**" if l > r else ""
        r_bold = "**" if r > l else ""
        cl.markdown(f"<div style='text-align:right;color:#3b82f6'>{l_bold}{l:.4f}{l_bold}</div>", unsafe_allow_html=True)
        with cc:
            # 진행 바
            max_v = max(l, r, 1.67) * 1.2
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[l], y=[''], orientation='h', marker_color='#3b82f6',
                                opacity=1.0 if l > r else 0.4, name='A',
                                hovertemplate=f'A: {l:.4f}<extra></extra>'))
            fig.add_trace(go.Bar(x=[r], y=[''], orientation='h', marker_color='#a855f7',
                                opacity=1.0 if r > l else 0.4, name='B',
                                hovertemplate=f'B: {r:.4f}<extra></extra>'))
            fig.update_layout(barmode='group', height=50, showlegend=False,
                            margin=dict(l=0, r=0, t=2, b=2),
                            xaxis=dict(range=[0, max_v], showgrid=False, zeroline=False),
                            yaxis=dict(showticklabels=False),
                            title=dict(text=key, x=0.5, font=dict(size=12)))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        cr.markdown(f"<div style='color:#a855f7'>{r_bold}{r:.4f}{r_bold}</div>", unsafe_allow_html=True)

    # 상세 통계 비교표
    st.divider()
    st.subheader("상세 통계량 비교")
    comp_df = pd.DataFrame({
        '지표': ['평균 X̄', 'σ_within', 'σ_overall', 'Cp', 'Cpk', 'Pp', 'Ppk', 'PPM 추정', '정규성 (min p)'],
        'A (왼쪽)': [
            f"{left_cap['xBar']:.4f}", f"{left_cap['sigmaWithin']:.4f}", f"{left_cap['sigmaOverall']:.4f}",
            f"{left_cap['Cp']:.4f}", f"{left_cap['Cpk']:.4f}", f"{left_cap['Pp']:.4f}", f"{left_cap['Ppk']:.4f}",
            f"{left_cap['ppmOverall']:,.0f}", f"{left_norm['pValue']:.3f}"
        ],
        'B (오른쪽)': [
            f"{right_cap['xBar']:.4f}", f"{right_cap['sigmaWithin']:.4f}", f"{right_cap['sigmaOverall']:.4f}",
            f"{right_cap['Cp']:.4f}", f"{right_cap['Cpk']:.4f}", f"{right_cap['Pp']:.4f}", f"{right_cap['Ppk']:.4f}",
            f"{right_cap['ppmOverall']:,.0f}", f"{right_norm['pValue']:.3f}"
        ],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # 분포 겹쳐 그리기
    st.subheader("분포 비교 (정규분포 적합)")
    lo = min(left['LSL'], right['LSL'], left_cap['xBar']-4*left_cap['sigmaOverall'], right_cap['xBar']-4*right_cap['sigmaOverall'])
    hi = max(left['USL'], right['USL'], left_cap['xBar']+4*left_cap['sigmaOverall'], right_cap['xBar']+4*right_cap['sigmaOverall'])
    x_pts = np.linspace(lo, hi, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_pts, y=sp_stats.norm.pdf(x_pts, left_cap['xBar'], left_cap['sigmaOverall']),
                            fill='tozeroy', line=dict(color='#3b82f6'), name=f"A: {left['meta']['source']}"))
    fig.add_trace(go.Scatter(x=x_pts, y=sp_stats.norm.pdf(x_pts, right_cap['xBar'], right_cap['sigmaOverall']),
                            fill='tozeroy', line=dict(color='#a855f7'), name=f"B: {right['meta']['source']}",
                            opacity=0.7))
    fig.add_vline(x=left['LSL'], line_color='#3b82f6', line_dash='dash', annotation_text='A LSL')
    fig.add_vline(x=left['USL'], line_color='#3b82f6', line_dash='dash', annotation_text='A USL')
    fig.add_vline(x=right['LSL'], line_color='#a855f7', line_dash='dot', annotation_text='B LSL')
    fig.add_vline(x=right['USL'], line_color='#a855f7', line_dash='dot', annotation_text='B USL')
    fig.update_layout(height=400, margin=dict(l=40, r=20, t=20, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # 자동 비교 결론
    st.divider()
    st.subheader("🤖 자동 비교 결론")
    if left_cap['Cpk'] > right_cap['Cpk']:
        winner = ('A', '🔵', left['meta']['source'])
    elif right_cap['Cpk'] > left_cap['Cpk']:
        winner = ('B', '🟣', right['meta']['source'])
    else:
        winner = None

    if winner:
        diff = abs(left_cap['Cpk'] - right_cap['Cpk'])
        ppm_diff = abs(left_cap['ppmOverall'] - right_cap['ppmOverall'])
        st.success(f"### 🏆 {winner[1]} {winner[0]}가 더 우수\n"
                  f"**{winner[2]}**의 Cpk가 `{max(left_cap['Cpk'], right_cap['Cpk']):.3f}`로 "
                  f"상대보다 `{diff:.3f}`만큼 큽니다. PPM 차이 약 `{ppm_diff:,.0f}`.")
    else:
        st.info("두 데이터셋의 Cpk가 동등합니다.")


# ────────────────────────────────────────────────────────────────────
# 페이지 6: What-if 시뮬레이션
# ────────────────────────────────────────────────────────────────────
elif page == '⑥ What-if 시뮬':
    page_header(
        "What-if 시뮬레이션",
        hint="μ · σ_within · σ_overall · LSL · USL 슬라이더를 움직여 Cp/Cpk가 실시간으로 변화하는 모습을 직접 체험해 보세요."
    )

    # 기본값 (현재 데이터셋에서 추출)
    d = st.session_state.dataset
    if d and d['mode'] == 'variable' and d.get('LSL') is not None:
        all_v = pd.to_numeric(d['df'][d['var_col']], errors='coerce').dropna().values
        default_mu = float(np.mean(all_v))
        default_sigma_o = float(np.std(all_v, ddof=1))
        default_sigma_w = default_sigma_o * 0.85
        default_lsl = float(d['LSL'])
        default_usl = float(d['USL'])
    else:
        default_mu, default_sigma_w, default_sigma_o = 100.0, 2.5, 3.0
        default_lsl, default_usl = 90.0, 110.0

    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("🎚️ 파라미터 조정")
        lsl = st.slider("LSL (규격하한)", default_mu - 30.0, default_mu - 0.1, default_lsl, 0.1)
        usl = st.slider("USL (규격상한)", default_mu + 0.1, default_mu + 30.0, default_usl, 0.1)
        mu = st.slider("평균 μ", lsl - 5.0, usl + 5.0, default_mu, 0.1)
        sigma_w = st.slider("σ_within (단기)", 0.1, (usl - lsl) / 3.0, default_sigma_w, 0.05)
        sigma_o = st.slider("σ_overall (장기)", sigma_w, (usl - lsl) / 2.0, max(default_sigma_o, sigma_w), 0.05)

        # 실시간 계산
        Cp = (usl - lsl) / (6 * sigma_w)
        Cpk = min((usl - mu) / (3 * sigma_w), (mu - lsl) / (3 * sigma_w))
        Pp = (usl - lsl) / (6 * sigma_o)
        Ppk = min((usl - mu) / (3 * sigma_o), (mu - lsl) / (3 * sigma_o))
        ppm_o = (sp_stats.norm.cdf(lsl, mu, sigma_o) + (1 - sp_stats.norm.cdf(usl, mu, sigma_o))) * 1e6
        g_cp = grade_capability(Cp)
        g_cpk = grade_capability(Cpk)

        kc1, kc2 = st.columns(2)
        kc1.metric("Cp",  f"{Cp:.4f}",  help=g_cp['level'])
        kc2.metric("Cpk", f"{Cpk:.4f}", help=g_cpk['level'])
        kc3, kc4 = st.columns(2)
        kc3.metric("Pp",  f"{Pp:.4f}")
        kc4.metric("Ppk", f"{Ppk:.4f}")
        st.metric("예상 PPM (장기)", f"{ppm_o:,.0f}")

        if Cpk >= 1.67:   st.success("🌟 매우 우수 — 비용 절감 검토 가능")
        elif Cpk >= 1.33: st.success("✅ 충분 — 현재 유지")
        elif Cpk >= 1.0:  st.warning("⚠️ 경계 — 산포 축소 권장")
        else:             st.error("🚨 부족 — 시급한 개선 필요")

    with c2:
        st.subheader("분포 시각화")
        x_lo = min(lsl, mu - 4*max(sigma_w, sigma_o))
        x_hi = max(usl, mu + 4*max(sigma_w, sigma_o))
        x_pts = np.linspace(x_lo, x_hi, 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_pts, y=sp_stats.norm.pdf(x_pts, mu, sigma_o),
                                fill='tozeroy', line=dict(color='#a855f7', width=1.5),
                                name=f'장기 σ_o={sigma_o:.2f}'))
        fig.add_trace(go.Scatter(x=x_pts, y=sp_stats.norm.pdf(x_pts, mu, sigma_w),
                                fill='tozeroy', line=dict(color='#3b82f6', width=1.5),
                                name=f'단기 σ_w={sigma_w:.2f}'))
        fig.add_vline(x=lsl, line_color='#ef4444', line_dash='dash', annotation_text=f"LSL={lsl:.2f}")
        fig.add_vline(x=usl, line_color='#ef4444', line_dash='dash', annotation_text=f"USL={usl:.2f}")
        fig.add_vline(x=mu, line_color='#10b981', annotation_text=f"μ={mu:.2f}")
        fig.update_layout(height=450, margin=dict(l=40, r=20, t=20, b=30),
                         legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"💡 슬라이더를 움직여 보세요. 예를 들어 σ_within을 절반으로 줄이면 Cp/Cpk가 2배로 커지는 것을 직접 확인할 수 있습니다.")


# ────────────────────────────────────────────────────────────────────
# 페이지 7: LIVE 모드
# ────────────────────────────────────────────────────────────────────
elif page == '⑦ 🔴 LIVE 모드':
    page_header(
        "LIVE 공정 모니터링 시뮬레이션",
        hint="실제 공장에서 데이터가 실시간으로 들어오는 것처럼 시연합니다. 시나리오 버튼으로 외란을 발생시켜 보세요.",
        live=True
    )

    # 파라미터 + 컨트롤
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("⚙️ 파라미터")
        target = st.number_input("목표값 μ₀", value=100.0, key='live_t')
        sigma = st.number_input("기본 산포 σ", value=2.0, min_value=0.1, key='live_s')
        lsl = st.number_input("LSL", value=target - 6, key='live_lsl')
        usl = st.number_input("USL", value=target + 6, key='live_usl')
        n_samples = st.slider("측정 횟수", 10, 100, 30, key='live_n')

        st.subheader("🎬 시나리오")
        scenario = st.radio("시나리오", ['정상 운영', '평균 시프트', '산포 증가', '🚨 심각 이상'],
                          label_visibility='collapsed', key='live_scen')

        shift, sigma_scale = 0, 1
        if scenario == '평균 시프트':
            shift, sigma_scale = sigma, 1
        elif scenario == '산포 증가':
            shift, sigma_scale = 0, 1.8
        elif scenario == '🚨 심각 이상':
            shift, sigma_scale = 2 * sigma, 2

        col_run, col_clr = st.columns(2)
        with col_run:
            run_button = st.button("▶ 시뮬레이션 시작", type='primary', use_container_width=True)
        with col_clr:
            if st.button("🗑 초기화", use_container_width=True):
                st.session_state.live_points = []
                st.rerun()

    with c2:
        st.subheader("📊 실시간 차트")

        chart_placeholder = st.empty()
        kpi_placeholder = st.empty()
        toast_placeholder = st.empty()

        if run_button:
            np.random.seed(int(time.time()) % 10000)
            collected = []
            for i in range(n_samples):
                # 새 측정값
                value = target + shift + np.random.normal(0, sigma * sigma_scale)
                collected.append({'t': i + 1, 'value': value})

                # 차트 업데이트
                with chart_placeholder.container():
                    df_live = pd.DataFrame(collected)
                    is_oos = (df_live['value'] < lsl) | (df_live['value'] > usl)
                    colors = ['#ef4444' if oos else '#3b82f6' for oos in is_oos]
                    sizes = [10 if oos else 6 for oos in is_oos]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_live['t'], y=df_live['value'],
                                            mode='lines+markers',
                                            line=dict(color='#3b82f6', width=1.5),
                                            marker=dict(color=colors, size=sizes,
                                                       line=dict(color='white', width=1))))
                    fig.add_hline(y=usl, line_color='#ef4444', line_dash='dash',
                                 annotation_text=f"USL={usl}")
                    fig.add_hline(y=lsl, line_color='#ef4444', line_dash='dash',
                                 annotation_text=f"LSL={lsl}")
                    fig.add_hline(y=target, line_color='#10b981',
                                 annotation_text=f"target={target}")
                    fig.update_layout(height=400, margin=dict(l=40, r=80, t=20, b=30),
                                     showlegend=False,
                                     xaxis_title='측정 시점',
                                     plot_bgcolor='rgba(248,250,252,0.5)')
                    st.plotly_chart(fig, use_container_width=True)

                # KPI 업데이트
                with kpi_placeholder.container():
                    df_live = pd.DataFrame(collected)
                    xbar_live = df_live['value'].mean()
                    sigma_live = df_live['value'].std(ddof=1) if len(df_live) > 1 else 0
                    oos_count = ((df_live['value'] < lsl) | (df_live['value'] > usl)).sum()
                    if sigma_live > 0:
                        cpk_live = min((usl - xbar_live) / (3 * sigma_live),
                                      (xbar_live - lsl) / (3 * sigma_live))
                    else:
                        cpk_live = 0
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("X̄ 실시간", f"{xbar_live:.3f}")
                    k2.metric("Cpk 실시간", f"{cpk_live:.3f}")
                    k3.metric("규격 외", oos_count, delta_color='inverse')
                    k4.metric("측정 #", i + 1)

                time.sleep(0.15)  # 부드러운 애니메이션

            # 완료 알림
            st.session_state.live_points = collected
            st.success(f"✅ 시뮬레이션 완료 ({n_samples}개 측정)")
            if scenario != '정상 운영':
                st.toast(f"🎬 시나리오 '{scenario}' 적용됨", icon='⚠️')

        elif st.session_state.live_points:
            # 이전 결과 표시
            df_live = pd.DataFrame(st.session_state.live_points)
            is_oos = (df_live['value'] < lsl) | (df_live['value'] > usl)
            colors = ['#ef4444' if oos else '#3b82f6' for oos in is_oos]
            sizes = [10 if oos else 6 for oos in is_oos]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_live['t'], y=df_live['value'], mode='lines+markers',
                                    line=dict(color='#3b82f6', width=1.5),
                                    marker=dict(color=colors, size=sizes, line=dict(color='white', width=1))))
            fig.add_hline(y=usl, line_color='#ef4444', line_dash='dash', annotation_text=f"USL={usl}")
            fig.add_hline(y=lsl, line_color='#ef4444', line_dash='dash', annotation_text=f"LSL={lsl}")
            fig.add_hline(y=target, line_color='#10b981', annotation_text=f"target={target}")
            fig.update_layout(height=400, margin=dict(l=40, r=80, t=20, b=30), showlegend=False,
                            xaxis_title='측정 시점', plot_bgcolor='rgba(248,250,252,0.5)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("▶ **시뮬레이션 시작** 버튼을 눌러 LIVE 데이터를 발생시키세요. 시나리오를 바꿔가며 공정 변화에 따른 즉각적인 반응을 관찰할 수 있습니다.")
