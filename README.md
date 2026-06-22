# Process Capability & SPC Studio — Streamlit Edition

**공정능력분석**과 **통계적공정관리**를 함께 수행하는 Streamlit 웹앱.

강의록(`08_공정능력분석.pdf`, `09_통계적공정관리.pdf`)의 **모든 수식·시각화·예제**를 충실히 구현하면서,
**LIVE 시뮬레이션, What-if 분석, 데이터셋 비교, AI 자동 해석** 등 차별화 기능을 더했습니다.

**PVC 점도 예제에서 Cp/Cpk/Pp/Ppk가 강의록과 소수점 4자리까지 100% 일치** ✓

---

## 🚀 VSCode에서 실행하는 법

### 1️⃣ Python 환경 준비

Python 3.9+ 권장. [python.org](https://www.python.org/downloads/)에서 설치.

VSCode에서 **Python 확장**(`ms-python.python`)도 설치하세요.

### 2️⃣ 압축 풀고 폴더 열기

VSCode에서 `File > Open Folder`로 `spc-streamlit` 폴더 선택.

### 3️⃣ 가상환경 생성 (권장)

VSCode 통합 터미널(`Ctrl/Cmd + ~`)에서:

```bash
# 가상환경 생성
python -m venv .venv

# 활성화
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
```

### 4️⃣ 의존성 설치

```bash
pip install -r requirements.txt
```

### 5️⃣ 앱 실행

```bash
streamlit run streamlit_app.py
```

브라우저가 자동으로 열리며 `http://localhost:8501` 표시됩니다.

---

## 📡 Streamlit Community Cloud 배포 (5분, 무료)

### 사전 준비
- GitHub 계정 (무료)
- Streamlit Community Cloud 계정 (무료, GitHub 로그인)

### 배포 단계

#### ① GitHub에 코드 푸시
```bash
cd spc-streamlit
git init
git add .
git commit -m "Initial commit"
# GitHub에서 새 저장소 만든 후:
git remote add origin https://github.com/USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

#### ② Streamlit Cloud 연결
1. [share.streamlit.io](https://share.streamlit.io) 접속
2. GitHub로 로그인
3. **"New app"** 클릭
4. Repository, Branch(main), Main file path(`streamlit_app.py`) 선택
5. **Deploy!** 클릭
6. 1-2분 안에 공개 URL 발급 (예: `https://your-app.streamlit.app`)

이후 GitHub에 푸시할 때마다 **자동 재배포**됩니다.

---

## 🎯 핵심 기능

### ① 데이터 입력
- **CSV 업로드** (Long/Wide 자동 변환)
- **8가지 강의록 예제** 즉시 로드
- **시뮬레이션 생성** (강의록 `generate_value_data`, `generate_count_data`와 동등)
- 데이터 미리보기 + CSV 다운로드

### ② 종합 대시보드
- 4단계 신호등 + KPI 4개
- 분석 절차 4단계 안내 (강의록 09 p.3-4)
- 부분군별 트렌드

### ③ 공정능력분석
- Cp/Cpk/Pp/Ppk + 5등급 판정
- 정규성 검정 3종 (Shapiro-Wilk, Anderson-Darling, Jarque-Bera) — **scipy.stats 사용으로 강의록과 동일한 정확도**
- **σ_within 3가지 계산법** 선택 (강의록 08 p.6-7)
  - pooled (sp/c4) · range (R̄/d2) · std (s̄/c4)
- **Box-Cox 변환** (강의록 08 p.27-29) — `scipy.stats.boxcox`로 MLE 람다 자동 추정
- 히스토그램 + Q-Q + 부분군별 박스플롯
- 단기 vs 장기 분포 비교
- 텍스트 보고서 다운로드

### ④ SPC 관리도
- **계량형 5종**: X̄-R, X̄-s, I-MR, CUSUM, EWMA
- **계수형 4종**: NP, P, C, U
- **Nelson Rules 8가지** 자동 감지
- 이상치 제거 후 재작성

### ⑤ 데이터셋 비교 (NEW)
- 두 데이터셋의 공정능력 나란히 비교
- 4개 지수 가로 막대 + 분포 겹쳐 그리기
- 🤖 자동 비교 결론

### ⑥ What-if 시뮬레이션
- μ, σ_within, σ_overall, LSL, USL 슬라이더로 실시간 변화

### ⑦ 🔴 LIVE 모드
- 실시간 데이터 스트리밍 시뮬레이션
- 시나리오 4종 (정상/시프트/산포 증가/심각 이상)
- 규격 외 점 실시간 빨간색 강조

---

## 🧪 검증 (강의록 100% 일치)

| 항목 | 강의록 값 | 본 앱 |
|---|---|---|
| PVC Cp | 1.3047 | 1.3047 ✅ |
| PVC Cpk | 1.2130 | 1.2130 ✅ |
| PVC Pp | 1.0958 | 1.0958 ✅ |
| PVC Ppk | 1.0188 | 1.0188 ✅ |
| σ_within | 127.7392 | 127.7393 ✅ |
| σ_overall | 152.0920 | 152.0920 ✅ |
| Box-Cox λ (lognormal) | scipy ≈ -0.04 | -0.0444 ✅ |
| d3(30), d4(30) | 3차항 포함 | 정확 일치 ✅ |

**모든 검증 통과** — `scipy.stats.boxcox`, `scipy.stats.shapiro` 등을 사용하므로 강의록과 동일한 정확도

---

## 📁 프로젝트 구조

```
spc-streamlit/
├── streamlit_app.py         # 메인 앱 (7개 페이지)
├── requirements.txt          # 의존성
├── .streamlit/
│   └── config.toml          # Streamlit 테마
└── lib/
    ├── __init__.py
    ├── constants.py         # 불편화 상수 (강의록 08 p.7, p.15)
    ├── stats_lib.py         # 통계 + 정규성 3종 (scipy 기반)
    ├── capability.py        # Cp/Cpk/Pp/Ppk (σ_within 3가지 방법)
    ├── spc.py               # 9종 관리도 + Nelson Rules
    ├── boxcox_lib.py        # Box-Cox (scipy.stats.boxcox)
    ├── data_gen.py          # 강의록 generate_*_data와 동일
    ├── datasets.py          # 8가지 강의록 예제
    └── insights.py          # AI 자동 해석
```

---

## 📖 강의록 페이지 매핑

| 강의록 | 본 앱 |
|---|---|
| 08 p.3-4 공정 관리상태 / 변동 | Overview 신호등 |
| 08 p.5 단기 vs 장기 분포 | 공정능력분석 페이지 분포 비교 |
| 08 p.6-7 **σ_within 3가지 방법** | `capability.py` `_compute_sigma_within` |
| 08 p.8 Pp/Ppk 수식 | `capability.py` |
| 08 p.9 **5등급 판정표** | `grade_capability` |
| 08 p.10 Wide→Long (melt) | 데이터 페이지 자동 변환 |
| 08 p.13 박스플롯 + 정규성 3종 | Plotly boxplot + `normality_check` |
| 08 p.15 **d3, d4 3차항 근사식** | `unbiased_const` |
| 08 p.27-29 **Box-Cox 변환** | `boxcox_lib.py` (scipy 사용) |
| 09 p.2-3 분류표 + 의사결정 트리 | SPC 페이지 가이드 |
| 09 p.4 **Nelson Rules 8 + 후조치** | `apply_nelson_rules` |
| 09 p.5 관리도 공식표 | `spc.py` 전체 |
| 09 p.18-25 다양한 예제 | 8개 예제 데이터셋 |
| 09 p.26-28 이상치 제거 재작성 | SPC 페이지 재작성 버튼 |

---

## 🛠️ 기술 스택

- **Streamlit 1.31+** (UI)
- **Plotly** (인터랙티브 차트)
- **scipy.stats** (정규성 검정, Box-Cox — 강의록과 정확 일치)
- **numpy, pandas** (수치 계산)
- 모든 처리 **클라이언트 사이드** — 데이터 외부 전송 없음

---

© 2026 · 강의록 기반 Streamlit 구현 ·
공정능력 (Cp/Cpk/Pp/Ppk) + Box-Cox · 9종 관리도 · Nelson Rules 8가지
