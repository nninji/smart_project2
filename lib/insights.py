"""
AI 스타일 자동 해석 (규칙 기반 — 신뢰 가능, 할루시네이션 없음)
"""


def interpret_capability(cap, norm, target=None):
    """공정능력 분석 결과를 자연어 인사이트로 변환"""
    if not cap or 'error' in cap:
        return []
    insights = []

    Cpk = cap['Cpk']
    Cp = cap['Cp']
    ppm = cap['ppmOverall']

    # 1. 종합 판정
    if Cpk >= 1.67:
        insights.append({
            'icon': '🌟', 'tone': 'success', 'title': '매우 우수한 공정',
            'body': f"Cpk = {Cpk:.3f}로 6시그마 수준에 근접합니다. 산포가 매우 작아 {ppm:,.0f} PPM 수준의 극저 불량률을 보입니다. 비용 절감과 관리 간소화를 검토할 수 있습니다."
        })
    elif Cpk >= 1.33:
        insights.append({
            'icon': '✅', 'tone': 'success', 'title': '충분한 공정능력',
            'body': f"Cpk = {Cpk:.3f}로 4σ 수준의 안정된 공정입니다. 예상 불량률 {ppm/1e4:.3f}%. 현재 상태를 유지하면 됩니다."
        })
    elif Cpk >= 1.0:
        insights.append({
            'icon': '⚠️', 'tone': 'warn', 'title': '경계선 수준',
            'body': f"Cpk = {Cpk:.3f}로 1.33 미만입니다. 예상 불량률은 {ppm/1e4:.2f}% 정도로, 공정 변동이 조금만 커져도 규격을 벗어날 위험이 있습니다."
        })
    else:
        insights.append({
            'icon': '🚨', 'tone': 'bad', 'title': '공정능력 부족',
            'body': f"Cpk = {Cpk:.3f}로 1.0 미만입니다. {ppm:,.0f} PPM 수준의 불량 발생이 예상됩니다. 산포 축소나 평균 중심 조정이 시급합니다."
        })

    # 2. 중심 이탈
    center_lsl = cap['xBar'] - cap['LSL']
    center_usl = cap['USL'] - cap['xBar']
    skew_ratio = abs(center_lsl - center_usl) / (cap['USL'] - cap['LSL'])
    if skew_ratio > 0.1:
        direction = '상한(USL)' if center_lsl > center_usl else '하한(LSL)'
        insights.append({
            'icon': '↔️', 'tone': 'warn', 'title': '평균이 한쪽으로 치우침',
            'body': f"공정 평균({cap['xBar']:.3f})이 {direction} 쪽으로 {skew_ratio*100:.1f}% 치우쳐 있습니다. Cp({Cp:.3f})와 Cpk({Cpk:.3f})의 차이는 {Cp-Cpk:.3f}로, 산포를 줄이기보다 평균을 중앙으로 이동시키는 것이 효과적입니다."
        })

    # 3. 군내 vs 군간 변동
    ratio = cap['sigmaOverall'] / cap['sigmaWithin']
    if ratio > 1.5:
        insights.append({
            'icon': '📊', 'tone': 'bad', 'title': '군간변동이 매우 큼',
            'body': f"σ_overall / σ_within = {ratio:.2f}. 부분군 간 평균이 크게 다릅니다. 외부 요인(원료 lot 차이, 작업 교대조, 환경 변화 등) 영향이 의심됩니다."
        })
    elif ratio > 1.2:
        insights.append({
            'icon': '📈', 'tone': 'warn', 'title': '군간변동 다소 큼',
            'body': f"σ_overall / σ_within = {ratio:.2f}. 군내변동에 비해 군간변동이 약간 큰 편으로, 장기 추세 모니터링이 필요합니다."
        })
    else:
        insights.append({
            'icon': '🎯', 'tone': 'success', 'title': '일관성 있는 공정',
            'body': f"σ_overall / σ_within = {ratio:.2f}. 군내·군간 변동의 차이가 작아 일관성 있게 운영되고 있습니다."
        })

    # 4. 정규성 검정
    if norm and not norm.get('passed', True):
        p_val = norm.get('pValue', 0)
        insights.append({
            'icon': '📐', 'tone': 'warn', 'title': '정규분포 가정 의심',
            'body': f"정규성 검정 3종 중 일부에서 정규분포 가정이 기각되었습니다 (최소 p={p_val:.3f}). 공정능력지수는 정규분포 가정 하에 계산되므로, Box-Cox 변환 후 다시 분석하는 것이 정확합니다."
        })

    # 5. 개선 방향
    if Cpk < 1.33:
        gap = 1.33 - Cpk
        reduction_pct = (1 - Cpk / 1.33) * 100
        insights.append({
            'icon': '💡', 'tone': 'info', 'title': '개선 방향 제안',
            'body': f"Cpk를 1.33까지 끌어올리려면 산포(σ_within)를 약 {reduction_pct:.0f}% 축소하거나, 평균을 {gap * 3 * cap['sigmaWithin']:.3f} 만큼 중심 쪽으로 이동시키면 됩니다."
        })

    # 6. 목표값과의 비교
    if target is not None:
        drift = cap['xBar'] - target
        if abs(drift) > cap['sigmaWithin']:
            insights.append({
                'icon': '🎚️', 'tone': 'warn', 'title': '목표값에서 평균 이탈',
                'body': f"공정 평균({cap['xBar']:.3f})이 목표값({target})에서 {'+' if drift >= 0 else ''}{drift:.3f} ({drift/cap['sigmaWithin']:.1f}σ) 떨어져 있습니다."
            })

    return insights


def interpret_spc(checked_df, chart_type):
    """SPC 관리도 결과를 자연어로 해석"""
    if checked_df is None or len(checked_df) == 0:
        return []

    insights = []
    violations_df = checked_df[checked_df['violations'].apply(lambda v: len(v) > 0)]

    if len(violations_df) == 0:
        insights.append({
            'icon': '✅', 'tone': 'success',
            'title': f'{chart_type} 관리도 정상',
            'body': f"검사된 {len(checked_df)}개 점 모두 Nelson Rules 8가지를 통과했습니다. 공정이 통계적 관리상태에 있습니다."
        })
        return insights

    # 규칙별 집계
    rule_count = {}
    for v_list in violations_df['violations']:
        for r in v_list:
            rule_count[r] = rule_count.get(r, 0) + 1

    rules_text = ', '.join(f"R{r}({rule_count[r]})" for r in sorted(rule_count.keys()))
    insights.append({
        'icon': '⚠️', 'tone': 'warn',
        'title': f'이상점 {len(violations_df)}개 검출',
        'body': f"{len(checked_df)}개 점 중 {len(violations_df)}개에서 Nelson Rule 위반이 발견되었습니다. 위반된 규칙: {rules_text}."
    })

    rule_desc = {
        1: '한계 이탈은 명백한 이상원인이 있음을 의미합니다. 해당 부분군 운영 조건(원료, 설비, 작업자)을 점검하세요.',
        2: '9점 연속 한쪽 편향은 평균 이동을 시사합니다. 설비 마모, 원료 변경, 작업 표준 변경을 검토하세요.',
        3: '6점 연속 증가/감소는 점진적 추세를 의미합니다. 공구 마모, 환경 변화(온도/습도)를 의심하세요.',
        4: '14점 교대는 시스템적 변동(2개 교대조, 2개 설비 등)을 시사합니다.',
        5: '2σ 외 2점은 평균 시프트 시작 신호일 수 있습니다.',
        6: '1σ 외 4점은 약한 평균 시프트를 의미합니다.',
        7: '1σ 내 15점 연속은 부적절한 데이터(층화) 또는 측정 오차를 시사합니다.',
        8: '1σ 밖 8점은 표본 추출 문제(혼합 모집단)를 시사합니다.',
    }
    for r in sorted(rule_count.keys()):
        if r in rule_desc:
            insights.append({
                'icon': '🔍', 'tone': 'info',
                'title': f'Rule {r}: {rule_count[r]}건 — 분석',
                'body': rule_desc[r],
            })

    insights.append({
        'icon': '🛠️', 'tone': 'info', 'title': '권장 조치',
        'body': '① 이상값을 분석하여 이상원인 제거 → ② 남은 데이터로 관리도 재작성 → ③ 모든 점이 관리상태가 될 때까지 반복 (Shewhart 표준 절차).'
    })

    return insights
