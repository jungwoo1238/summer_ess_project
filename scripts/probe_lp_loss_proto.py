"""solve_avg 목적함수에 Q의 손실저감 편익을 넣는 두 방식(PWL/QP)을 프로토타입으로 구현해
정확도·속도를 비교한다 (확인 전용, 본 파이프라인 미포함 - probe_q_selective.py의 직접 후속.
본 구현(lower_lp.py 수정) 전 방식 선택을 위한 실험).

## ★★ 1차 실험이 무효였던 이유 - 2차 개정의 동기
기준해 q_star의 Q는 압도적으로 PEAK_DAYS에 몰려 있다(1차 실행 결과 실측):
  P1: AVG 4시각/0.080 Mvar  vs  PEAK 40시각/1.750 Mvar (22배)
  P2: AVG 10시각/0.800      vs  PEAK 45시각/2.230
  P3: AVG 10시각/1.800      vs  PEAK 48시각/4.940
그런데 1차 버전은 solve_avg만 손실 인식형으로 고쳤다 - LP는 AVG_DAYS에서만 Q를 냈고
기준해는 PEAK_DAYS에서 Q를 냈다. 두 시나리오군이 겹치는 시각은 4~10개뿐이라 j_net 격차가
"근사 정확도"가 아니라 "서로 다른 전략을 비교한 결과"였다.

## ★★ 2차 개정도 무효였던 이유 - 3차 개정의 동기
2차는 solve_peak 변형에 C_CAP_PER_MW_YR로 원화환산한 피크저감 항을 넣어 "AVG/PEAK 양쪽
Q를 자유롭게 LP가 결정"하도록 확장했다. 그런데 편익 함수(benefits.b_loss/b_energy)는
AVG_DAYS만 집계한다 - PEAK_DAYS의 손실저감은 장부(j_net)에 거의 반영되지 않는데도,
프로토타입 LP는 그 시각의 Q를 "장부에 없는" 손실편익만 보고 정격까지 밀어붙였다
(2차 실행 실측: P1·P2에서 48시각 중 40시각이 q=S). 그 결과 PCS 다각형 제약이 P를
정격의 26.8%로까지 묶어버렸다 - 측정된 j_net 격차는 방식(PWL vs QP)의 근사오차가 아니라
**정식화 자체의 오류**(장부에 없는 편익을 좇은 부작용)였다.

두 방식 각각의 구현도 그동안 제대로 된 적이 없었다:
- PWL: 구간 경계 {0, 0.05, 0.15, 0.5}인데 실제 최적 Q(AVG_DAYS 기준)는 0.02~0.05 수준이라
  모든 실측값이 첫 구간(0~0.05) 안에 통째로 들어갔다 - M=1/2/4의 나머지 분할점(0.15, 0.3)이
  전부 실측 범위 **위**에 있어 M을 늘려도 실제 데이터를 가로지르는 분할이 하나도 생기지
  않았다. M=1/2/4가 사실상 동일한 답을 낸 근본 원인이 이것이다(경계 위치가 잘못됐을 뿐,
  PWL 방식 자체의 결함이 아니었다).
- QP: 제곱을 전개하지 않아 DPP가 깨졌다(직접 cp.square(P_e)를 시도 -> is_dcp(dpp=True)=False
  실측). 전개 형태는 이번이 첫 시도다.

## ★★★ 3차 개정 (현재) - 통제 설계·경계 재설정·QP 전개·솔버 정확도
### 통제 설계: solve_peak을 양쪽 방식·기준해 모두 force_q_zero=True로 고정
2차의 실패 원인(장부에 없는 편익을 LP가 좇음)은 "회계 규칙(b_loss/b_energy가 AVG_DAYS만
집계)"을 이 실험에서 바꿀 수 없다는 데서 나온다(그 회계 자체를 고치는 것은 편익함수
재구조화이고 별도 안건 - 이 실험의 범위 밖). 따라서 3차는 **PEAK_DAYS의 Q를 프로토타입
LP·기준해(q_star) 양쪽에서 전부 0으로 고정**해 애초에 "장부에 없는 편익을 좇을 여지"를
구조적으로 없앤다. 대신 solve_peak 자체의 프로토타입(C_CAP 원화환산 등, 2차가 만든 것)은
전부 제거했다 - **더 이상 필요하지 않다**(force_q_zero=True는 lower_lp.py 원본에 이미
있는 경로이고, evaluate.py의 force_q_zero 우회(probe_q_value._evaluate_with_force_q)가
그대로 이 값을 준다). 비교 범위는 AVG_DAYS로만 좁아지지만, AVG_DAYS는 손실이 b_loss/
b_energy에 그대로 집계되고 solve_avg의 결정이 b_defer에 영향을 주지 않아 회계가
정합하다 - 이 실험이 "근사 정확도"만을 순수하게 재도록 만드는 유일한 방법이다.
결과적으로 기준해도 재구성한다: `_build_qstar_unit_q`는 PEAK_DAYS를 CSV 값과 무관하게
항상 0으로 채운다(원본 CSV의 PEAK q_star는 컨텍스트 확인용으로만 stdout에 참고 출력한다).

### PWL 경계 재설정: 실제 분포 안에 분할점을 둔다
2차 실행 결과(위 "1차 실험" 절 데이터, AVG_DAYS q_star)가 이미 알려준 범위 - 시각당
비영값 0.02~0.05 Mvar - 안에 분할점을 둬야 M을 늘리는 것이 실제로 근사오차를 줄이는지
검증할 수 있다. 이번 실행 시작 시 `_print_qstar_avg_distribution`이 로드된
probe_q_selective CSV의 실제 AVG_DAYS 비영 분포(백분위수)를 먼저 출력한다 - 아래
SEGMENT_BOUNDARIES가 그 분포와 여전히 맞는지 실행마다 눈으로 확인할 수 있게 하기 위함이다
(부합하지 않으면 이 상수부터 다시 볼 것 - 코드 상수는 요청 시점에 알려진 값을 반영한
것이지 매 실행 데이터에 자동으로 맞춰지지 않는다).

### QP 전개: DPP를 지키려면 bus_onehot을 포인트별 build-time 상수로 구워야 한다
지시서의 전개식(Q_e = Q_e_base - dQ_e, Q_e^2 = 상수 - 2*Q_e_base*dQ_e + dQ_e^2)은
"dQ_e가 순수 변수"라는 전제 위에서만 뒤 두 항이 DPP를 만족한다. 그런데 원래 구조에서
dQ_e = D@(bus_onehot.T@Q)/S_BASE_MVA는 bus_onehot이 **Parameter**라 계수 자체가
파라미터다 - 이걸 제곱하면(dQ_e^2) 파라미터가 제곱으로 등장해(bus_onehot 이 두 번
곱해지는 형태) DPP가 요구하는 "임의 비선형 항의 계수는 파라미터에 대해 최대 1차"를
위반한다(smp를 곱하기 전에 이미 깨진다 - PWL 섹션의 "두 파라미터를 곱하면 깨진다"는
교훈과 같은 계보의 문제이지만 원인은 다르다: 여기는 파라미터 하나가 제곱으로 등장하는
경우다). **해결: bus_onehot을 QP 문제에서만 build-time 상수(그 포인트의 실제 버스로
구운 one-hot 배열)로 바꾼다.** 그러면 dQ_e/dP_e는 계수가 전부 상수인 순수 변수 표현식이
되어(파라미터 0개) 제곱해도 DPP가 깨지지 않는다. 대가는 **QP 문제를 포인트마다(버스마다)
새로 지어야 한다**는 것이다(PWL/avg-baseline은 bus_onehot이 여전히 Parameter라 M별로
한 번만 짓고 3개 포인트에 재사용 - 기존 구조 그대로). 3개 포인트뿐이라 이 재빌드 비용은
무시할 만하다(지시서 "구현부담 반나절" 수준).
그 다음 문제: smp(Parameter, 시나리오마다 값이 바뀜)를 이 손실항에 곱해야 하는데, 손실항
자체가 이미 Q_e_base(또 다른 Parameter, 시나리오의 무효부하에서 나옴)를 포함하는 교차항을
가지므로 "smp x Q_e_base"라는 **서로 다른 두 Parameter의 곱**이 다시 등장한다(PWL 섹션이
이미 겪은 바로 그 함정). **해결: PWL과 같은 트릭 - smp*Q_e_base(그리고 smp*P_e_base)를
cvxpy 밖에서 numpy로 미리 곱해 하나의 합성 Parameter 값으로 채운다**(_set_params의
cross_p/cross_q 참조). 제곱 항(dQ_e^2/dP_e^2)의 계수는 smp*r_pu 하나뿐이라(둘 다 애초에
Parameter/상수 하나씩이라 곱이 파라미터 1개 등급을 넘지 않음) 그대로 cvxpy 안에서
곱해도 안전하다.
**구현 후 반드시 problem.is_dcp(dpp=True)를 확인한다** - `dpp_preserved` 필드로 보고되고,
False면 `_diagnose_dpp_terms`가 하위 표현식(dP_e/dQ_e/제곱항/교차항/전체 loss_term)을
하나씩 `.is_dpp()`로 짚어 어느 항이 원인인지 stdout에 낸다(지시서 요구사항).

### 손실 테이블 실측 범위 축소 (ALL_DAYS -> AVG_DAYS, 2차의 확장을 되돌림)
PEAK_DAYS 프로토타입이 통째로 사라졌으므로 PEAK_DAYS의 PWL 계수 실측도 더 이상 필요 없다
(2차가 넓혔던 것을 3차가 되돌린다). 32버스 x AVG_DAYS 3개 x 24시간 x 경계점 7개 =
16,128회 조류계산(2차의 19,200회보다 적다 - 시나리오는 줄고 경계점은 늘었다).

### 솔버 정확도 대응
2차 실행에서 avg+PWL 조합에서만 OPTIMAL_INACCURATE 6/60(10%)이 났다(CLARABEL이 59/60
선택). PWL 세그먼트 폭이 좁을수록(이번 개정으로 0.01처럼 더 좁아진 구간이 생겼다) 계수
스케일 차가 커져 내점법 수렴이 아슬아슬해지는 것으로 추정한다. `_solve_timed`의 CLARABEL
1차 시도에 `max_iter`를 기본값(200)보다 크게 주어 "느리지만 확실히 수렴"을 우선하고,
그래도 안 되면 완화된 허용오차로 2차 CLARABEL 시도, 그다음 OSQP, 마지막으로 cvxpy 기본
자동선택 순으로 재시도한다(값을 버리지 않고 OPTIMAL_INACCURATE로 표시만 함 - 지시서
"해결되지 않으면 명확히 표시"). ts_rows CSV에 `inaccurate` 열을 추가해 해당 (method,M,
scenario)에서 나온 시각별 오차행을 사후에 걸러낼 수 있게 했다 - 이 세션은 실행 없이
작성만 하므로 이 조정이 실제로 10%를 없앴는지는 **실행 후 `_print_solver_diagnostics`
출력으로 확인할 것**(안 없어지면 그 조합을 표에서 제외하고 판단할 것 - 지시서 대안).

## ★★★★ 4차 개정 - 다각형 버그 수정·타이밍 재설계·QP vs PWL 손실추정 대조
### 버그: 다각형이 원을 바깥에서 감싸고 있었다(circumscribed, 정격 초과 허용)
3차의 `s_app <= S_col`은 다각형의 지지초평면 우변을 S까지 허용했는데, 이는 "꼭짓점이
반지름 S 원 위에 있는" 내접(inscribed) 다각형이 아니라 **아포뎀(apothem)이 S인**
다각형을 만든 것과 같다 - 이 경우 꼭짓점(원과의 접점 방향 사이 중점)은 반지름
S/cos(pi/N)에 있어 정격을 1/cos(pi/12)-1=3.53%(N=12) 초과한다. 실측으로도 확인됐다
(P1에서 비영 q_lp 30시각 중 14시각이 (P,Q)=(0.176,0.0472)에 붙었고 피상전력이
0.1822=1.0353*S). **수정: `s_app <= S_col*cos(pi/POLY_N)`**(원본 lower_lp.py가
`s_cap = S*cos(pi/12)`로 고정 상수를 거는 것과 동일한 관례 - 이제 다각형이 원에
내접해 정격을 절대 넘지 않는다). 수정 직후 `_assert_pcs_circle`이 모든 (unit,scenario,
t)에서 sqrt(P_net^2+Q^2)<=S*(1+1e-9)를 확인하고, 위반이 하나라도 있으면 즉시
AssertionError로 중단·보고한다(지시서 요구 - 워밍업 solve 직후, 타이밍 반복 전에 1회
확인하면 충분하다 - Parameter 값이 같으면 재-solve 결과도 물리적으로 동일해야 하므로).

### 타이밍 측정 재설계 - 컴파일 시간과 순수 solve 시간을 분리
3차는 각 시나리오를 1회만 풀어 그 시간을 그대로 보고했는데, 그 1회에 DPP 컴파일(최초
호출) 시간이 우연히 섞여 세그먼트 수와 무관한 비단조 결과가 났다(실측: PWL M=1
P1=0.70초 > M=2 P1=0.49초 - M을 늘렸는데 오히려 빨라 보이는 건 방식 성능이 아니라
"어느 호출이 컴파일을 떠안았는지"의 문제였다). `_compute_schedule`을 워밍업 1회
(컴파일 포함, 결과값 확정) + `N_TIMING_REPS`(5)회 반복 측정(컴파일 제외, 결과값은
버림) 구조로 바꾸고 **중앙값**을 solve_time으로 보고한다. 워밍업 총합도 별도
반환해(`warmup_time_avg`) "컴파일 비용이 실제로 얼마였는지"를 median과의 배율로
볼 수 있게 했다.

### QP의 Problem 재빌드 비용 - 실배포 함의 명시
QP는 3차 개정에서 이미 `bus_onehot`을 build-time 상수로 굽어 포인트마다 Problem을
새로 짓도록 바꿨다(DPP 유지를 위한 필연적 설계 - 3차 개정 절 참조). 이번에 그 **빌드
자체의 시간**을 별도로 측정해(`qp_build_times`, `_print_method_summary`가 출력) "실제
배포 시에도 그런지"를 지시서 요구대로 명시한다: **그렇다.** lower_lp.py의 현재 캐싱
철학(build once per (kind,n,force_q_zero,...), 이후 Parameter만 갱신)은 bus가
Parameter라는 전제 위에 있는데, QP를 그대로 채택하면 이 전제가 깨진다 - PSO가 매
입자마다 던지는 서로 다른 버스마다 Problem을 새로 지어야 하므로, 본실험 규모(입자32 x
세대100 x run30)에서 이 빌드비용이 반복 지불되며 지배적 병목이 될 수 있다. PWL은
bus_onehot이 여전히 Parameter라 이 비용이 없다 - **QP 채택 여부는 정확도뿐 아니라
이 구조적 비용차이로도 판단해야 한다**(이 세션은 실행하지 않았으므로 실제 빌드시간
수치는 실행 후 확인할 것).

### QP vs PWL 손실추정 자체의 대조 (P3에서 2.5배 갈린 원인 진단)
P3에서 QP(0.685 Mvar)와 PWL(1.711 Mvar)의 AVG_DAYS Q 사용량이 2.5배 갈렸다(3차 실행
실측). `_diagnose_q_prediction_gap`이 고정 Q=0.05(이후 QP 단위 수정 라운드에서
`Q_DIAG_LEVELS` 리스트로 확장됨 - 아래 참조)를 실제로 주입했을 때
두 방식이 예측하는 손실 저감량과 실측(loss_table 직접 조회)을 대조한다. 0.05는
SEGMENT_BOUNDARIES[4]의 세 번째 경계와 정확히 같으므로 PWL의 이 지점 예측은 시컨트들의
텔레스코핑 합이라 실측과 **대수적으로 정확히 같다**(근사가 아니라 항등식 - 0이 아닌
차이가 나오면 구현 버그를 의심할 것). 따라서 이 비교에서 'QP vs 실측' 차이가 QP
자신의 근사오차(V^2~=1 근사)만을 순수하게 분리해서 보여준다. 자동판정은 하지 않는다
(지시서 - 수치만 stdout에 낸다, 사람이 P3의 격차가 이 손실추정 차이로 설명되는지
직접 판단할 것).

### QP 단위 인수 수정 (S_BASE_MVA 누락) + 정식화 안전장치 assert 5종
3차 실행 결과 Q=0.05 손실저감 예측이 실측 대비 **QP만** 약 10.8배 작게 나왔다(P1 실측
0.000938 MW vs QP 0.000087 MW). 근사오차가 아니라 단위 누락이었다 - dP_e/dQ_e는
S_BASE_MVA로 나눈 pu이므로 r_pu*(dP_e^2+dQ_e^2)는 "pu 손실"인데 SMP(원/MWh)와 곱해
원화로 만들려면 먼저 MW로 환산해야 했다(그 인수가 `_set_params`의 `rsmp`에서 빠져
있었다). 실측 비율 10.70~10.78을 10(이 단위 인수) x 1/V^2(LinDistFlow V=1 근사 효과)
로 분해하면 함의 전압이 0.9631~0.9666으로 나오는데, 이는 해당 버스(15/17/31)의
실제 계통전압(1절: 슬랙 1.02에서 Vmin 0.9620 @ bus 17)과 정합한다 - 즉 단위 인수를
고치면 QP의 잔여 오차(~7~8%)는 전부 V^2 근사 효과로 설명된다. `_set_params`의
`rsmp`(및 이로부터 파생되는 `cross_p`/`cross_q`)와 `_diagnose_q_prediction_gap`의
`qp_reduction_t`에 `PM.S_BASE_MVA`를 곱해 수정했다.

진단도 강화했다: `Q_DIAG_LEVELS=[0.05, 0.0375]` - 0.05(SEGMENT_BOUNDARIES[4] 경계,
PWL이 텔레스코핑 항등식으로 정확) 하나만 재면 비교가 PWL에 유리하게 기울므로, 경계
"사이"인 0.0375(`_pwl_predicted_reduction`가 세그먼트 내 선형보간으로 계산 - 이 점에서는
PWL도 실측과 다를 수 있음)를 함께 본다. 0.0375는 실측(근사 아님)이 되도록
`Q_BOUNDARY_POINTS`에 추가했다.

**정식화 안전장치 5종(전부 "물리/정식화 계약 위반 검출" - 판정이 아니다):**
1. **POLY_N 짝수 검사**(모듈 로드 시 1회) - 홀수면 충전(P_net<0) 시각의 Q=0 회귀
   앵커가 구조적으로 깨진다(theta=pi가 다각형 격자에 없으므로).
2. **상보성 검사** `_assert_no_padding_exploit`(`_compute_schedule` 워밍업 직후,
   `_assert_pcs_circle`과 같은 자리) - P_ch*P_dis<=tol. q_penalty 항은 "같은 시각에
   P_ch/P_dis를 함께 늘려 P_net은 그대로 두고 q_penalty만 줄이는" 패딩 exploit에
   취약한데, `_assert_pcs_circle`(P_net 기준)은 이를 못 잡는다 - 별도 검사가 필요한
   이유다. 손익분기(지시서 유도) 상 현재 ETA_PCS=0.97(C_PCS=0.030)은 임계(~0.960/
   0.0399) 아래지만 여유가 1.33배뿐이다.
3. **Q=0 회귀 앵커 검사** `_verify_q_zero_anchor`(통제점당 1회, 메인 앞부분) - Q==0
   제약을 추가한 별도 Problem으로 q_penalty가 정확히 0인지 확인한다.
4. **PWL 세그먼트 기울기 단조성 검사** `_check_pwl_slope_monotonicity`/
   `_print_pwl_monotonicity_check`(경고만, 중단하지 않음) - 비단조면 LP가 세그먼트를
   물리적 순서 무시하고 이득 큰 구간부터 채워 손실편익을 과대평가할 수 있다.
5. **무료 Q 구간 계측** `_compute_free_zone_stats`(판정 없이 수치만, ts_rows CSV에
   반복 기입) - phi<=pi/POLY_N인 부채꼴에서는 q_penalty가 항상 0이라 LP가 그 구간
   끝(free_zone_width=S*sin(pi/POLY_N))까지 손실계수와 무관하게 Q를 밀어붙인다. 4차
   실행에서 P1은 PWL M=1과 QP가 14시각을 정확히 0.045552(=0.176*sin(pi/12))에
   고정해 두 방식을 구분할 수 없었다 - 이 계측은 각 통제점의 판별력을 보기 위한 것이다.

### ★★★★★ 확장·수정 라운드 - 검산 2-2 위양성 수정 + QP V^2 보정 (4차 개정 계속)
직전 실행에서 두 가지가 확인됐다: (1) QP 단위 수정은 성공했고 잔차(~7~8%)의 정체가
V=1 근사임이 규명됐다(작업 B로 보정), (2) 위 "검산 2-2"(상보성, `P_ch*P_dis<=tol`)가
위양성으로 실행을 3회 연속 중단시켰다 - 곱의 단위(MW^2)가 P_dis 스케일에 비례해
P_dis=0.173이면 P_ch가 9.32e-9(CLAUDE.md 7절 실측 수치잡음 1e-8~1e-9 MW와 같은
자릿수)만 되어도 tol(1e-9)을 넘었다. 위 docstring의 "2. 상보성 검사
`_assert_no_padding_exploit`"(P_ch*P_dis 기준)와 `PCS_COMPLEMENTARITY_TOL` 상수는
**이 라운드에서 폐기**됐다 - 검사량을 곱이 아니라 패딩 크기 자체
`pad=min(P_ch,P_dis)`[MW]로 바꾼 `_compute_padding_stats`로 대체했다(항상 계측,
`PAD_WARN_MW`=1e-6(~20.8원/년) 초과 시 경고 수집, `PAD_ABORT_MW`=1e-4(~2,080원/년)
초과 시에만 중단 - 유도는 상수 정의부 주석 참조).

**작업 B(QP V^2 보정):** `_measure_loss_table`이 손실 테이블 실측 루프의 q=0 패스
(그 자체가 이미 완전한 no-op 기저 조류계산 - bus 배정과 무관)에서 추가 조류계산 없이
from-bus 전압 제곱(`v_sq_line_table[scenario]`, shape (n_branch,T))을 캡처한다.
`_set_params`가 `QP_V2_CORRECTION`(기본 True) 플래그를 따라 `rsmp`를 이 값으로 나눠
V=1 근사를 제거한다. `_diagnose_q_prediction_gap`은 이 플래그와 무관하게 항상
보정 전/후를 나란히 계산·출력하고(지시서 B-4), (예측/실측) 비의 분포(min/p25/median/
p75/max)도 낸다(B-5 - 중앙값만으로는 편향이 일정한 비율인지 알 수 없으므로).

**작업 C(무료구간 필터 수정):** `_compute_free_zone_stats`의 기존 조건(`arr>0.0`)은
부동소수점 잡음(수치적으로 0인데 엄밀히는 양수)까지 세어 P1에서 71/72(사실상 전체)를
무료구간으로 오판했다. `_group_stats`와 같은 계보의 "비영" 기준(`FREE_ZONE_NONZERO_TOL`
=1e-6)으로 하한을 엄격히 하고, 분모(비영 시각 수)와 그 비(`frac_in_zone`)를 함께 낸다.

**작업 D(실행 메타):** 실행 시작 시각·호스트명·`QP_V2_CORRECTION`·`PAD_WARN_MW`/
`PAD_ABORT_MW`를 정식화 상수와 함께 stdout에 남긴다 - 같은 조합의 solve_time이
실행마다 크게 달랐던 현상(머신 부하 변동으로 추정)을 판단할 근거 자료다.

**작업 A-5(부분 결과 보존):** ts_rows CSV를 실행 끝에 한 번에 쓰지 않고
`_open_csv_writer`로 파일을 실행 내내 열어 둔 채 `_process` 조합이 끝날 때마다
`_append_rows`로 즉시 append+flush한다 - 어느 조합에서 `PAD_ABORT_MW` 중단이 나도
그 이전까지 완료된 조합의 결과는 CSV에 남는다.

## ★★★★★★ 계측 추가 라운드 - 세그먼트 기울기 + 기저 조류 대조 (정식화 불변)
직전 실행은 네 방식(PWL M=1/2/4, QP)이 모두 완주했고 결과 자체는 유효하다 - 이번 라운드는
**정식화(SEGMENT_BOUNDARIES, POLY_N 등)를 전혀 바꾸지 않고 계측만 추가**해 두 가지 미해명을
좁힌다: (1) PWL이 특정 Q에서 정확히 멈추는 현상, (2) QP가 실측 대비 7~9% 과소예측하는
잔여 오차의 원인. 이 절의 작업들은 **판정하지 않는다** - 원자료를 stdout/CSV로 남기고,
해석이 필요한 부분(작업 D)은 코드가 아니라 별도 보고 문서로 전달한다(지시 - "코드를 고치지
마라, 보고만 하라").

### 작업 A: PWL 세그먼트 기울기 실측 노출
`_lhs_rows_for`가 이미 계산해 LP Parameter로 흘려보내는 세그먼트 시컨트 기울기
(`_segment_slopes`)를 그 자체로 stdout/CSV에 노출한다 - 지금까지는 LP 안으로 들어가기만
하고 사람 눈에 보인 적이 없었다. `_pwl_segment_slope_report`가 통제점 x M(1,2,4) x
세그먼트별로 AVG_DAYS 72개 시각의 기울기 분포(min/p25/median/p75/max)와 기울기<=0인
(scenario,t) 개수를 낸다. 같은 표에 `_local_slope_near`가 계산하는 "그 구간 중앙값 Q
근방의 국소 기울기"(Q_BOUNDARY_POINTS 중 그 중앙값을 포함하는 가장 좁은 인접 구간의
시컨트 - 세그먼트 자신의 시컨트보다 촘촘한 실측값)를 나란히 찍어, 넓은 세그먼트가 실제로
얼마나 "평평해졌는지"(시컨트가 국소기울기보다 훨씬 작은지)를 사람이 직접 비교할 수 있게
한다. 시각별 원자료는 별도 CSV(`_lhs` 접미사, `_write_lhs_csv`)로 남긴다 - ts_rows에
합치면 (point x M x scenario x t x m) 조합으로 행이 폭증하기 때문이다.

### 작업 B: LinDistFlow Qe_base vs AC 실측 조류 대조
QP의 손실 예측은 `Qe_base = D@(load_q/S_BASE_MVA)`("하류 부하의 합", `_set_params`/
`_diagnose_q_prediction_gap`이 이미 이 값을 쓴다)를 쓰는데, 실제 선로 조류는 "하류 부하 +
하류 손실"이다(LinDistFlow가 조류식에서 손실을 무시하므로). `_measure_loss_table`의 기존
q=0 패스(작업B가 이미 이 지점에서 `v_sq_line_table`을 캡처하고 있었다 - 그 관례를 그대로
잇는다)에서 **추가 조류계산 없이** `net.res_line`의 `q_from_mvar`/`q_to_mvar`/`p_from_mw`/
`p_to_mw`/`pl_mw`/`ql_mvar`와 `net.res_bus.vm_pu`(전 버스, 작업C가 함께 씀)를 더 캡처한다
(`ac_flow_table`, `v_bus_table` - `_measure_loss_table`의 반환값에 추가됨, 기존
`loss_table`/`v_sq_line_table`의 의미·형태는 손대지 않는다). branch 순서는 기존
`v_sq_line_table`과 동일하게 `_branch_line_idxs`(정렬 규칙: `sorted(lines.index)`, 기존
`_branch_from_bus_array`가 내부에서 쓰던 것을 별도 함수로 뽑아 `_branch_to_bus_array`와
공유한다)를 쓴다. `_print_qe_base_ac_comparison`이 전 (branch,t)와, 통제점별 경로
(D 행렬로 걸러낸 슬랙->해당 버스 선로만)의 `|q_from|/Qe_base`·`|q_to|/Qe_base` 분포를
낸다(분모<0.001 Mvar인 선로는 제외하고 제외 건수를 함께 보고). 비를 내기 전에
`_print_sign_convention_check`가 pandapower `q_from_mvar` 부호규약과 LinDistFlow
Baran-Wu 부하양수 규약의 대응을 코드 위치를 인용해 확인하고, 실측 부호 불일치 건수를
직접 세어 보고한다(규약이 맞았다고 주장만 하지 않고 데이터로 확인).

### 작업 C: 손실 공식·전압 규약의 항등성 검증
직렬 임피던스 가지에서 `loss=r*(P_from^2+Q_from^2)/V_from^2`은 근사가 아니라 항등식이다
(to단도 마찬가지, 무손실 병렬소자 전제 - 3절 슬랙 수지 검증에서 이미 `net.shunt` 빈
테이블·`net.line.c_nf_per_km` 전 선로 0 확인됨, build_net.py에도 `net.trafo` 관련 코드가
없다). `_print_loss_formula_identity_check`가 작업B에서 캡처한 AC 실측값으로 이 항등식이
재현되는지 확인해, 남은 QP 오차가 "손실공식/전압규약" 탓인지 "Qe_base 자체" 탓인지
분리한다. 단위계는 물리단위(Ohm·MW·MVAr·kV)를 쓴다 - `r_ohm = r_pu*Z_BASE_OHM`(lower_lp.py
가 `r_pu`를 만들 때 쓴 나눗셈의 역연산)로 되돌리고, `V=vm_pu*VN_KV`[kV]로 실제 선간전압을
쓴다. 이 조합(MW/MVAr/kV/Ohm)에서 `Ploss=R*(P^2+Q^2)/V^2`는 3상 등가회로의 표준 관계식이며
별도 배수가 필요 없다(pandapower 자체가 `res_line`을 이 관례로 계산한다) - pu로도 검산할
수 있었으나 이미 확립된 물리단위 관례를 그대로 쓰는 쪽이 lower_lp.py의 pu 변환 자체가
맞는지까지 함께 확인하는 효과가 있어 이쪽을 택했다.

### 작업 D: QP 7~9% 과소예측 원인 조사
코드를 고치지 않고 가설과 근거만 보고한다(지시 사항) - 결과는 이 파일이 아니라 별도
보고로 전달된다. 작업 B/C가 그 보고를 뒷받침하는 원자료를 만든다.

### 작업 E: stale 판정 문구 수정 - 코드에 박아 둔 결론은 근거가 바뀌어도 저절로 안 바뀐다
QP 방식 요약의 "실배포 함의" 문구가 이전 버전의 빌드시간(0.022~0.030초) 전제로 "지배적
병목이 될 수 있다"는 판정성 결론을 담고 있었는데, 직전 실행 실측(P2 0.0093초/P3 0.0094초,
P1 0.0646초는 콜드스타트 포함)은 그보다 훨씬 작았고, 버스 정의역이 1~32 32개뿐이라는
사실(워커당 사전 빌드 캐시 가능)도 그 판정에 반영돼 있지 않았다. `_print_method_summary`의
QP 분기에서 판정 문장을 지우고 사실(측정된 시간)과 "사용자가 판단하라"는 문구로
바꿨다 - 이 스크립트 전체가 지키는 "stdout에 판정성 문장을 넣지 않는다" 원칙이 코드
자체에서 깨졌던 실례이자, 그 원칙이 왜 필요한지를 보여주는 사례라 여기 남겨 둔다.

## ★★★★★★★ 계측 추가 라운드(2차분) - AC 되먹임 규명 + 세그먼트 경계 세분화 + 기준시간 측정
직전 실행(1차분 계측)에서 확정된 것: (1) 손실공식 항등식이 AC 실측 대비 상대오차
1e-16~7.5e-12로 사실상 정확하다 - 공식·규약·pu변환은 배제. (2) "Qe_base가 손실을
제외해 작다"는 가설은 경로 기준 median 1.4~2.2%로 대체로 기각. (3) PWL이 Q=0.05에서
멈추는 이유가 규명됐다 - 다각형 facet의 한계 PCS비용이 계단함수 C_PCS*sin(2*pi*k/N)이고
P1 M=4의 [0.05,0.5] 세그먼트 시컨트(0.009559)가 그 첫 계단(0.015, PCS_FACET_THRESHOLD)을
못 넘어 LP가 채우지 않는다 - PWL의 한계가 아니라 그 구간이 너무 넓다는 **경계 설계**의
결과다(국소기울기 선형근사로 역산한 진짜 최적점 ~0.125가 QP 실측 최댓값 0.1245와 일치).
남은 것: QP의 7.8% 과소예측 중 단위수정(+1.8%)·V^2보정(-1.3%)으로 설명되는 부분을 뺀
약 7%가 미규명 - 유력 후보는 AC 2차 되먹임(Q주입->손실감소->슬랙유입감소->손실이
loss~(P^2+Q^2)이므로 추가감소, 실측 loss_table은 이를 포함하지만 QP의 단발 선형화는
못 담는다)이나 확정 자료가 없었다(직전 계측이 전부 Q=0 기저 상태만 잡았기 때문).

### 작업 A(2차분): AC 되먹임 성분 분해
`_measure_loss_table`의 기존 (bus==ALL_BUSES[0], q==0.0) 전용 캡처와 **별개로**,
CONTROL_POINT_BUSES(3개, POINTS에서 유도)에 한해 **모든 q 지점**에서 res_line/res_bus를
읽어 `ac_full_table[bus][scenario]`(shape (T,n_q,n_branch 또는 n_bus))에 담는다(새
pp.runpp 없음 - 이미 도는 (bus,s,t,qi) 루프의 조건만 넓힌 것). `_decompose_ac_feedback`이
작업C에서 이미 확인한 loss=r*(P^2+Q^2)/V^2 항등식으로 q=0->q_level 손실 변화를
A(Q항)+B(P항)+C(V항)으로 정확히 쪼갠다 - 세 항의 합이 실측 총손실차와 일치하는지(항등성,
구현 검증이지 판정이 아니다)를 먼저 확인하고, 그 다음 A/B/C 각각이 총손실차에서 차지하는
비율분포(min/p25/median/p75/max)를 낸다. `_qp_predicted_A`가 QP(보정후)의 예측을
**독립적으로 재구현**해(공유 함수로 리팩터링하지 않음 - A'/A, A'/dLoss_total 비율이
"두 독립 구현의 일치 여부"를 보는 교차검증이라 하나로 합치면 동어반복이 된다) 기존
`_diagnose_q_prediction_gap`이 보고하던 비율(0.9276 등)과 이 라운드의 A'/dLoss_total이
일치하는지도 확인한다. `_print_dqe_ratio`가 (Q_e0-Q_eq)/q(경로 선로만)의 분포를 추가로
낸다 - Q 주입이 하류 무효손실도 줄이므로 1보다 클 것으로 예상되나 확정은 수치로 한다.

### 작업 B(2차분): 세그먼트 경계 세분화 (기존 M=1/2/4 보존, M=9 추가)
Q_BOUNDARY_POINTS에 0.075/0.15/0.2를 추가해 8점->11점으로 넓혔다(손실 테이블 실측
비용 +37%, 32버스x3시나리오x24hx11점=25,344회 조류계산). SEGMENT_BOUNDARIES[9]=
[0,0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.5](경계 10개=세그먼트 9개, 전부
Q_BOUNDARY_POINTS의 부분집합 - 모듈 로드 시 assert로 전 M에 대해 확인)가 [0.05,0.5]
구간(직전 라운드에서 PWL이 못 채우던 바로 그 구간)을 실측 최적점(~0.125) 근방에서
세분한다 - "PWL의 한계가 아니라 경계 설계 문제"라는 위 가설을 M=9로 검증한다. 기존
M=1/2/4·POLY_N=12는 전혀 건드리지 않았다(통제 유지 - 지시서 요구). 실행 순서는
PWL(M=1->2->4->9)->QP로, `_print_pwl_segment_slopes`가 M=9의 시컨트 기울기 표에
PCS_FACET_THRESHOLD 값을 나란히 낸다(판정은 하지 않는다 - "넘는다/못 넘는다"는 사람이
본다). QP는 대조군이므로 이 라운드에서 정식화·순서 어느 쪽도 건드리지 않았다.

### 작업 C(2차분): 기준시간 측정 (같은 하네스로 apples-to-apples)
기존 solve_time 배율의 기준(REFERENCE_SOLVE_TIME_SEC=0.549초)은 다른 머신·다른 범위
(evaluate.py 전체: LP 5시나리오+사후 조류계산+편익집계)에서 측정된 값이라 "손실 항 추가로
얼마나 느려졌는가"의 기준이 될 수 없었다(실제로 "손실 항을 넣었더니 8배 빨라졌다"는
있을 수 없는 결론이 나왔다). `_build_problem_proto`에 새 method 2개를 추가해 **완전히
동일한 타이밍 하네스**(`_compute_schedule` - 워밍업 1회+5회 반복, 컴파일 제외, 중앙값)로
측정한다: `'none'`(기준1, force_q_zero=True 등가 - Q 자체가 cp.Constant(0)이고 다각형/
s_app/q_penalty를 아예 만들지 않는다. s_app<=S*cos(pi/N)까지 남기면 원 제약이 3.53% 더
좁아져 "등가"가 깨지므로 폴리곤 자체를 뺐다 - lower_lp.py 원본의 force_q_zero=True 경로가
"다각형은 걸지 않는다"는 것과 동일한 관례), `'pcs_only'`(기준2, Q는 pwl/qp와 동일하게
자유·다각형/s_app/q_penalty/SOC/사이클 등식 전부 그대로 - **loss_term만 0**으로 둬 손실
항의 존재 자체가 만드는 비용을 손실 항 크기와 분리해서 잰다). 두 기준 모두 3개 통제점
전부에서, 방식 목록(PWL/QP)보다 **먼저** 측정·출력한다. solve_time 옆 배율의 기준을
기준1로 교체하고 라벨에 명시했다 - REFERENCE_SOLVE_TIME_SEC은 실행 메타에 참고용으로만
남긴다. 실행 메타에 CPU 모델명(`platform.processor()`)·논리코어수(`os.cpu_count()`)도
추가해(추가 의존성 없음) 머신 간 비교 근거를 남겼다.

## 방식 A: PWL (조각선형)
Q를 M개 구간으로 나누고 구간별 "실측" 기울기(시컨트, 근사·대표값 아님)를 선형 계수로 준다.
전 32버스 x AVG_DAYS 3개 x 24시간에서 Q_BOUNDARY_POINTS 각 점의 손실을 직접 조류계산으로
실측해(probe_q_residual.py의 단일버스 실험을 전면 확장한 것) 세그먼트 경계마다 실제
시컨트 기울기를 쓴다 - 근사는 "구간 안에서 선형"이라는 가정 하나뿐이다.

## 방식 B: QP (2차 손실)
LinDistFlow가 이미 계산하는 분기조류 P_e,Q_e를 그대로 재사용해
Sum_e r_e*(P_e^2+Q_e^2)*SMP*DT를 목적함수에 비용으로 더한다(V_ref=1.0 고정 - 방사형
배전망 전압이 1.0 pu 근방에 머무는 것을 이용한 근사, 1절: 기저 전압범위 0.962~1.02).
DPP 유지를 위해 위 "QP 전개" 절대로 전개형을 쓴다.

## PCS 손실 항 (양쪽 공통) - s_app을 다각형의 고정 상수 대신 변수로 승격
기존 lower_lp.py는 force_q_zero=False일 때 다각형 상한을 고정 상수 S*cos(pi/12)로 건다.
이 프로토타입은 그 상수를 변수 s_app(다각형이 실제로 근사하는 피상전력 상한)으로 바꾸고,
q_penalty = max(0, s_app-(P_ch+P_dis))에 비용을 매긴다 - Q=0이면 최소화 목적상 s_app이
|P_net|까지 줄어들어 q_penalty가 정확히 0이 되므로(회귀 앵커 보존), Q>0일 때만 "PCS가
그만큼 더 여유를 확보하는 데 드는 비용"이 원화로 매겨진다. C_PCS=1-ETA_PCS를 계수로 써서
benefits.loss_pcs와 동일한 물리 상수를 공유한다(새 상수 발명 안 함).

**★ "기존 42개 테스트" 재확인 불필요 - 이 스크립트는 lower_lp.py를 한 줄도 수정하지 않는다.**
여기서 시도하는 solver 설정(CLARABEL/OSQP 우선순위, tolerance/max_iter 조정)은 이
프로토타입 파일 안의 독자적인 `cp.Problem` 인스턴스에만 적용되고, lower_lp.py의
`_PROBLEM_CACHE`/`solve_avg`/`solve_peak`와는 아무 상태도 공유하지 않는다.

★★ lower_lp.py 원본은 이 스크립트 전체에서 한 줄도 수정하지 않는다. 아래
_build_problem_proto()가 lower_lp._build_problem()의 구조(SOC/PCS개별한계/LinDistFlow/
전압유도항)를 복사해 확장한 것이고, lower_lp._get_topology()/_prepare_common()만
읽기 전용으로 재사용한다. **solve_peak 손실 항 실험은 하지 않는다**(편익 함수
재구조화 후 별도 안건 - 3차 개정 통제 설계 절 참조).

실행: `python scripts/probe_lp_loss_proto.py`  (★ 이 스크립트는 작성만 하고 실행하지 않는다 -
실행은 사용자가 터미널에서 직접 한다. 입력으로 scripts/results/의 가장 최근
probe_q_selective_*.csv가 필요하다 - 없으면 즉시 에러로 알린다. 32버스x3시나리오x24h
x8경계점(QP 단위 수정 라운드에서 0.0375 추가) = 18,432회 조류계산 + LP는
4방식x3포인트x3시나리오x(워밍업1+타이밍5회) = 216회 + 검산 2-3용 3회(통제점당 1회) -
컴파일과 순수 solve 시간을 분리 측정하기 위한 의도된 증가, LP 1회는 조류계산보다
훨씬 가볍다.)

# ------------------------------------------------------------------
# 무엇을 확정했는가 (실행 후 채울 것 - scripts/ 규약, CLAUDE.md 부록A 참조)
#   실행 일시:
#   머신 사양:
#   결론:
# ------------------------------------------------------------------
"""

import os
import sys
import csv
import glob
import time
import socket
import datetime
import platform   # 계측 추가 라운드 작업C-4: CPU 모델명 - 추가 의존성 없이 얻을 수 있는 범위
import contextlib   # 작업 지시(9차세션) 작업2: _count_runpp_calls 컨텍스트매니저용

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cvxpy as cp
import pandapower as pp

import params as PM
import evaluate
import lower_lp

from probe_q_value import (
    POINTS,
    TARGET_PF,
    _build_net_with_pf,
    _prepare_condition,
    _restore_evaluate_state,
    _evaluate_with_force_q,
    section,
    _check_env,
)
from probe_q_sensitivity import _reinject_and_evaluate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')          # ★ scripts/results 관례
ROOT_RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'results')

# ============================================================
# 상수
# ============================================================

# 세그먼트 경계 후보의 합집합 - 이 점들에서만 실측(Loss_line, P_inj=0 고정 -
# probe_q_residual.py와 동일 규약)하고, 각 M의 세그먼트는 이 집합의 부분집합만 쓴다.
# ★ 3차 개정: 0.01/0.025/0.05/0.10을 추가했다 - 실제 q_star(AVG_DAYS)가 0.02~0.05
# 범위에 몰려 있는데(모듈 docstring "1차 실험" 절 실측값) 2차의 경계
# {0,0.05,0.15,0.3,0.5}는 전부 그 범위 **위**에 있어 M을 늘려도 실측 데이터를 가로지르는
# 분할이 하나도 안 생겼다 - PWL M별 정확도 비교가 애초에 성립하지 않았다. 0.5는 지시서
# 권장목록({0,0.01,0.025,0.05,0.10,0.30})에는 없지만 probe_q_selective.py의 격자탐색
# 상한(Q_GRID 최댓값 0.5)과 맞춰 뒀다 - 0.30에서 끊으면 PWL이 표현 가능한 Q의 최댓값이
# 기준해 탐색 상한보다 구조적으로 좁아져, 개별 시각의 q_star가 0.30을 넘는 경우(드물지만
# P3처럼 큰 유닛에서는 배제 못함) "근사오차"가 아니라 "표현 불가"가 섞여 PWL 유효성
# 판정이 오염된다(세그먼트 합의 상한이 곧 최상단 경계값이라는 구조 - 아래
# _build_problem_proto의 Q_seg<=delta_m 제약 참조).
Q_BOUNDARY_POINTS = [
    0.0, 0.01, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.10, 0.125,
    0.15, 0.175, 0.2, 0.25, 0.30, 0.4, 0.5,
]   # Mvar
# PWL 회귀 통제: 기존 11개 점의 조류계산 순서를 먼저 그대로 보존하고, 신규 M=9 중점은
# 뒤에서 측정한다. 저장 배열은 Q_BOUNDARY_POINTS의 오름차순 인덱스를 계속 사용한다.
Q_MEASUREMENT_ORDER = [
    0.0, 0.01, 0.025, 0.0375, 0.05, 0.075, 0.10, 0.15, 0.2, 0.30, 0.5,
    0.0625, 0.125, 0.175, 0.25, 0.4,
]
assert set(Q_MEASUREMENT_ORDER) == set(Q_BOUNDARY_POINTS)
# ★ 4차 개정(QP 단위 수정 라운드): 0.0375(0.025~0.05 구간의 중점)를 추가했다 -
# Q_DIAG_LEVELS의 두 번째 진단점(SEGMENT_BOUNDARIES[4]의 경계 "사이" 지점)을
# 근사·보간이 아니라 실측으로 대조하기 위함이다(이 스크립트의 다른 모든 "실측"과
# 동일한 원칙 - 근사식 추정 금지).
# ★ 계측 추가 라운드(2차분): 0.075/0.15/0.2를 추가했다(8점->11점, 실측 조류계산
# 비용 +37%) - SEGMENT_BOUNDARIES[9]가 [0.05,0.5] 구간(직전 라운드에서 다각형 한계
# PCS비용 임계 때문에 PWL이 못 채우던 바로 그 구간, 모듈 docstring "B-3" 참조)을
# 최적점(QP 실측 ~0.1245) 근방에서 세분하는 데 쓴다.

# M=1: 미세분 없음(2차와 동일하게 기준선으로 유지 - "분할 자체의 효과"를 M=2/4와 대조).
# M=2: 분할점 하나를 실측 범위(0.02~0.05)의 중앙 근방(0.025)에 둬 최소한의 분할로도
#   실제 데이터를 가로지르게 한다.
# M=4: 0~0.05 구간을 셋으로 세분(0.01/0.025/0.05)해 실측 범위 내부의 해상도를 최대로
#   높이고, 마지막 구간(0.05~0.5)이 꼬리(P3의 큰 개별값)를 담당한다.
# M=9(계측 추가 라운드 신규): 기존 M=1/2/4는 손대지 않고 추가만 한다(지시서 "통제 유지").
#   [0.05,0.5] 구간을 다각형 한계비용 임계 근방(0.075/0.1/0.15/0.2/0.3)에서 세분해,
#   PWL이 그 구간을 세분해도 여전히 임계(PCS_FACET_THRESHOLD, 아래 정의)를 넘는 세그먼트만
#   골라 채우는지(=한계가 세그먼트 폭이 아니라 다각형 구조 자체에 있다는 가설의 재확인)를
#   본다 - 모듈 docstring "B-3" 절 참조.
SEGMENT_BOUNDARIES = {
    1: [0.0, 0.5],
    2: [0.0, 0.025, 0.5],
    4: [0.0, 0.01, 0.025, 0.05, 0.5],
    9: [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5],
}

# ★ 계측 추가 라운드 검산(작업B-2): SEGMENT_BOUNDARIES의 모든 경계가 Q_BOUNDARY_POINTS의
# 부분집합이어야 텔레스코핑 항등식(_segment_slopes가 그 점들에서 실측된 손실만으로 시컨트를
# 만드는 전제)이 성립한다 - 모듈 로드 시 1회, 전 M에 대해 확인한다.
for _M_check, _boundaries_check in SEGMENT_BOUNDARIES.items():
    assert set(_boundaries_check).issubset(set(Q_BOUNDARY_POINTS)), (
        f"SEGMENT_BOUNDARIES[{_M_check}]={_boundaries_check}가 Q_BOUNDARY_POINTS"
        f"({Q_BOUNDARY_POINTS})의 부분집합이 아니다 - 텔레스코핑 항등식이 깨진다."
    )
del _M_check, _boundaries_check

ALL_BUSES = list(range(PM.B_BOUNDS[0], PM.B_BOUNDS[1] + 1))   # 1..32

C_PCS = 1.0 - PM.ETA_PCS   # benefits.loss_pcs와 동일 물리상수 재사용(새로 만들지 않음)

# ★ 계측 추가 라운드(작업B-3/B-4) - 다각형 facet k의 한계 PCS 비용은 ds_app/dQ=sin(theta_k)
# (theta_k=2*pi*k/POLY_N)에 비례한다. 가장 작은 양의 계단(k=1)이 PWL 세그먼트가 넘어야
# 하는 실질적 임계다(직전 실행에서 규명됨 - 모듈 docstring "B-3" 참조). 판정에 쓰지 않고
# 세그먼트 기울기 표에 값만 나란히 출력한다(지시서: "임계값 자체도 상수로 출력하라... 판정은
# 하지 마라").
PCS_FACET_THRESHOLD = C_PCS * np.sin(2.0 * np.pi / PM.POLY_N)

CONTROL_POINT_BUSES = [int(p['b']) for p in POINTS]   # 계측 추가 라운드 작업A-1 - 3개 통제점의
                                                         # 버스(POINTS에서 유도, 하드코딩 안 함)

# ★ 4차 개정(QP 단위 수정 라운드) 검산 2-1: POLY_N 짝수 검사 (모듈 로드 시 1회).
# 근거: q_penalty>=s_app-P_ch-P_dis가 Q=0에서 정확히 0이 되려면(회귀 앵커, 검산 2-3)
# max_k(P_net*cos(theta_k))가 |P_net|이어야 한다. P_net<0(충전)일 때 이 값이 |P_net|이
# 되려면 theta=pi가 다각형 꼭짓점 격자(theta_k=2*pi*k/POLY_N, k=0..POLY_N-1)에 있어야
# 하고, 이는 POLY_N이 짝수일 때만 성립한다(POLY_N/2가 정수여야 k=POLY_N/2에서
# theta_k=pi). 홀수면 충전 시각에서 Q=0인데도 q_penalty>0이 되어 회귀 앵커가 깨진다.
assert PM.POLY_N % 2 == 0, (
    f"PM.POLY_N={PM.POLY_N}는 짝수여야 한다 - 홀수면 theta=pi가 다각형 꼭짓점 격자에 "
    "없어 충전(P_net<0) 시각의 Q=0 회귀 앵커(q_penalty=0)가 구조적으로 깨진다."
)

# ★ 계측 추가 라운드 작업C-1/C-3: 이 상수는 더 이상 solve_time 배율의 기준으로 쓰지
# 않는다. (i) evaluate.py 전체(LP 5시나리오+사후 조류계산+편익집계)를 잰 것이고
# (ii) 다른 머신에서 측정됐으며 (iii) 이 스크립트의 solve_avg 3시나리오와 범위가 달라
# "손실 항 추가로 몇 배 느려졌는가"의 기준이 될 수 없다(실제로 이 기준으로는 "손실 항을
# 넣었더니 빨라졌다"는 있을 수 없는 결론이 나왔다). **다른 머신·다른 범위의 참고값으로만**
# 남겨 두고(실행 메타에서 1회 출력), 배율은 아래 기준1(force_q_zero=True 등가, 같은
# 스크립트·같은 통제점·같은 타이밍 하네스로 실측)로 교체했다 - _print_method_summary 참조.
REFERENCE_SOLVE_TIME_SEC = 0.549

# ★ 계측 추가 라운드 작업B-2: 기존 M=1/2/4는 그대로 두고 실행 순서 끝에 M=9를 추가한다
# (지시서: "PWL M=1 -> M=2 -> M=4 -> M=9 -> QP"). QP는 대조군이므로 이 라운드에서
# 일절 건드리지 않는다(정식화·순서상 위치 모두 - 통제 유지).
METHODS = [('pwl', 1), ('pwl', 2), ('pwl', 4), ('pwl', 9), ('qp', None)]

TS_CSV_FIELDS = ['method', 'M', 'point_id', 'scenario', 't', 'q_lp', 'q_star', 'abs_err',
                  'rel_err', 'inaccurate',
                  # ★ 4차 개정(QP 단위 수정 라운드) 지시서 2-5 - (method,M,point) 단위
                  # 상수를 모든 ts_row에 반복해 담는다(이 CSV의 기존 비정규화 관례와 동일 -
                  # probe_q_value.py의 s_total_mva/power_factor_actual 반복 패턴 참조).
                  'free_zone_width_mvar', 'max_uncounted_loss_mva',
                  'n_free_zone_hours', 'n_vertex_stuck_hours',
                  # ★ 작업 C(무료구간 필터 수정) 추가 컬럼 - 분모(비영 시각 수)와 그 비.
                  'n_nonzero_q_hours', 'frac_in_free_zone',
                  # ★ 작업 A(검산 2-2 재설계) 추가 컬럼 - (method,M,point) 단위 상수 반복.
                  'max_pad_mw', 'max_pad_scenario', 'max_pad_t', 'max_pad_unit',
                  'max_pad_annual_won_implied',
                  # ★ 작업 B(QP V^2 보정) - 이 실행에서 실제 최적화 경로가 보정을 썼는지.
                  'qp_v2_correction',
                  # ★ 계측·진단 라운드 작업 1: 기존 열은 그대로 두고 뒤에만 추가한다.
                  # 모두 워밍업 solve의 확정값이며 타이밍 반복 solve 값은 사용하지 않는다.
                  'p_ch', 'p_dis', 'p_net', 's_app', 'q_penalty',
                  'pcs_true', 'pcs_charged']

# ★ 4차 개정 신규 상수
PCS_CIRCLE_TOL = 1e-9        # sqrt(P^2+Q^2)<=S*(1+tol) 판정 여유(부동소수점, 지시서 검산)
N_TIMING_REPS = 5            # 워밍업 이후 반복 측정 횟수(지시서 "속도 측정 설계 수정")

# ★ 4차 개정(QP 단위 수정 라운드) 신규 상수
# ★ 계측·진단 라운드 작업 2: 직전 운전점의 Q 범위 전체에서 alpha=A/dLoss를 잰다.
# 전부 기존 Q_BOUNDARY_POINTS에 이미 있는 실측점이어야 하며, 아래 assert가 이를 강제한다.
Q_DIAG_LEVELS = [
    0.0375, 0.05, 0.0625, 0.075, 0.1, 0.125,
    0.15, 0.175, 0.2, 0.25, 0.3, 0.4,
]
assert set(Q_DIAG_LEVELS).issubset(set(Q_BOUNDARY_POINTS)), (
    f"Q_DIAG_LEVELS={Q_DIAG_LEVELS} 중 Q_BOUNDARY_POINTS={Q_BOUNDARY_POINTS}에 없는 값이 "
    "있다. 진단점을 추가 조류계산으로 보충하지 말고 기존 실측점만 사용해야 한다."
)
Q_ZERO_ANCHOR_TOL = 1e-9         # q_penalty==0(Q=0 고정 시) 판정 여유(지시서 2-3)

# ★ 4차 개정 확장 라운드(검산 2-2 재설계) - 지시서 작업 A.
# 이전 상수 PCS_COMPLEMENTARITY_TOL(P_ch*P_dis<=1e-9)은 위양성을 냈다: 곱의 단위는
# MW^2이고 그 스케일이 P_dis에 비례하므로, P_dis=0.173(정상 크기)이면 P_ch가
# 9.32e-9(수치 잡음 수준, CLAUDE.md 7절 실측 P_slack 잡음 1e-8~1e-9 MW와 같은 자릿수)만
# 되어도 곱이 1.611e-9로 tol을 넘었다 - 물리적 exploit이 아니라 검사량 설계 오류였다
# (CLAUDE.md 7절 "테스트 설계 원칙4" 위반: 비교 대상의 성격에 안 맞는 허용오차).
# 검사량을 곱이 아니라 패딩 크기 자체 pad=min(P_ch,P_dis)[MW]로 바꾸고, 그 물리적
# 금전 영향을 직접 유도해 임계로 쓴다.
#
# 패딩 eps[MW]가 연간 j_net에 주는 영향(지시서 유도):
#   연간영향 ~= C_PCS * SMP * 2 * (연간 대표일수 가중합) * (해당 시각수) * eps
#            ~= 0.03 * 140,000원/MWh * 2 * 247일 * 10시각 * eps
#            ~= 2.08e7 * eps  [원/년]
#   eps=4.8e-6 MW -> 약 100원/년,  eps=4.8e-5 MW -> 약 1,000원/년.
#   관측된 솔버 잔차 규모(1e-8~1e-9 MW)는 연간 0.02~0.2원 수준 - 완전히 무해하다.
ANNUAL_WON_PER_PAD_MW = 2.08e7    # 원/년 per MW 패딩 (위 유도값)
PAD_WARN_MW = 1e-6                # ~20.8원/년 - 넘으면 경고 수집(중단하지 않음)
PAD_ABORT_MW = 1e-4                # ~2,080원/년 - 넘으면 즉시 AssertionError로 중단

# ★ 작업 B(QP 손실항 V^2 보정) 신규 상수. True/False로 켜고 끌 수 있게 해 보정 전/후를
# 나란히 비교할 수 있게 한다(지시서 B-4). _process(실제 최적화 경로)는 이 값을 따르고,
# _diagnose_q_prediction_gap(진단 전용)은 이 값과 무관하게 항상 양쪽을 다 계산해 보고한다.
QP_V2_CORRECTION = True

# ★ QP 보정 적용 라운드: 두 플래그는 QP 최적화 경로에만 적용한다. PWL/기준 경로는
# `_set_params`의 method 분기상 이 값을 읽지 않는다. 진단 함수는 네 변형을 계속 모두 계산한다.
QP_QE_BASE_AC = True
QP_GROSSUP = True
# 2026-07-28 직전 실행(probe_lp_loss_proto_PSL_20260728_110753)의 p=0 실측 alpha.
# 통제점별 원값: P1=0.9526, P2=0.9524, P3=0.9555. 아래 값은 세 원값의 median이며,
# 실측 alpha이지 튜닝 파라미터가 아니다. 실행 중 재측정값과의 차이는 stdout/report에 낸다.
QP_GROSSUP_ALPHA = 0.9526

# ★ 작업 C(무료구간 계측 필터 수정) 신규 상수. 기존 in_zone 조건(arr>0.0)은 부동소수점
# 잡음(예: 1e-12처럼 수치적으로는 0인데 엄밀히는 양수인 값)까지 "비영"으로 세어 P1에서
# 71/72라는 사실상 전체를 무료구간으로 오판했다. _group_stats가 이미 쓰는 것과 동일한
# 비영 판정 기준(1e-6 Mvar)을 여기서도 재사용한다(새 상수를 만들지 않고 일관성 유지).
FREE_ZONE_NONZERO_TOL = 1e-6


# ============================================================
# 1) 손실 테이블 실측 (probe_q_residual.py 방식의 전 버스 확장, AVG_DAYS 한정)
# ============================================================

def _branch_line_idxs(net):
    """lower_lp._build_topology()와 정확히 동일한 branch 정렬 규칙(sorted(lines.index))을
    재현한 pandapower 원본 line 인덱스 리스트(압축 안 함) - r_pu/D/from_bus_arr와 같은
    branch 순서를 보장하기 위해 이 정렬을 모든 branch-indexed 테이블이 공유한다."""
    lines = net.line[net.line['in_service']]
    return sorted(lines.index)


def _branch_from_bus_array(net):
    """각 branch(=_branch_line_idxs 순서)의 from_bus. 작업 B(V^2 보정)의 "송단(from bus)
    전압" 규약에 쓰인다(lower_lp.py 원본은 건드리지 않는다 - net.line만 읽기 전용으로 읽음)."""
    line_idxs = _branch_line_idxs(net)
    return np.array([int(net.line.at[idx, 'from_bus']) for idx in line_idxs])


def _branch_to_bus_array(net):
    """각 branch(=_branch_line_idxs 순서)의 to_bus - 작업C(손실공식 항등성 검증)의
    "수단(to bus)" 전압에 쓰인다."""
    line_idxs = _branch_line_idxs(net)
    return np.array([int(net.line.at[idx, 'to_bus']) for idx in line_idxs])


@contextlib.contextmanager
def _count_runpp_calls():
    """작업 지시(9차세션) 작업2: pp.runpp 실제 호출 횟수를 호출 지점에서 직접 센다
    (루프 크기 x 경계점 수 등으로 역산하지 않는다 - _measure_loss_table의 조류계산은
    evaluate._run_pf_with_retry를 거치는데, 그 함수는 1차(init='results')가 실패하면
    2차(init='flat')로 재시도한다(evaluate.py:77-91) - 즉 루프 1회당 runpp 호출이
    1회가 아니라 최대 2회일 수 있어 루프 크기로부터의 역산은 부정확할 수 있다.

    evaluate.py는 `import pandapower as pp`로 pp.runpp를 매 호출 시점에 속성조회하므로,
    여기서 전역 pandapower 모듈의 runpp 속성 자체를 계측 래퍼로 바꿔치기하면 evaluate.py
    안에서(파일 자체는 수정하지 않고) 일어나는 호출까지 같은 모듈 객체를 통해 셀 수 있다.
    래퍼는 원본 pp.runpp를 인자 그대로 호출·반환할 뿐이므로 조류계산의 동작·결과·연산
    순서에는 영향이 없다(카운터 증분만 추가) - with 블록을 벗어나면(정상/예외 무관) 원본으로
    복원한다."""
    counter = {'n': 0}
    original_runpp = pp.runpp

    def _wrapped_runpp(*args, **kwargs):
        counter['n'] += 1
        return original_runpp(*args, **kwargs)

    pp.runpp = _wrapped_runpp
    try:
        yield counter
    finally:
        pp.runpp = original_runpp


def _measure_loss_table(net, base_p, base_q):
    """AVG_DAYS x 24h x 전 32버스 x Q_BOUNDARY_POINTS에서 Loss_line(P_inj=0,Q_inj=q)를
    실측한다. 반환: (loss_table, v_sq_line_table, ac_flow_table, v_bus_table).
    loss_table[bus][scenario] = ndarray shape (24, len(Q_BOUNDARY_POINTS)).
    P_inj=0 고정 - "Q 단독의 손실저감 효과"를 재는 것이 이 실험의 정의이기 때문이다.
    ★ 3차 개정: PEAK_DAYS 실측을 뺐다(solve_peak 프로토타입 자체가 사라졌으므로 그
    계수가 더 이상 필요 없다 - 2차의 ALL_DAYS 확장을 되돌린 것).

    ★ 작업 B(QP V^2 보정) 신규: v_sq_line_table[scenario] = ndarray(n_branch,T) -
    기저(ESS 없음) from-bus 전압 제곱. bus=ALL_BUSES[0]의 q=0.0(=p=0도 함께 0이므로
    이 sgen은 완전한 no-op - 어느 버스에 달려 있든 결과에 영향이 없다) 패스를 그
    "기저 조류계산"으로 재사용해 캡처한다 - **추가 조류계산을 전혀 돌리지 않는다**
    (지시서 요구사항). 이 값은 (scenario,t)에만 의존하고 bus 루프와 무관하므로 첫
    버스에서 한 번만 캡처하면 전 버스가 공유해도 정확하다.

    ★ 계측 추가 라운드(작업 A/B/C) 신규 - 기존 v_sq_line_table 캡처와 **정확히 같은
    호출(같은 if 블록 안)**에서 추가로 읽는다(추가 pp.runpp 없음):
      ac_flow_table[scenario] = dict(q_from,q_to,p_from,p_to,pl,ql) 각 ndarray(n_branch,T) -
        작업B(Qe_base vs AC 실측 대조)·작업C(손실공식 항등성 검증)가 함께 쓴다.
      v_bus_table[scenario] = ndarray(n_bus,T) - 전 버스 vm_pu(제곱 안 함, raw) - 작업C가
        to_bus 전압을 뽑는 데 쓴다(기존 v_sq_line_table은 from_bus만 이미 제곱해 저장하므로
        재사용하지 않고 별도로 둔다 - 기존 테이블의 형태·의미는 손대지 않는다는 원칙).

    ★ 계측 추가 라운드(작업A-1) 신규 - ac_full_table[bus][scenario] = dict(p_from,q_from,
    p_to,q_to,pl,vm_bus) 각 ndarray(T,len(Q_BOUNDARY_POINTS),n_branch 또는 n_bus) -
    CONTROL_POINT_BUSES(3개)에 한해 **모든 q 지점**에서 캡처한다(위 v_sq_line_table 등은
    bus==ALL_BUSES[0]의 q==0.0 지점 1개뿐이라 별개 - 이 테이블은 그 제약을 없앤 것).
    같은 t,qi 축을 공유하므로 "q=0 상태"와 "q=q_level 상태"가 같은 (scenario,t)에서
    왔음이 인덱싱 자체로 보장된다(별도 시각 매칭 로직 불필요 - 작업A-2 참조). 새 pp.runpp를
    추가하지 않는다 - 이미 도는 (bus,s,t,qi) 루프에서 조건만 넓힌다(bus==ALL_BUSES[0] and
    q==0.0 한 지점 -> bus in CONTROL_POINT_BUSES인 전 q 지점)."""
    if len(net.sgen) == 0:
        pp.create_sgen(net, bus=ALL_BUSES[0], p_mw=0.0, q_mvar=0.0, name='probe_lp_loss_proto')
    sgen_idx = net.sgen.index[0]

    line_idxs = _branch_line_idxs(net)
    from_bus_arr = _branch_from_bus_array(net)
    n_branch = len(from_bus_arr)
    n_bus = len(net.bus)
    n_q = len(Q_BOUNDARY_POINTS)
    v_sq_line_table = {s: np.zeros((n_branch, PM.TIME_STEPS)) for s in PM.AVG_DAYS}
    v_bus_table = {s: np.zeros((n_bus, PM.TIME_STEPS)) for s in PM.AVG_DAYS}
    ac_flow_table = {
        s: dict(
            q_from=np.zeros((n_branch, PM.TIME_STEPS)), q_to=np.zeros((n_branch, PM.TIME_STEPS)),
            p_from=np.zeros((n_branch, PM.TIME_STEPS)), p_to=np.zeros((n_branch, PM.TIME_STEPS)),
            pl=np.zeros((n_branch, PM.TIME_STEPS)), ql=np.zeros((n_branch, PM.TIME_STEPS)),
        ) for s in PM.AVG_DAYS
    }
    ac_full_table = {
        bus: {
            s: dict(
                p_from=np.zeros((PM.TIME_STEPS, n_q, n_branch)),
                q_from=np.zeros((PM.TIME_STEPS, n_q, n_branch)),
                p_to=np.zeros((PM.TIME_STEPS, n_q, n_branch)),
                q_to=np.zeros((PM.TIME_STEPS, n_q, n_branch)),
                pl=np.zeros((PM.TIME_STEPS, n_q, n_branch)),
                vm_bus=np.zeros((PM.TIME_STEPS, n_q, n_bus)),
            ) for s in PM.AVG_DAYS
        } for bus in CONTROL_POINT_BUSES
    }

    loss_table = {}
    n_total = len(ALL_BUSES) * len(PM.AVG_DAYS)
    done = 0
    for bus in ALL_BUSES:
        loss_table[bus] = {}
        for s in PM.AVG_DAYS:
            profile = PM.LOAD[s]
            arr = np.zeros((PM.TIME_STEPS, len(Q_BOUNDARY_POINTS)))
            for t in range(PM.TIME_STEPS):
                scale = profile[t]
                net.load['p_mw'] = base_p * scale
                net.load['q_mvar'] = base_q * scale
                for q in Q_MEASUREMENT_ORDER:
                    qi = Q_BOUNDARY_POINTS.index(q)
                    net.sgen.at[sgen_idx, 'bus'] = bus
                    net.sgen.at[sgen_idx, 'p_mw'] = 0.0
                    net.sgen.at[sgen_idx, 'q_mvar'] = q
                    ok = evaluate._run_pf_with_retry(net)
                    if not ok:
                        raise RuntimeError(
                            f'조류계산 발산: bus={bus} s={s} t={t} q={q} - 정상범위 비정상.'
                        )
                    arr[t, qi] = float(net.res_line.pl_mw.sum())
                    if bus == ALL_BUSES[0] and q == 0.0:
                        v_bus_all = net.res_bus.vm_pu.to_numpy()
                        v_sq_line_table[s][:, t] = (v_bus_all ** 2)[from_bus_arr]
                        v_bus_table[s][:, t] = v_bus_all
                        ac_flow_table[s]['q_from'][:, t] = net.res_line.loc[line_idxs, 'q_from_mvar'].to_numpy()
                        ac_flow_table[s]['q_to'][:, t] = net.res_line.loc[line_idxs, 'q_to_mvar'].to_numpy()
                        ac_flow_table[s]['p_from'][:, t] = net.res_line.loc[line_idxs, 'p_from_mw'].to_numpy()
                        ac_flow_table[s]['p_to'][:, t] = net.res_line.loc[line_idxs, 'p_to_mw'].to_numpy()
                        ac_flow_table[s]['pl'][:, t] = net.res_line.loc[line_idxs, 'pl_mw'].to_numpy()
                        ac_flow_table[s]['ql'][:, t] = net.res_line.loc[line_idxs, 'ql_mvar'].to_numpy()
                    if bus in CONTROL_POINT_BUSES:
                        tbl = ac_full_table[bus][s]
                        tbl['vm_bus'][t, qi, :] = net.res_bus.vm_pu.to_numpy()
                        tbl['p_from'][t, qi, :] = net.res_line.loc[line_idxs, 'p_from_mw'].to_numpy()
                        tbl['q_from'][t, qi, :] = net.res_line.loc[line_idxs, 'q_from_mvar'].to_numpy()
                        tbl['p_to'][t, qi, :] = net.res_line.loc[line_idxs, 'p_to_mw'].to_numpy()
                        tbl['q_to'][t, qi, :] = net.res_line.loc[line_idxs, 'q_to_mvar'].to_numpy()
                        tbl['pl'][t, qi, :] = net.res_line.loc[line_idxs, 'pl_mw'].to_numpy()
            loss_table[bus][s] = arr
            done += 1
            if done % 20 == 0 or done == n_total:
                print(f'  손실 테이블 실측 진행: {done}/{n_total} (bus={bus}, scenario={s})',
                      flush=True)
    return loss_table, v_sq_line_table, ac_flow_table, v_bus_table, ac_full_table


def _segment_slopes(arr_row, boundaries):
    """arr_row: len(Q_BOUNDARY_POINTS) 배열(한 t의 Loss_line 실측값, Q_BOUNDARY_POINTS와 같은
    순서). boundaries는 Q_BOUNDARY_POINTS의 부분집합(오름차순)이어야 한다. 반환:
    len(boundaries)-1개의 실측 시컨트 기울기(LHS_m, m=0..M-1, MW/Mvar, 양수=손실감소)."""
    idx = {q: i for i, q in enumerate(Q_BOUNDARY_POINTS)}
    slopes = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        loss_lo = arr_row[idx[lo]]
        loss_hi = arr_row[idx[hi]]
        slopes.append(float((loss_lo - loss_hi) / (hi - lo)))
    return slopes


def _lhs_rows_for(loss_table, bus, scenario, M):
    """(bus,scenario)의 24시간 각각에 대해 _segment_slopes를 적용해 (M,T) 배열을 만들고,
    lhs_params[m].value에 넣을 (1,T) 조각들의 리스트로 반환한다(n=1 고정)."""
    boundaries = SEGMENT_BOUNDARIES[M]
    arr = loss_table[bus][scenario]   # (T, len(Q_BOUNDARY_POINTS))
    T = arr.shape[0]
    slopes_by_t = np.array([_segment_slopes(arr[t], boundaries) for t in range(T)])  # (T,M)
    return [slopes_by_t[:, m][None, :] for m in range(M)]   # M개의 (1,T) 배열


def _check_pwl_slope_monotonicity(loss_table, bus, scenario, M):
    """★ 4차 개정(QP 단위 수정 라운드) 검산 2-4 (경고만 - 중단하지 않음): PWL이
    "세그먼트를 순서대로(0번부터) 채운다"는 가정과 실제로 맞물리려면 세그먼트 기울기가
    m=0..M-1에서 단조감소(손실이 Q에 대해 볼록/체감)여야 한다. 현재 PWL은 각
    세그먼트를 자신의 폭(delta_m)까지 채우는 것을 허용할 뿐 "이전 세그먼트를 먼저
    다 채워야 한다"는 순서 제약을 걸지 않는데, 목적함수가 세그먼트별 기울기(이득)를
    보고 자유롭게 배분하므로, 기울기가 비단조(뒤 세그먼트가 앞보다 이득이 큼)면 LP가
    물리적 순서를 무시하고 이득 큰 세그먼트부터 채워 손실편익을 과대평가할 수 있다.
    _build_problem_proto 주석("Q_flow 부호가 뒤집혀 손실이 오히려 늘 수 있다")이 이미
    비단조 구간이 실재할 수 있음을 경고하고 있다. 위반은 AssertionError가 아니라
    경고로만 보고한다 - 처리 방법은 사용자가 판단한다(지시서 2-4)."""
    boundaries = SEGMENT_BOUNDARIES[M]
    arr = loss_table[bus][scenario]
    violations = []
    for t in range(PM.TIME_STEPS):
        slopes = _segment_slopes(arr[t], boundaries)
        for m in range(len(slopes) - 1):
            if slopes[m + 1] > slopes[m] + 1e-9:   # 다음 구간 기울기가 더 큼 = 비단조
                violations.append((scenario, t, m, slopes[m], slopes[m + 1]))
    return violations


def _print_pwl_monotonicity_check(loss_table):
    """M=2/4/9에 대해 전 통제점 x AVG_DAYS를 훑어 비단조 구간을 보고한다(M=1은 세그먼트가
    1개뿐이라 단조성 정의 자체가 없음). 경고만 출력 - 중단하지 않는다(지시서 2-4).
    ★ 계측 추가 라운드: M=9 추가(기존 M=2/4는 그대로)."""
    section('PWL 세그먼트 기울기 단조성 점검 (경고만 - 위반해도 중단하지 않음)')
    total_violations = 0
    for point in POINTS:
        bus = point['b']
        for M in (2, 4, 9):
            for s in PM.AVG_DAYS:
                violations = _check_pwl_slope_monotonicity(loss_table, bus, s, M)
                total_violations += len(violations)
                for (scenario, t, m, slope_m, slope_m1) in violations[:5]:
                    print(f"  ⚠ {point['point_id']}(bus={bus}) M={M} scenario={scenario} "
                          f"t={t} m={m}->{m + 1}: 기울기 {slope_m:.6f} -> {slope_m1:.6f} "
                          "(증가 - 비단조)", flush=True)
                if len(violations) > 5:
                    print(f"    ... 같은 조합에서 {len(violations) - 5}건 더", flush=True)
    print(f"\n  총 위반 건수 = {total_violations} (0이면 전 구간 단조감소 확인됨 - LP의 "
          "'세그먼트 순서대로 채움' 가정이 이 데이터에서는 안전하다는 뜻. 0이 아니면 "
          "비단조 구간이 실재하며 처리 방법은 판단이 필요하다 - 이 함수는 판정하지 않는다)",
          flush=True)


# ============================================================
# 1-A) 계측 추가 라운드 - 작업 A: PWL 세그먼트 기울기 실측 노출
# ============================================================

def _local_slope_near(loss_row, q_mid):
    """Q_BOUNDARY_POINTS 중 q_mid를 포함하는 가장 좁은 인접 구간의 실측 시컨트 기울기를
    낸다("그 구간 중앙값에서의 국소 기울기" - 지시서 A-3). 세그먼트 경계 자체가
    Q_BOUNDARY_POINTS의 원소이므로, 넓은 세그먼트(예: M=1의 [0,0.5])의 중앙값(0.25)은
    더 촘촘한 두 경계점([0.10,0.30]) 사이에 들어가 세그먼트 자신의 시컨트보다 해상도가
    높은 참고값이 나온다. q_mid가 Q_BOUNDARY_POINTS의 원소와 정확히 같으면(예: M=4의
    [0.025,0.05] 세그먼트 중앙값 0.0375는 그 자체가 실측점이다) 그 값을 포함하는 첫
    구간([0.025,0.0375])을 쓴다 - 세그먼트의 앞쪽 절반만큼의 더 촘촘한 해상도가 된다.
    반환: (slope, bracket_lo, bracket_hi) 또는 (None,None,None)(범위 밖일 때)."""
    pts = Q_BOUNDARY_POINTS
    idx = {q: i for i, q in enumerate(pts)}
    for i in range(len(pts) - 1):
        lo, hi = pts[i], pts[i + 1]
        if lo <= q_mid <= hi:
            slope = float((loss_row[idx[lo]] - loss_row[idx[hi]]) / (hi - lo))
            return slope, lo, hi
    return None, None, None


def _pwl_segment_slope_report(loss_table, bus, M):
    """M의 각 세그먼트에 대해 AVG_DAYS 72개 시각의 실측 시컨트 기울기 분포(min/p25/
    median/p75/max, MW/Mvar)와 기울기<=0인 (scenario,t) 개수, 그리고 그 구간 중앙값
    근방의 국소 기울기 분포(_local_slope_near)를 계산한다. 판정하지 않는다 - 수치만."""
    boundaries = SEGMENT_BOUNDARIES[M]
    n_seg = len(boundaries) - 1
    rows = []
    for m in range(n_seg):
        lo, hi = boundaries[m], boundaries[m + 1]
        q_mid = (lo + hi) / 2.0
        seg_vals, local_vals = [], []
        local_bracket = None
        for s in PM.AVG_DAYS:
            arr = loss_table[bus][s]
            for t in range(PM.TIME_STEPS):
                slopes = _segment_slopes(arr[t], boundaries)
                seg_vals.append(slopes[m])
                local_slope, l_lo, l_hi = _local_slope_near(arr[t], q_mid)
                if local_slope is not None:
                    local_vals.append(local_slope)
                    local_bracket = (l_lo, l_hi)
        seg_arr = np.array(seg_vals)
        pct = np.percentile(seg_arr, [0, 25, 50, 75, 100])
        n_nonpositive = int(np.sum(seg_arr <= 0.0))
        local_pct = (np.percentile(np.array(local_vals), [0, 25, 50, 75, 100])
                     if local_vals else None)
        rows.append(dict(m=m, boundary_lo=lo, boundary_hi=hi, delta_m=hi - lo,
                          slope_pct=pct, n_nonpositive=n_nonpositive,
                          local_bracket=local_bracket, local_pct=local_pct))
    return rows


def _print_pwl_segment_slopes(loss_table):
    """지시서 작업A-1: 통제점별 M=1/2/4/9를 나란히 출력한다. 판정 없음 - 수치만.
    ★ 계측 추가 라운드 작업B-4: 다각형 facet 한계비용 임계(PCS_FACET_THRESHOLD)를 값만
    함께 출력한다 - "넘는다/못 넘는다" 판정은 하지 않는다(지시서 요구)."""
    section('작업A: PWL 세그먼트 기울기 실측 분포 (판정 없음 - 수치와 위치만)')
    print(f"  참고: 다각형 facet 한계 PCS비용 임계 PCS_FACET_THRESHOLD = "
          f"C_PCS*sin(2*pi/POLY_N) = {PCS_FACET_THRESHOLD:.6f} - 아래 시컨트 기울기(MW/Mvar)"
          f"와 같은 단위계에서 직접 비교 가능하다(SMP*dt가 편익·비용 양쪽에 동일하게 곱해져"
          f" 부등식에서 상쇄되므로 - 배경 절 참조). 판정 아님, 값만 나란히 볼 것.", flush=True)
    for point in POINTS:
        bus = point['b']
        print(f"\n  {point['point_id']} (bus={bus}):", flush=True)
        for M in (1, 2, 4, 9):
            print(f"    M={M} (경계={SEGMENT_BOUNDARIES[M]}):", flush=True)
            for r in _pwl_segment_slope_report(loss_table, bus, M):
                pct = r['slope_pct']
                print(f"      m={r['m']} 구간=[{r['boundary_lo']},{r['boundary_hi']}] "
                      f"delta={r['delta_m']:.4f}: 시컨트기울기(MW/Mvar) "
                      f"min={pct[0]:.6f} p25={pct[1]:.6f} median={pct[2]:.6f} "
                      f"p75={pct[3]:.6f} max={pct[4]:.6f}  기울기<=0인 (s,t) 개수="
                      f"{r['n_nonpositive']}/72", flush=True)
                if r['local_pct'] is not None:
                    lb = r['local_bracket']
                    lp = r['local_pct']
                    q_mid = (r['boundary_lo'] + r['boundary_hi']) / 2.0
                    print(f"        (참고) 중앙값 Q={q_mid:.4f} 근방 국소기울기"
                          f"[{lb[0]},{lb[1]}]: min={lp[0]:.6f} p25={lp[1]:.6f} "
                          f"median={lp[2]:.6f} p75={lp[3]:.6f} max={lp[4]:.6f}", flush=True)


LHS_CSV_FIELDS = ['point_id', 'M', 'scenario', 't', 'm', 'boundary_lo', 'boundary_hi',
                   'delta_m', 'slope']


def _make_lhs_path():
    """지시서 작업A-2: 파일명 접미사 _lhs (probe_lp_loss_proto.py 본 CSV와 같은 호스트/
    타임스탬프 규약, scripts/results 관례)."""
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(RESULTS_DIR, f'probe_lp_loss_proto_lhs_{hostname}_{ts}.csv')


def _write_lhs_csv(path, loss_table):
    """지시서 작업A-2: 세그먼트 기울기는 시각별 값이라 ts_rows에 넣으면 행이 폭증하므로
    (point_id x M x scenario x t x m) 별도 CSV에 남긴다."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for point in POINTS:
        bus = point['b']
        for M in (1, 2, 4, 9):
            boundaries = SEGMENT_BOUNDARIES[M]
            for s in PM.AVG_DAYS:
                arr = loss_table[bus][s]
                for t in range(PM.TIME_STEPS):
                    slopes = _segment_slopes(arr[t], boundaries)
                    for m, slope in enumerate(slopes):
                        rows.append(dict(
                            point_id=point['point_id'], M=M, scenario=s, t=t, m=m,
                            boundary_lo=boundaries[m], boundary_hi=boundaries[m + 1],
                            delta_m=boundaries[m + 1] - boundaries[m], slope=slope,
                        ))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LHS_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    print(f'세그먼트 기울기 CSV 저장(작업A-2): {path} ({len(rows)}행)', flush=True)


# ============================================================
# 1-B) 계측 추가 라운드 - 작업 B: LinDistFlow Qe_base vs AC 실측 조류 대조
# ============================================================

def _compute_qe_base_mvar(topo, profile):
    """Qe_base[branch,t] = D@(하류 부하 무효전력 합)/S_BASE_MVA, 물리단위(Mvar)로 환산해
    반환한다 - _diagnose_q_prediction_gap이 이미 쓰는 것과 동일한 산식이다(그 함수의
    기존 지역 계산은 건드리지 않았다 - "정식화는 그대로, 계측만 추가" 원칙에 따라 이
    함수는 작업B 전용으로 별도로 새로 정의했을 뿐 리팩터링이 아니다)."""
    D = topo['D']
    _, base_load_q_bus = lower_lp.base_load_bus_arrays()
    load_q_val = base_load_q_bus[:, None] * np.asarray(profile, dtype=float)[None, :]
    Qe_base_pu = D @ (load_q_val / PM.S_BASE_MVA)
    return Qe_base_pu * PM.S_BASE_MVA


def _path_branches_for_bus(D, bus):
    """D[e,bus]==1인 branch 집합 = 슬랙~bus 경로(lower_lp.py의 D 행렬 정의 - "bus가 branch
    e의 하류(자손)"). probe_q_marginal.py의 path(i) 정의와 동일하다."""
    return np.where(D[:, bus] == 1)[0]


def _print_sign_convention_check(ac_flow_table, topo):
    """지시서 작업B-4: 비를 내기 전에 부호 규약을 코드로 확인한다(단정하지 않고 실측
    데이터로 검증).

    관찰(코드 인용):
    - lower_lp.py `_build_topology`(children[i].append((j,idx)) for i=from_bus,j=to_bus)는
      from_bus를 항상 상류(슬랙에 가까운 쪽)로 간주해 트리를 구성한다 - 이 스크립트의
      `_branch_from_bus_array`/`_branch_to_bus_array`도 동일한 `_branch_line_idxs` 정렬을
      공유하므로 같은 규약이 유지된다.
    - pandapower의 `res_line.q_from_mvar`는 "from_bus에서 그 선로로 흘러 들어가는" 무효
      전력이다(probe_q_marginal.py 모듈 docstring에서 이미 실측 확인된 규약 - 이 스크립트가
      새로 검증하는 것이 아니라 그 결론을 재사용한다).
    - `Qe_base = D@(load_q/S_BASE)`는 "그 branch 하류 전체 부하의 무효전력 합"이며 부하는
      소비(양수)이므로 Qe_base는 항상 >=0이다(하류에 ESS가 없는 이 기저 상태에서는).
    해석/가설: 위 둘 다 "상류->하류 방향으로 흐르는, 부하가 끌어가는 무효전력"을 가리키므로
    부호를 뒤집지 않고 그대로 비교해야 한다 - 그러나 이것이 "가정"인지 "실측 확인"인지는
    아래에서 직접 센 부호 불일치 건수로 판별한다(관찰과 추론을 구분 - 지시서 요구)."""
    section('작업B-4: 부호 규약 확인 (q_from_mvar vs Qe_base, 뒤집지 않고 그대로 비교)')
    total_checked, total_mismatch = 0, 0
    for s in PM.AVG_DAYS:
        Qe_base = _compute_qe_base_mvar(topo, PM.LOAD[s])   # (n_branch,T), 항상 >=0(부하 소비)
        q_from = ac_flow_table[s]['q_from']
        mask = np.abs(Qe_base) >= 0.001   # 거의 0인 분기(말단 등)는 부호 자체가 잡음일 수 있어 제외
        mismatch = np.sum((q_from[mask] > 0) != (Qe_base[mask] > 0))
        total_checked += int(np.sum(mask))
        total_mismatch += int(mismatch)
    print(f"  Qe_base>=0.001Mvar인 (branch,scenario,t) {total_checked}개 중 "
          f"sign(q_from_mvar) != sign(Qe_base) 불일치 = {total_mismatch}건", flush=True)
    if total_mismatch == 0:
        print("  -> 관찰: 불일치 0건. 위 해석(부호를 뒤집지 않는다)이 이 데이터에서 실측으로도 "
              "확인됨.", flush=True)
    else:
        print(f"  -> ⚠ 관찰: 불일치 {total_mismatch}건 존재. 아래 B-3 비율(ratio_from/"
              "ratio_to)을 해석하기 전에 이 불일치가 어느 branch/scenario/t에서 나는지 "
              "먼저 좁혀볼 것(이 함수는 집계만 하고 원인을 판정하지 않는다).", flush=True)


def _print_qe_base_ac_comparison(ac_flow_table, topo):
    """지시서 작업B-3: |q_from|/Qe_base, |q_to|/Qe_base의 분포(전체 + 통제점 경로만).
    Qe_base<0.001Mvar인 branch는 비가 발산하므로 제외하고 제외 건수를 보고한다.
    판정 없음 - 수치만."""
    section('작업B-3: |AC q_from,to| / Qe_base 비율 분포 (판정 없음 - 수치만)')
    D = topo['D']
    for s in PM.AVG_DAYS:
        Qe_base = _compute_qe_base_mvar(topo, PM.LOAD[s])
        q_from = ac_flow_table[s]['q_from']
        q_to = ac_flow_table[s]['q_to']

        denom_ok = np.abs(Qe_base) >= 0.001
        n_excluded = int(np.sum(~denom_ok))
        ratio_from = np.abs(q_from[denom_ok]) / np.abs(Qe_base[denom_ok])
        ratio_to = np.abs(q_to[denom_ok]) / np.abs(Qe_base[denom_ok])
        pf = np.percentile(ratio_from, [0, 25, 50, 75, 100])
        pt = np.percentile(ratio_to, [0, 25, 50, 75, 100])
        print(f"\n  [{s}] 전체 {Qe_base.size}개 (branch,t) 중 분모<0.001Mvar 제외 "
              f"{n_excluded}개", flush=True)
        print(f"    ratio_from(|q_from|/Qe_base): min={pf[0]:.4f} p25={pf[1]:.4f} "
              f"median={pf[2]:.4f} p75={pf[3]:.4f} max={pf[4]:.4f}", flush=True)
        print(f"    ratio_to  (|q_to|/Qe_base)  : min={pt[0]:.4f} p25={pt[1]:.4f} "
              f"median={pt[2]:.4f} p75={pt[3]:.4f} max={pt[4]:.4f}", flush=True)

        for point in POINTS:
            bus = int(point['b'])
            path_e = _path_branches_for_bus(D, bus)
            if len(path_e) == 0:
                print(f"    {point['point_id']}(bus={bus}): 경로 선로 없음", flush=True)
                continue
            sub_ok = denom_ok[path_e, :]
            sub_from = np.abs(q_from[path_e, :])[sub_ok] / np.abs(Qe_base[path_e, :])[sub_ok]
            sub_to = np.abs(q_to[path_e, :])[sub_ok] / np.abs(Qe_base[path_e, :])[sub_ok]
            if sub_from.size == 0:
                print(f"    {point['point_id']}(bus={bus}, 경로선로 {len(path_e)}개): "
                      "유효 표본 없음(전부 분모<0.001Mvar)", flush=True)
                continue
            pfb = np.percentile(sub_from, [0, 25, 50, 75, 100])
            ptb = np.percentile(sub_to, [0, 25, 50, 75, 100])
            print(f"    {point['point_id']}(bus={bus}, 경로선로 {len(path_e)}개, "
                  f"유효표본 {sub_from.size}개): ratio_from min={pfb[0]:.4f} "
                  f"p25={pfb[1]:.4f} median={pfb[2]:.4f} p75={pfb[3]:.4f} max={pfb[4]:.4f} / "
                  f"ratio_to min={ptb[0]:.4f} p25={ptb[1]:.4f} median={ptb[2]:.4f} "
                  f"p75={ptb[3]:.4f} max={ptb[4]:.4f}", flush=True)


# ============================================================
# 1-C) 계측 추가 라운드 - 작업 C: 손실 공식·전압 규약의 항등성 검증
# ============================================================

def _print_loss_formula_identity_check(ac_flow_table, v_bus_table, net):
    """지시서 작업C: loss=r*(P^2+Q^2)/V^2이 실제 AC 조류계산 결과에서 재현되는지 확인한다.

    단위계(지시서 요구 - 명시): 물리단위(Ohm/MW/MVAr/kV)를 쓴다.
      r_ohm = r_pu * PM.Z_BASE_OHM   (lower_lp.py가 r_pu = r_ohm_per_km*length_km/Z_BASE_OHM
                                       으로 만든 것의 역연산)
      V_kv  = vm_pu * PM.VN_KV       (실제 선간전압, kV)
      loss_calc = r_ohm * (P_mw^2 + Q_mvar^2) / V_kv^2   [MW]
    이 (Ohm,MW,MVAr,kV) 조합에서 위 식은 3상 등가회로의 표준 관계식이며 별도 배수가
    필요 없다(pandapower 자신이 res_line을 이 관례로 계산한다) - pu로도 검산할 수 있었으나
    lower_lp.py의 pu 변환(r_pu 등)이 맞는지까지 함께 확인하는 효과가 있어 물리단위를 택했다.
    판정 없음 - 상대오차 분포만 낸다."""
    section('작업C: 손실공식(loss=r*(P^2+Q^2)/V^2) 항등성 검증 (판정 없음 - 오차 분포만)')
    topo = lower_lp._get_topology()
    r_pu = topo['r_pu']
    r_ohm = r_pu * PM.Z_BASE_OHM
    from_bus_arr = _branch_from_bus_array(net)
    to_bus_arr = _branch_to_bus_array(net)

    for s in PM.AVG_DAYS:
        p_from, q_from = ac_flow_table[s]['p_from'], ac_flow_table[s]['q_from']
        p_to, q_to = ac_flow_table[s]['p_to'], ac_flow_table[s]['q_to']
        pl = ac_flow_table[s]['pl']
        vm = v_bus_table[s]   # (n_bus,T), raw vm_pu

        v_from_kv = vm[from_bus_arr, :] * PM.VN_KV
        v_to_kv = vm[to_bus_arr, :] * PM.VN_KV

        loss_calc_from = r_ohm[:, None] * (p_from ** 2 + q_from ** 2) / (v_from_kv ** 2)
        loss_calc_to = r_ohm[:, None] * (p_to ** 2 + q_to ** 2) / (v_to_kv ** 2)

        mask = np.abs(pl) > 1e-12   # 0으로 나누기 방지 - pl~=0인 (branch,t) 제외
        n_excl = int(np.sum(~mask))
        rel_err_from = np.abs(loss_calc_from[mask] - pl[mask]) / np.abs(pl[mask])
        rel_err_to = np.abs(loss_calc_to[mask] - pl[mask]) / np.abs(pl[mask])
        pf = np.percentile(rel_err_from, [0, 50, 100])
        pt = np.percentile(rel_err_to, [0, 50, 100])
        print(f"  [{s}] pl_mw~=0(<=1e-12) 제외 {n_excl}개  "
              f"from단 상대오차: min={pf[0]:.3e} median={pf[1]:.3e} max={pf[2]:.3e}  "
              f"to단 상대오차: min={pt[0]:.3e} median={pt[1]:.3e} max={pt[2]:.3e}", flush=True)


# ============================================================
# 1-D) 계측 추가 라운드 - 작업 A-2/A-3/A-4: AC 되먹임 성분 분해
# ============================================================

def _qp_predicted_A(bus, s, q_level, v_sq_line_table):
    """QP(보정후)가 예측하는 손실저감량 A' - _diagnose_q_prediction_gap의 qp_corr_t와
    동일한 공식을 **독립적으로 재구현**한다(공유 함수로 리팩터링하지 않는다 - 지시서
    작업A-3의 "A'/dLoss_total이 기존 비율분포와 일치해야 한다"는 두 개의 독립 구현이
    일치하는지 보는 교차검증이므로, 하나로 합치면 그 검증이 자기 자신과 비교하는
    동어반복이 된다). D@(bus_onehot.T@q) 구조상 dQe는 그 bus의 슬랙 경로(path) 밖의
    branch에서 자동으로 0이 되므로, 전 branch에 대한 합이 곧 지시서가 요구한
    "sum_{e in path}"와 수학적으로 동치다(별도 경로 필터가 필요 없다)."""
    topo = lower_lp._get_topology()
    D, r_pu, n_bus = topo['D'], topo['r_pu'], topo['n_bus']
    bus_onehot_np = np.zeros((1, n_bus))
    bus_onehot_np[0, bus] = 1.0
    _base_load_p_bus, base_load_q_bus = lower_lp.base_load_bus_arrays()
    profile = np.asarray(PM.LOAD[s], dtype=float)
    load_q_val = base_load_q_bus[:, None] * profile[None, :]
    Qe_base = D @ (load_q_val / PM.S_BASE_MVA)
    q_fixed = np.full((1, PM.TIME_STEPS), q_level)
    dQe = D @ ((bus_onehot_np.T @ q_fixed) / PM.S_BASE_MVA)
    per_branch_uncorr = r_pu[:, None] * (2.0 * Qe_base * dQe - dQe ** 2) * PM.S_BASE_MVA
    v_sq = v_sq_line_table[s]
    return (per_branch_uncorr / v_sq).sum(axis=0)   # (T,)


def _decompose_ac_feedback(ac_full_table, loss_table, from_bus_arr, bus, s, q_level):
    """작업A-2: loss=r*(P^2+Q^2)/V^2 항등식(작업C에서 이미 확인됨 - 같은 r_ohm/V-in-kV
    단위계를 그대로 쓴다)으로 q=0->q_level 손실 변화를 A(Q항)+B(P항)+C(V항)으로
    정확히 쪼갠다.

    ★ 첨자 0/q가 같은 (scenario,t)에서 온 데이터임을 보장하는 방법(지시서 보고 요구 3):
    ac_full_table[bus][s]의 배열은 shape (T, n_q, n_branch)다 - t축은 그대로 유지한 채
    idx0/idxq로 **n_q축만** 인덱싱하므로(`tbl['p_from'][:, idx0, :]` vs
    `[:, idxq, :]`), 두 슬라이스는 항상 동일한 T개 행(=동일 시각들)을 가리킨다. 이는
    "시각을 맞춰 찾는" 별도 매칭 로직이 아니라 배열 축 구조 자체가 강제하는 것이다 -
    t번째 행끼리 어긋날 방법이 없다(같은 배열의 같은 축이므로).

    반환: A,B,C,dLoss_total 각 (T,) - dLoss_total은 loss_table(실측 총손실)에서 직접
    구한다(=검산 대상 - A+B+C가 이것과 같아야 한다)."""
    idx0 = Q_BOUNDARY_POINTS.index(0.0)
    idxq = Q_BOUNDARY_POINTS.index(q_level)
    tbl = ac_full_table[bus][s]
    topo = lower_lp._get_topology()
    r_ohm = topo['r_pu'] * PM.Z_BASE_OHM   # (n_branch,) - 작업C와 동일 역연산

    A, B, C = _decompose_flow_pair(
        tbl['p_from'][:, idx0, :], tbl['q_from'][:, idx0, :],
        tbl['vm_bus'][:, idx0, :], tbl['p_from'][:, idxq, :],
        tbl['q_from'][:, idxq, :], tbl['vm_bus'][:, idxq, :],
        from_bus_arr, r_ohm,
    )
    dLoss_total = loss_table[bus][s][:, idx0] - loss_table[bus][s][:, idxq]
    return A, B, C, dLoss_total


def _decompose_flow_pair(P_e0, Q_e0, vm_bus0, P_eq, Q_eq, vm_busq,
                         from_bus_arr, r_ohm):
    """동일 운전점의 q=0/q>0 AC 상태를 A(Q)+B(P)+C(V)로 분해하는 공통 수식.

    첫 축은 표본 축이며 단일 표본을 넘길 때도 shape (1,n_branch)/(1,n_bus)로 맞춘다.
    `_decompose_ac_feedback`과 별도 P 스윕이 이 한 코드 경로를 공유한다.
    """
    P_e0 = np.atleast_2d(np.asarray(P_e0, dtype=float))
    Q_e0 = np.atleast_2d(np.asarray(Q_e0, dtype=float))
    P_eq = np.atleast_2d(np.asarray(P_eq, dtype=float))
    Q_eq = np.atleast_2d(np.asarray(Q_eq, dtype=float))
    vm_bus0 = np.atleast_2d(np.asarray(vm_bus0, dtype=float))
    vm_busq = np.atleast_2d(np.asarray(vm_busq, dtype=float))
    V_e0_kv_sq = (vm_bus0[:, from_bus_arr] * PM.VN_KV) ** 2
    V_eq_kv_sq = (vm_busq[:, from_bus_arr] * PM.VN_KV) ** 2
    A = np.sum(r_ohm[None, :] * (Q_e0 ** 2 - Q_eq ** 2) / V_e0_kv_sq, axis=1)
    B = np.sum(r_ohm[None, :] * (P_e0 ** 2 - P_eq ** 2) / V_e0_kv_sq, axis=1)
    C = np.sum(
        r_ohm[None, :] * (P_eq ** 2 + Q_eq ** 2)
        * (1.0 / V_e0_kv_sq - 1.0 / V_eq_kv_sq),
        axis=1,
    )
    return A, B, C


def _alpha_by_point_and_level(ac_full_table, loss_table, net):
    """작업 2/4가 공유하는 실행 중 실측 alpha=A/dLoss의 통제점별 분포."""
    from_bus_arr = _branch_from_bus_array(net)
    result = {}
    for point in POINTS:
        bus = int(point['b'])
        result[point['point_id']] = {}
        pooled_A, pooled_dloss = [], []
        for q_level in Q_DIAG_LEVELS:
            A_all, dloss_all = [], []
            for s in PM.AVG_DAYS:
                A, _B, _C, dloss = _decompose_ac_feedback(
                    ac_full_table, loss_table, from_bus_arr, bus, s, q_level
                )
                A_all.append(A)
                dloss_all.append(dloss)
            A_arr = np.concatenate(A_all)
            dloss_arr = np.concatenate(dloss_all)
            mask = dloss_arr != 0.0
            alpha = float(np.median(A_arr[mask] / dloss_arr[mask])) if np.any(mask) else np.nan
            result[point['point_id']][q_level] = alpha
            pooled_A.append(A_arr)
            pooled_dloss.append(dloss_arr)
        A_pool = np.concatenate(pooled_A)
        dloss_pool = np.concatenate(pooled_dloss)
        pool_mask = dloss_pool != 0.0
        result[point['point_id']]['pooled'] = (
            float(np.median(A_pool[pool_mask] / dloss_pool[pool_mask]))
            if np.any(pool_mask) else np.nan
        )
    return result


ALPHA_CSV_FIELDS = [
    'point_id', 'bus', 'S', 'p_level', 'q_level', 'feasible', 'n_samples',
    'alpha_median', 'dloss_ratio_to_p0_median', 'identity_abs_max_mw',
]
ALPHA_RAW_CSV_FIELDS = [
    'point_id', 'bus', 'S', 'scenario', 't', 'p_level', 'q_level', 'feasible',
    'A', 'B', 'C', 'dLoss_total', 'alpha', 'dloss_ratio_to_p0',
    'identity_abs_mw',
]


def _measure_alpha_p_sweep(net, base_p, base_q):
    """작업 3: 기존 P_inj=0 loss_table을 건드리지 않는 별도 P×Q AC 진단.

    부호는 lower_lp.py의 `P_net=P_dis-P_ch`와 evaluate.py의
    `net.sgen.at[i,'p_mw']=unit_p-loss_pcs`를 따른다. 따라서 여기서 p>0은 방전 주입,
    p<0은 충전 흡수다. PCS 손실은 이 진단 격자에 별도로 넣지 않는다.
    """
    if len(net.sgen) == 0:
        pp.create_sgen(net, bus=CONTROL_POINT_BUSES[0], p_mw=0.0, q_mvar=0.0,
                       name='probe_lp_loss_proto_alpha')
    sgen_idx = net.sgen.index[0]
    line_idxs = _branch_line_idxs(net)
    from_bus_arr = _branch_from_bus_array(net)
    r_ohm = lower_lp._get_topology()['r_pu'] * PM.Z_BASE_OHM
    q_levels = [0.0] + Q_DIAG_LEVELS
    expected = (
        len(POINTS) * len(PM.AVG_DAYS) * PM.TIME_STEPS * 5 * len(q_levels)
    )
    print(f"alpha P×Q 스윕 예상 격자 조류계산 횟수={expected} "
          f"({len(POINTS)} point x {len(PM.AVG_DAYS)} scenario x "
          f"{PM.TIME_STEPS} h x 5 P x {len(q_levels)} Q)", flush=True)
    print("P 부호 규약: lower_lp.py P_net=P_dis-P_ch, evaluate.py sgen.p_mw=unit_p-loss_pcs; "
          "p>0 방전 주입, p<0 충전 흡수", flush=True)

    buckets = {}
    for point in POINTS:
        pid, bus, S = point['point_id'], int(point['b']), float(point['S'])
        p_levels = [-S, -S / 2.0, 0.0, S / 2.0, S]
        for p_level in p_levels:
            for q_level in q_levels:
                buckets[(pid, p_level, q_level)] = dict(
                    alpha=[], identity=[], dloss_ratio=[]
                )
    raw_rows = []
    for point in POINTS:
        pid, bus, S = point['point_id'], int(point['b']), float(point['S'])
        p_levels = [-S, -S / 2.0, 0.0, S / 2.0, S]
        for s in PM.AVG_DAYS:
            profile = PM.LOAD[s]
            for t in range(PM.TIME_STEPS):
                net.load['p_mw'] = base_p * profile[t]
                net.load['q_mvar'] = base_q * profile[t]
                for p_level in p_levels:
                    base_state = None
                    for q_level in q_levels:
                        net.sgen.at[sgen_idx, 'bus'] = bus
                        net.sgen.at[sgen_idx, 'p_mw'] = p_level
                        net.sgen.at[sgen_idx, 'q_mvar'] = q_level
                        ok = evaluate._run_pf_with_retry(net)
                        if not ok:
                            raise RuntimeError(
                                f"alpha P×Q 조류계산 발산: point={pid} scenario={s} t={t} "
                                f"p={p_level} q={q_level}"
                            )
                        state = dict(
                            p_from=net.res_line.loc[line_idxs, 'p_from_mw'].to_numpy(),
                            q_from=net.res_line.loc[line_idxs, 'q_from_mvar'].to_numpy(),
                            vm_bus=net.res_bus.vm_pu.to_numpy(),
                            loss=float(net.res_line.pl_mw.sum()),
                        )
                        bucket = buckets[(pid, p_level, q_level)]
                        if q_level == 0.0:
                            base_state = state
                            bucket['identity'].append(0.0)
                            raw_rows.append(dict(
                                point_id=pid, bus=bus, S=S, scenario=s, t=t,
                                p_level=p_level, q_level=q_level,
                                feasible=bool(p_level ** 2 + q_level ** 2 <= S ** 2),
                                A=0.0, B=0.0, C=0.0, dLoss_total=0.0,
                                alpha=np.nan, dloss_ratio_to_p0=np.nan,
                                identity_abs_mw=0.0,
                            ))
                            continue
                        A, B, C = _decompose_flow_pair(
                            base_state['p_from'], base_state['q_from'], base_state['vm_bus'],
                            state['p_from'], state['q_from'], state['vm_bus'],
                            from_bus_arr, r_ohm,
                        )
                        dloss = base_state['loss'] - state['loss']
                        identity = abs(float(A[0] + B[0] + C[0]) - dloss)
                        bucket['identity'].append(identity)
                        alpha_value = np.nan
                        if dloss != 0.0:
                            alpha_value = float(A[0]) / dloss
                            bucket['alpha'].append(alpha_value)
                        raw_rows.append(dict(
                            point_id=pid, bus=bus, S=S, scenario=s, t=t,
                            p_level=p_level, q_level=q_level,
                            feasible=bool(p_level ** 2 + q_level ** 2 <= S ** 2),
                            A=float(A[0]), B=float(B[0]), C=float(C[0]),
                            dLoss_total=dloss, alpha=alpha_value,
                            dloss_ratio_to_p0=np.nan, identity_abs_mw=identity,
                        ))

    p0_dloss = {
        (row['point_id'], row['scenario'], row['t'], row['q_level']): row['dLoss_total']
        for row in raw_rows if row['p_level'] == 0.0
    }
    for row in raw_rows:
        denom = p0_dloss[(row['point_id'], row['scenario'], row['t'], row['q_level'])]
        if denom != 0.0:
            ratio = row['dLoss_total'] / denom
            row['dloss_ratio_to_p0'] = ratio
            buckets[(row['point_id'], row['p_level'], row['q_level'])][
                'dloss_ratio'
            ].append(ratio)

    rows = []
    section('작업 3: alpha의 P×Q 정의역 스윕 (A/dLoss median, 판정 없음)')
    for point in POINTS:
        pid, bus, S = point['point_id'], int(point['b']), float(point['S'])
        p_levels = [-S, -S / 2.0, 0.0, S / 2.0, S]
        print(f"\n  {pid} (bus={bus}, S={S}):", flush=True)
        print("    p_level     q_level   feasible  alpha_median  "
              "dLoss(p,q)/dLoss(0,q) median  identity_max_MW", flush=True)
        for p_level in p_levels:
            for q_level in q_levels:
                bucket = buckets[(pid, p_level, q_level)]
                alpha_median = (
                    float(np.median(bucket['alpha'])) if bucket['alpha'] else np.nan
                )
                identity_max = (
                    float(np.max(bucket['identity'])) if bucket['identity'] else np.nan
                )
                dloss_ratio_median = (
                    float(np.median(bucket['dloss_ratio']))
                    if bucket['dloss_ratio'] else np.nan
                )
                feasible = bool(p_level ** 2 + q_level ** 2 <= S ** 2)
                print(f"    {p_level:+.6f}  {q_level:.6f}  {str(feasible):>8}  "
                      f"{alpha_median:12.6f}  {dloss_ratio_median:14.6f}  "
                      f"{identity_max:.3e}", flush=True)
                rows.append(dict(
                    point_id=pid, bus=bus, S=S, p_level=p_level, q_level=q_level,
                    feasible=feasible, n_samples=len(bucket['identity']),
                    alpha_median=alpha_median,
                    dloss_ratio_to_p0_median=dloss_ratio_median,
                    identity_abs_max_mw=identity_max,
                ))
    return rows, raw_rows, expected


def _write_alpha_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ALPHA_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"alpha P×Q CSV 저장: {path}", flush=True)


def _write_alpha_raw_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ALPHA_RAW_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"alpha P×Q 원값 CSV 저장: {path}", flush=True)


def _ratio_percentiles(num_arr, denom_arr):
    mask = denom_arr != 0.0
    n_excl = int(np.sum(~mask))
    if not np.any(mask):
        return None, n_excl
    return np.percentile(num_arr[mask] / denom_arr[mask], [0, 25, 50, 75, 100]), n_excl


def _print_ac_feedback_decomposition(ac_full_table, loss_table, v_sq_line_table, net):
    """작업A-2/A-3: 판정 없음 - 항등성 잔차(구현 검증, 지시서상 예외로 허용됨)와 비율
    분포만 낸다."""
    section('작업A-2/A-3: AC 되먹임 성분 분해 (A=Q항,B=P항,C=V항, A\'=QP예측) - '
            '판정 없음, 항등성만 검산')
    from_bus_arr = _branch_from_bus_array(net)
    for point in POINTS:
        bus = int(point['b'])
        if bus not in ac_full_table:
            continue
        print(f"\n  {point['point_id']} (bus={bus}):", flush=True)
        for q_level in Q_DIAG_LEVELS:
            A_list, B_list, C_list, dtot_list, Aqp_list = [], [], [], [], []
            for s in PM.AVG_DAYS:
                A, B, C, dtot = _decompose_ac_feedback(ac_full_table, loss_table, from_bus_arr,
                                                        bus, s, q_level)
                Aqp = _qp_predicted_A(bus, s, q_level, v_sq_line_table)
                A_list.append(A); B_list.append(B); C_list.append(C)
                dtot_list.append(dtot); Aqp_list.append(Aqp)
            A_arr = np.concatenate(A_list)
            B_arr = np.concatenate(B_list)
            C_arr = np.concatenate(C_list)
            dtot_arr = np.concatenate(dtot_list)
            Aqp_arr = np.concatenate(Aqp_list)

            identity_resid = np.abs(A_arr + B_arr + C_arr - dtot_arr)
            print(f"    Q={q_level}: 항등성 |A+B+C-dLoss_total| 분포(MW, 구현 검증 - "
                  f"기계정밀도 수준이 아니면 구현을 의심할 것): "
                  f"min={identity_resid.min():.3e} median={np.median(identity_resid):.3e} "
                  f"max={identity_resid.max():.3e}", flush=True)

            for label, num_arr, denom_arr in (
                ('A/dLoss_total', A_arr, dtot_arr),
                ('B/dLoss_total', B_arr, dtot_arr),
                ('C/dLoss_total', C_arr, dtot_arr),
                ("A'/A", Aqp_arr, A_arr),
                ("A'/dLoss_total", Aqp_arr, dtot_arr),
            ):
                pct, n_excl = _ratio_percentiles(num_arr, denom_arr)
                if pct is None:
                    print(f"      {label}: 전 표본 분모=0 - 비 정의 불가", flush=True)
                else:
                    print(f"      {label}(분모=0 {n_excl}건 제외): min={pct[0]:.4f} "
                          f"p25={pct[1]:.4f} median={pct[2]:.4f} p75={pct[3]:.4f} "
                          f"max={pct[4]:.4f}", flush=True)


def _print_dqe_ratio(ac_full_table):
    """작업A-4: 경로 선로에서 (Q_e0-Q_eq)/q 분포. 판정 없음 - 수치만."""
    section("작업A-4: 실제 무효조류 감소량(dQe)/주입량(q) 비 (경로 선로만, 판정 없음)")
    topo = lower_lp._get_topology()
    D = topo['D']
    for point in POINTS:
        bus = int(point['b'])
        if bus not in ac_full_table:
            continue
        path_e = _path_branches_for_bus(D, bus)
        print(f"\n  {point['point_id']} (bus={bus}, 경로선로 {len(path_e)}개):", flush=True)
        if len(path_e) == 0:
            print("    경로 선로 없음", flush=True)
            continue
        idx0 = Q_BOUNDARY_POINTS.index(0.0)
        for q_level in Q_DIAG_LEVELS:
            idxq = Q_BOUNDARY_POINTS.index(q_level)
            vals = []
            for s in PM.AVG_DAYS:
                tbl = ac_full_table[bus][s]
                Q_e0 = tbl['q_from'][:, idx0, :][:, path_e]
                Q_eq = tbl['q_from'][:, idxq, :][:, path_e]
                vals.append(((Q_e0 - Q_eq) / q_level).ravel())
            arr = np.concatenate(vals)
            pct = np.percentile(arr, [0, 25, 50, 75, 100])
            print(f"    Q={q_level}: dQe/q 분포: min={pct[0]:.4f} p25={pct[1]:.4f} "
                  f"median={pct[2]:.4f} p75={pct[3]:.4f} max={pct[4]:.4f}", flush=True)


# ============================================================
# 2) 프로토타입 LP 빌더 (avg 전용 - kind='peak'은 3차 개정에서 완전히 제거)
# ============================================================

def _build_problem_proto(method, n, T, M=None, bus_idx=None):
    """lower_lp._build_problem('avg',...)의 구조(SOC/PCS개별한계/LinDistFlow/전압유도항)를
    그대로 복사하되, 다각형 상한을 변수 s_app으로 승격하고 손실편익(method)·PCS손실비용
    (공통)을 목적함수에 추가한다. force_q_zero 경로는 다루지 않는다(이 실험은 Q 자유
    최적화만 대상 - force_q_zero=True 대조군은 probe_q_selective.py가 이미 만들었다).
    lower_lp.py 원본은 이 함수가 건드리지 않는다(읽기 전용 재사용: _get_topology()).

    bus_idx: None이면 bus_onehot을 기존처럼 cp.Parameter로 둔다(포인트마다 값만 바꿔
    재사용 - method='pwl'/'none'/'pcs_only'용). 정수면 bus_onehot을 그 버스로 구운
    numpy 상수로 굽는다(파라미터가 아니므로 이 Problem은 그 버스 전용이 되고 포인트마다
    새로 지어야 한다 - method='qp' 전용, 모듈 docstring "QP 전개" 절 참조: dQ_e/dP_e를
    제곱하려면 그 계수(bus_onehot)가 파라미터가 아니라 상수여야 DPP가 유지된다).

    ★ 계측 추가 라운드 작업C-2 신규 method:
      'none'     - 기준1(force_q_zero=True 등가). Q 자체가 없다(cp.Constant(0)) - 다각형/
                   s_app/q_penalty를 아예 만들지 않는다(lower_lp.py 원본의 force_q_zero=True
                   경로가 "다각형은 걸지 않는다"는 것과 동일한 관례 - 그래야
                   P_ch<=S,P_dis<=S 개별한계만 남아 진짜 force_q_zero=True와 동등해진다.
                   s_app<=S*cos(pi/N)까지 남기면 원 제약이 3.53% 더 좁아져 "등가"가 깨진다).
      'pcs_only' - 기준2. Q는 pwl/qp와 똑같이 자유(다각형/s_app/q_penalty/SOC/사이클
                   등식 전부 그대로) - **loss_term만 0**으로 둬 손실 편익/비용 항의
                   존재 자체가 만드는 비용을 손실 항의 크기와 분리해서 잰다(지시서 C-2/
                   보고요청5 - "나머지는 그대로 두었는지 확인").
    """
    assert method in ('pwl', 'qp', 'none', 'pcs_only'), f'알 수 없는 method: {method}'
    assert (method == 'qp') == (bus_idx is not None), (
        f"method={method}, bus_idx={bus_idx}: QP만 bus_idx 필수(build-time 상수 소성), "
        "나머지(pwl/none/pcs_only)는 bus_idx를 주지 않는다(Parameter로 유지해 재사용)."
    )

    dt = PM.DT_HOURS
    topo = lower_lp._get_topology()
    D, r_pu, x_pu = topo['D'], topo['r_pu'], topo['x_pu']
    n_bus, n_branch = topo['n_bus'], topo['n_branch']
    R_MAT = np.tile(r_pu[:, None], (1, T))
    X_MAT = np.tile(x_pu[:, None], (1, T))

    P_ch = cp.Variable((n, T), nonneg=True)
    P_dis = cp.Variable((n, T), nonneg=True)
    soc = cp.Variable((n, T + 1))

    # ★ 기준1('none')만 다각형/s_app/q_penalty를 아예 안 만든다 - 위 docstring 참조.
    has_pcs_polygon = (method != 'none')
    s_app = cp.Variable((n, T), nonneg=True) if has_pcs_polygon else None
    q_penalty = cp.Variable((n, T), nonneg=True) if has_pcs_polygon else None

    lhs_params = None
    if method == 'pwl':
        Q_seg = cp.Variable((n, T, M), nonneg=True)
        Q = cp.sum(Q_seg, axis=2)
        # ★ nonneg=True를 주지 않는다 - 이 Parameter의 값은 "실측 시컨트 기울기 * SMP"다
        # (DPP 유지를 위한 사전곱 - _set_params 참조). 실측 기울기는 항상 양수라는 보장이
        # 없다(무효조류 총량을 넘는 영역에서는 Q_flow 부호가 뒤집혀 손실이 오히려 늘 수
        # 있다). nonneg=True로 선언했다가 실측 데이터의 음수 기울기로 "Parameter value must
        # be nonnegative" 에러가 실제로 났었다 - 물리적으로 정상이므로 제약을 없앤다.
        lhs_params = [cp.Parameter((n, T)) for _ in range(M)]
    elif method in ('qp', 'pcs_only'):
        Q_seg = None
        Q = cp.Variable((n, T))
    else:   # method == 'none'
        Q_seg = None
        # ★ 버그 수정(2차분 진단): 여기를 cp.Constant(np.zeros(...))로 두면 netinj_q =
        # (load_q_bus - bus_onehot.T@Q)/S_BASE_MVA에 변수가 전혀 안 남는다(Q도 상수,
        # load_q_bus/bus_onehot은 Parameter라 전부 "Parameter뿐인 affine식"이 됨). 이
        # 변수 0개 상태로 스칼라 나눗셈(div)을 cvxpy COO 백엔드가 canonicalize하면
        # coo_mul_elem의 rhs가 (33,24) dense_const로 브로드캐스트돼 "일반 케이스"(스칼라
        # 브로드캐스트 분기를 못 타는 경로)로 빠지고, 그 안에서 `mask = rhs_vals != 0`가
        # scipy sparse 객체의 __bool__을 건드려 "The truth value of an array with more
        # than one element is ambiguous" 예외를 던진다(전 솔버 공통이라 4회 재시도가
        # 전부 실패 -> status=None). lower_lp.py 원본의 force_q_zero=True 관례를 그대로
        # 따라 Q를 Variable로 유지하고 등식제약 Q==0으로 강제한다 - 그러면 netinj_q에
        # Variable이 남아 이 버그 경로를 타지 않는다(값·최적해는 상수 대입과 수학적으로
        # 동일 - Q==0이 유일해이므로).
        Q = cp.Variable((n, T))

    if bus_idx is None:
        bus_onehot = cp.Parameter((n, n_bus))
        bus_onehot_is_param = True
    else:
        bus_onehot = np.zeros((n, n_bus))
        bus_onehot[0, bus_idx] = 1.0
        bus_onehot_is_param = False

    S_param = cp.Parameter(n, nonneg=True)
    E_param = cp.Parameter(n, nonneg=True)
    load_p_bus = cp.Parameter((n_bus, T))           # 시나리오별 버스 유효부하 (MW, 소비=양수)
    load_q_bus = cp.Parameter((n_bus, T))
    # ★ nonneg=True 필수(lower_lp.py 원본과의 차이) - 원본 solve_avg의 smp_param은 부호를
    # 몰라도 됐다(곱해지는 대상이 P_ch-P_dis, 즉 affine이라 부호 무관하게 여전히 affine이라
    # DCP 자동 성립). 이 프로토타입의 QP 손실항(method='qp')은 smp_row를 **convex**
    # 표현식에 곱하므로 부호를 알아야 DCP가 성립한다 - 안 붙였다가 실제로
    # "Problem does not follow DCP rules"로 걸렸다. SMP는 물리적으로 항상 양수다.
    smp_param = cp.Parameter(T, nonneg=True)

    S_col = cp.reshape(S_param, (n, 1), order='C')
    E_col = cp.reshape(E_param, (n, 1), order='C')
    smp_row = cp.reshape(smp_param, (1, T), order='C')

    constraints = [P_ch <= S_col, P_dis <= S_col]
    constraints += [
        soc[:, 0] == PM.SOC_INIT_FRAC * E_param,
        soc[:, T] == PM.SOC_INIT_FRAC * E_param,
        soc >= PM.SOC_MIN_FRAC * E_col,
        soc <= PM.SOC_MAX_FRAC * E_col,
    ]
    for t in range(T):
        constraints.append(
            soc[:, t + 1] == soc[:, t] * (1 - PM.SELF_DISCHARGE_HOURLY)
            + PM.ETA_C * P_ch[:, t] * dt - P_dis[:, t] / PM.ETA_D * dt
        )

    P_net = P_dis - P_ch

    if has_pcs_polygon:
        # ---- 다각형: 고정 s_cap 대신 변수 s_app (모듈 docstring "PCS 손실 항" 참조) ----
        for k in range(PM.POLY_N):
            theta = 2.0 * np.pi * k / PM.POLY_N
            constraints.append(P_net * float(np.cos(theta)) + Q * float(np.sin(theta)) <= s_app)
        # ★ 4차 개정 버그 수정: s_app<=S_col이면 다각형이 반지름 S 원을 **바깥에서 감싸**
        # (circumscribed) 1/cos(pi/N)=3.53%(N=12) 초과 운전점을 허용한다(실측: P1에서
        # (P,Q)=(0.176,0.0472), 피상전력 0.1822=1.0353*S). 원본 lower_lp.py는 s_cap을
        # S*cos(pi/N)로 둬 다각형을 원에 **내접**시킨다(꼭짓점이 원 위, 절대 원 밖으로
        # 못 나감) - 그 관례를 그대로 따른다.
        s_cap = S_col * float(np.cos(np.pi / PM.POLY_N))
        constraints.append(s_app <= s_cap)
        constraints.append(q_penalty >= s_app - P_ch - P_dis)
    else:   # method == 'none'
        # 다각형/s_app/q_penalty는 안 건다(위 docstring) - 대신 Q를 Variable인 채로
        # 0에 등식제약한다(위 Q 선언부 주석의 cvxpy div 버그 회피와 동일 이유).
        # P_ch<=S,P_dis<=S 개별한계만 남아 lower_lp.py 원본의 force_q_zero=True와
        # 동등해진다(위 docstring 참조).
        constraints.append(Q == 0)

    if method == 'pwl':
        boundaries = SEGMENT_BOUNDARIES[M]
        for m in range(M):
            delta_m = float(boundaries[m + 1] - boundaries[m])
            constraints.append(Q_seg[:, :, m] <= delta_m)

    # ---- LinDistFlow: 버스별 순부하 -> 선로조류 -> V^2 (부록C.4, Baran-Wu load-positive) ----
    # bus_onehot이 Parameter든 상수든 이 식은 그대로 성립한다(cvxpy는 둘을 똑같이 다룬다) -
    # volt_penalty는 cp.pos()(조각별 선형)만 쓰므로 파라미터 등급 문제가 없다. 제곱이
    # 필요한 QP 손실항은 이 P_e/Q_e를 재사용하지 않고 아래에서 별도로 dP_e/dQ_e를 만든다
    # (이 P_e/Q_e에는 load_p_bus/load_q_bus라는 별도 Parameter가 덧셈으로 섞여 있어,
    # 그대로 제곱하면 그 파라미터와 smp_row가 곱해지는 문제가 다시 생긴다 - 모듈 docstring
    # "QP 전개" 절 참조).
    netinj_p = (load_p_bus - bus_onehot.T @ P_net) / PM.S_BASE_MVA   # (n_bus,T), pu
    netinj_q = (load_q_bus - bus_onehot.T @ Q) / PM.S_BASE_MVA
    P_e = D @ netinj_p                               # (n_branch,T)
    Q_e = D @ netinj_q
    v = PM.V_SLACK_SQ - 2.0 * (D.T @ (cp.multiply(R_MAT, P_e) + cp.multiply(X_MAT, Q_e)))
    v_nonslack = v[1:, :]
    # mu_volt: lower_lp.py와 동일 이유로 Parameter가 아니라 float 상수로 굽는다.
    volt_penalty = float(PM.MU_VOLT) * cp.sum(
        cp.pos(v_nonslack - PM.V_SQ_MAX) + cp.pos(PM.V_SQ_MIN - v_nonslack)
    )

    # method=='none'은 q_penalty 자체가 없으므로 pcs_cost=0(파이썬 float - cvxpy Variable이
    # 아니라도 아래 objective_expr의 '+ pcs_cost'는 상수 덧셈으로 그대로 유효하다).
    pcs_cost = (float(C_PCS) * cp.sum(cp.multiply(smp_row, q_penalty)) * dt
                if has_pcs_polygon else 0.0)

    dpp_terms = {}   # 진단용 - dpp_preserved=False일 때만 채워 stdout에 보고(_diagnose_dpp_terms)

    if method == 'pwl':
        # ★ DPP 함정(1차 작성 중 실제로 걸림): cp.multiply(smp_row, cp.multiply(lhs_params[m],
        # Q_seg))처럼 두 개의 서로 다른 Parameter(smp_row, lhs_params[m])를 곱하면 DPP가
        # 요구하는 "곱셈의 한쪽은 반드시 parameter-free"를 어긴다. 해결: SMP를
        # lhs_params[m]의 **값**에 미리 곱해 둔다(순수 numpy, cvxpy 밖 - _set_params 참조).
        loss_benefit = 0
        for m in range(M):
            loss_benefit = loss_benefit + cp.sum(cp.multiply(lhs_params[m], Q_seg[:, :, m]))
        loss_term = -loss_benefit * dt   # 비용에서 차감(이득)
    elif method in ('none', 'pcs_only'):
        # ★ 계측 추가 라운드 작업C-2: 기준1은 Q 자체가 없으니 손실항도 없고, 기준2는
        # "손실 편익 항만 제거"가 정의이므로 loss_term=0 - 나머지(다각형/s_app/q_penalty/
        # SOC/사이클 등식)는 위에서 이미 pwl/qp와 동일하게 만들어졌다(method=='pcs_only'는
        # has_pcs_polygon=True라 그 블록을 그대로 통과했다 - 보고요청5).
        loss_term = 0.0
    else:
        # ---- QP 전개형 (모듈 docstring "QP 전개" 절 - 3차 개정 신규) ----
        # dP_e/dQ_e: bus_onehot이 이제 상수(bus_idx가 주어졌으므로)라 계수가 파라미터를
        # 전혀 포함하지 않는 "순수 변수" 표현식이다 - 제곱해도 파라미터가 제곱으로 등장하지
        # 않는다(위 공유 P_e/Q_e와 달리 load_p_bus/load_q_bus를 섞지 않은 것도 이 때문 -
        # 섞으면 그 파라미터가 다시 끼어든다).
        # ★ 부호 주의: P_e = P_e_base - dP_e가 성립해야 아래 _set_params의
        # cross_p/cross_q(-2*rsmp*P_e_base 부호)와 정합한다. bus_onehot.T@P_net에
        # 음수를 씌우면(P_e = P_e_base + dP_e가 되어) 교차항 부호가 반대로 뒤집히므로
        # 음수를 넣지 않는다(한 번 이 부호를 잘못 넣었다가 셀프리뷰에서 발견했다).
        dP_e = D @ ((bus_onehot.T @ P_net) / PM.S_BASE_MVA)   # (n_branch,T), 순수 변수
        dQ_e = D @ ((bus_onehot.T @ Q) / PM.S_BASE_MVA)
        rsmp_param = cp.Parameter((n_branch, T), nonneg=True)     # 값 = r_pu*smp*dt (numpy)
        cross_p_param = cp.Parameter((n_branch, T))                # 값 = -2*r_pu*dt*smp*P_e_base
        cross_q_param = cp.Parameter((n_branch, T))                # 값 = -2*r_pu*dt*smp*Q_e_base
        quad_term = cp.sum(cp.multiply(rsmp_param, cp.square(dP_e) + cp.square(dQ_e)))
        cross_term = cp.sum(cp.multiply(cross_p_param, dP_e)) + cp.sum(cp.multiply(cross_q_param, dQ_e))
        # 버린 상수항: r_pu*smp*dt*(P_e_base^2+Q_e_base^2) - 변수와 무관하므로 argmin 불변
        # (지시서 "상수항은 변수와 무관하므로 목적함수에서 제외해도 최적해가 바뀌지 않는다").
        loss_term = quad_term + cross_term

        dpp_terms = dict(
            dP_e=dP_e, dQ_e=dQ_e,
            square_dP_e=cp.square(dP_e), square_dQ_e=cp.square(dQ_e),
            quad_term=quad_term, cross_term=cross_term, loss_term=loss_term,
        )

    objective_expr = (
        cp.sum(cp.multiply(smp_row, P_ch - P_dis)) * dt
        + 1e-6 * cp.sum(P_ch + P_dis)   # lower_lp.py의 EPS_REG와 동일 값(그 상수를
                                          # export하지 않으므로 문헌값 그대로 복제 -
                                          # 값이 갈리면 회귀검증 무의미해지니 원본과
                                          # 반드시 대조할 것).
        + volt_penalty + pcs_cost + loss_term
    )
    problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    dpp_preserved = bool(problem.is_dcp(dpp=True))

    params = dict(S=S_param, E=E_param, load_p_bus=load_p_bus, load_q_bus=load_q_bus,
                  smp=smp_param)
    if bus_onehot_is_param:
        params['bus_onehot'] = bus_onehot
    if method == 'pwl':
        params['lhs'] = lhs_params
    elif method == 'qp':
        params['rsmp'] = rsmp_param
        params['cross_p'] = cross_p_param
        params['cross_q'] = cross_q_param
    # 'none'/'pcs_only': 추가 Parameter 없음

    varset = dict(P_ch=P_ch, P_dis=P_dis, Q=Q, soc=soc, P_net=P_net,
                  s_app=s_app, q_penalty=q_penalty)

    return dict(problem=problem, params=params, vars=varset,
                kind='avg', method=method, M=M, bus_idx=bus_idx,
                dpp_preserved=dpp_preserved, dpp_terms=dpp_terms,
                D=D, r_pu=r_pu, n_branch=n_branch)


def _diagnose_dpp_terms(entry):
    """dpp_preserved=False(QP 전용 - PWL은 dpp_terms가 항상 비어 있음)일 때 하위
    표현식을 하나씩 .is_dpp()로 짚어 원인 항을 stdout에 보고한다(지시서 요구사항:
    "깨지면 어느 항이 원인인지 특정할 것"). 정상(dpp_preserved=True)이면 아무것도
    출력하지 않는다(호출부에서 이미 참임을 확인하고서만 부르므로 이 함수 자체는
    무조건 실행해도 안전하다)."""
    if entry['dpp_preserved'] or not entry['dpp_terms']:
        return
    print('  ★ DPP 원인 진단 (QP):', flush=True)
    for name, expr in entry['dpp_terms'].items():
        try:
            ok = bool(expr.is_dpp())
        except Exception as exc:
            ok = f'확인불가({exc})'
        print(f'    {name}: is_dpp()={ok}', flush=True)


def _set_params(entry, S, E, bus_idx, profile, smp, lhs_row_values=None, v_sq_line=None,
                qe_base_ac_mvar=None, apply_qp_corrections=True):
    """entry(=_build_problem_proto 반환)의 Parameter들에 실제 값을 채운다.
    lower_lp._prepare_common을 그대로 재사용해 load_p_bus/load_q_bus를 만든다
    (읽기 전용 재사용 - lower_lp.py 원본 미수정). bus_onehot이 이미 상수로 구워진
    경우(entry['params']에 'bus_onehot' 키가 없음, method='qp')에는 그 값을 건드리지
    않는다 - bus_idx 인자는 그 경우에도 받되(lower_lp._prepare_common이 항상 요구하므로)
    entry 자체가 그 버스 전용으로 지어졌다는 사실은 호출부(_process)가 보장한다.

    v_sq_line: (n_branch,T) - 이 시나리오의 기저(ESS 없음) 선로전압 제곱, from-bus
    기준(작업 B, _measure_loss_table이 만든 v_sq_line_table[scenario] 참조). method='qp'
    이고 QP_V2_CORRECTION=True일 때만 쓰인다 - None이면(또는 플래그가 False면) V=1 근사
    (보정 없음, 기존 동작)로 되돌아간다."""
    n, S_val, E_val, onehot, load_p_val, load_q_val = lower_lp._prepare_common(
        S, E, bus_idx, profile, PM.SELF_DISCHARGE_HOURLY, PM.SOC_INIT_FRAC
    )
    p = entry['params']
    p['S'].value = S_val
    p['E'].value = E_val
    p['load_p_bus'].value = load_p_val
    p['load_q_bus'].value = load_q_val
    smp_arr = np.asarray(smp, dtype=float)
    p['smp'].value = smp_arr
    if 'bus_onehot' in p:
        p['bus_onehot'].value = onehot

    if entry['method'] == 'pwl':
        # ★ SMP를 여기서 미리 곱한다(DPP 유지 목적 - _build_problem_proto의 loss_benefit
        # 주석 참조). lhs_row_values[m]은 순수 MW/Mvar 기울기(_lhs_rows_for)이고, 이 곱셈
        # 이후 Parameter에 담기는 값은 원/(Mvar*h) 단위다.
        for m, param in enumerate(p['lhs']):
            param.value = lhs_row_values[m] * smp_arr[None, :]   # (1,T)*(T,) -> (1,T)
    elif entry['method'] == 'qp':
        # ---- QP 교차항 계수 (모듈 docstring "QP 전개" 절) - cvxpy 밖 numpy로 계산 ----
        D, r_pu = entry['D'], entry['r_pu']
        Pe_base = D @ (load_p_val / PM.S_BASE_MVA)   # (n_branch,T), 순수 numpy(변수 없음), pu
        Qe_base = D @ (load_q_val / PM.S_BASE_MVA)
        if apply_qp_corrections and QP_QE_BASE_AC:
            if qe_base_ac_mvar is None:
                raise ValueError("QP_QE_BASE_AC=True인데 기저 AC q_from 조류가 전달되지 않았다.")
            # pandapower q_from_mvar는 이 방사형 load-positive 상태에서 양수다. 진단용
            # 부호 검산과 같은 관례로 절댓값을 취해 Baran-Wu Qe_base(부하양수)에 맞춘다.
            Qe_base = np.abs(np.asarray(qe_base_ac_mvar, dtype=float)) / PM.S_BASE_MVA
        # ★ 4차 개정(QP 단위 수정) 버그 수정: dP_e/dQ_e는 S_BASE_MVA로 나눈 pu다
        # (P_net/S_BASE_MVA를 D로 매핑한 것). 따라서 r_pu*(dP_e^2+dQ_e^2)는 "pu 손실"
        # (무차원, S_BASE 기준 비율)이지 MW가 아니다 - SMP(원/MWh)와 곱해 원화를 만들려면
        # 먼저 MW로 환산해야 하는데 그 인수(S_BASE_MVA)가 빠져 있었다. 3차 실행에서
        # Q=0.05 손실저감 예측이 실측 대비 QP만 약 10.8배 작게 나온 원인이 이것이다.
        # cross_p/cross_q는 이 rsmp에서 파생되므로 자동으로 함께 고쳐진다.
        rsmp = r_pu[:, None] * smp_arr[None, :] * PM.DT_HOURS * PM.S_BASE_MVA   # (n_branch,T), pu 기준
        # ★ 작업 B(QP 단위 수정 후 잔차 규명): 이 단위 수정 후에도 남은 ~7~8% 오차는
        # LinDistFlow 손실식 r*(P^2+Q^2)/V^2에서 V=1 pu로 둔 근사 때문이었다(실측 비율
        # 0.9272~0.9317 = 1/V^2, 함의 전압 0.9629~0.9652가 해당 버스의 실제 계통전압과
        # 정합). v_sq_line(기저 조류계산의 from-bus 전압 제곱, _measure_loss_table에서
        # 추가 조류계산 없이 캐시됨 - 모듈 docstring "V^2 보정" 절 참조)으로 나눠 이
        # 근사를 없앤다. QP_V2_CORRECTION=False나 v_sq_line=None이면 기존 동작(V=1)
        # 그대로 유지 - 보정 전/후 나란히 비교(지시서 B-4)를 위해 토글 가능하게 둔다.
        if QP_V2_CORRECTION and v_sq_line is not None:
            rsmp = rsmp / v_sq_line
        if apply_qp_corrections and QP_GROSSUP:
            rsmp = rsmp / float(QP_GROSSUP_ALPHA)
        p['rsmp'].value = rsmp
        p['cross_p'].value = -2.0 * rsmp * Pe_base
        p['cross_q'].value = -2.0 * rsmp * Qe_base
    return n


def _solve_timed(entry):
    """★ 3차 개정: 솔버 우선순위·정확도 진단(지시서 "solver 정확도 문제" 절). 2차 실행에서
    avg+PWL 조합에서만 OPTIMAL_INACCURATE 6/60(10%)이 났다(CLARABEL 선택 59/60). 세그먼트
    폭이 좁을수록(이번 개정으로 더 좁아진 구간이 생겼다) 계수 스케일 차가 커져 내점법
    수렴이 아슬아슬해지는 것으로 추정한다 - max_iter를 기본값(CLARABEL 200)보다 크게 준
    1차 시도를 먼저 하고, 그래도 OPTIMAL이 아니면 허용오차까지 완화한 2차 CLARABEL,
    그다음 OSQP(반복한도 완화), 마지막으로 cvxpy 기본 자동선택 순으로 재시도한다.
    OPTIMAL_INACCURATE도 받아들이되(값을 버리지 않는다) inaccurate=True로 표시해
    상위에서 (kind,method,M,point_id,scenario) 단위로 집계·보고한다 - 자동으로
    판정하지 않는다(사람이 보고 판단). 이 조정이 실제로 10%를 없앴는지는 실행 후
    stdout(솔버 진단 절)으로 확인할 것 - 이 세션은 실행하지 않는다.
    반환: (elapsed, solver_name, inaccurate)."""
    t0 = time.perf_counter()
    for solver_kwargs in (
        dict(solver=cp.CLARABEL, max_iter=2000),
        dict(solver=cp.CLARABEL, max_iter=2000, tol_gap_abs=1e-6, tol_gap_rel=1e-6,
             tol_feas=1e-6),
        dict(solver=cp.OSQP, max_iter=100000, eps_abs=1e-6, eps_rel=1e-6),
        dict(),
    ):
        try:
            entry['problem'].solve(**solver_kwargs)
        except Exception:
            continue
        if entry['problem'].status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            break
    elapsed = time.perf_counter() - t0
    if entry['problem'].status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            f"LP 미해결(status={entry['problem'].status}): kind={entry['kind']} "
            f"method={entry['method']} M={entry['M']} (CLARABEL x2/OSQP(완화)/기본 모두 실패)"
        )
    stats = entry['problem'].solver_stats
    solver_name = stats.solver_name if stats is not None else 'unknown'
    inaccurate = (entry['problem'].status == cp.OPTIMAL_INACCURATE)
    return elapsed, solver_name, inaccurate


def _assert_pcs_circle(unit_p, unit_q, S_arr, label):
    """★ 4차 개정 검산(지시서 요구): 다각형 근사가 실제로 원 제약 sqrt(P^2+Q^2)<=S*(1+tol)
    을 지키는지 모든 (unit,scenario,t)에서 확인한다. 위반이 하나라도 있으면 즉시
    AssertionError로 중단·보고한다(다각형은 원 제약의 내접 근사일 뿐이므로 이게 깨지면
    정식화 버그다 - lower_lp._assert_physics의 원 제약 검증과 같은 계보, CLAUDE.md 7절
    LP검증#2)."""
    S_col = np.asarray(S_arr, dtype=float)[:, None]
    limit = S_col * (1.0 + PCS_CIRCLE_TOL)
    for s, P in unit_p.items():
        Q = unit_q[s]
        apparent = np.sqrt(P ** 2 + Q ** 2)
        over = apparent > limit
        if np.any(over):
            i_idx, t_idx = np.where(over)
            i0, t0 = int(i_idx[0]), int(t_idx[0])
            raise AssertionError(
                f"{label}: PCS 원 제약 위반(scenario={s}) - unit={i0} t={t0} "
                f"apparent={apparent[i0, t0]:.6f} > S*(1+tol)={limit[i0, 0]:.6f} "
                f"(위반 총 {int(np.sum(over))}건 중 첫 건만 표시)"
            )


def _compute_padding_stats(unit_p_ch, unit_p_dis):
    """★ 작업 A (검산 2-2 재설계): 동시충방전 패딩 크기를 계측한다(판정 없이 수치만 -
    임계 판정은 호출부가 한다).

    검사량은 곱(P_ch*P_dis, 단위 MW^2 - P_dis 스케일에 비례해 위양성을 내던 이전
    설계)이 아니라 패딩 크기 자체 pad=min(P_ch,P_dis)[MW]다. q_penalty>=s_app-P_ch-P_dis
    항은 같은 시각에 P_ch/P_dis를 함께 eps만큼 늘려도 P_net(=P_dis-P_ch)이 불변인
    구조적 허점(패딩 exploit)을 갖는데, pad는 그 패딩의 크기를 직접·선형으로 잰다
    (모듈 docstring "검산 2-2" 절의 손익분기 유도 참조).

    ★ _assert_pcs_circle은 이것을 잡을 수 없다 - 패딩은 P_net을 바꾸지 않으므로
    sqrt(P_net^2+Q^2)<=S를 위반하지 않는다.

    반환: max_pad_mw(전 scenario/unit/t 중 최댓값), location=(scenario,t,unit),
    annual_won_implied(=ANNUAL_WON_PER_PAD_MW*max_pad_mw)."""
    max_pad = 0.0
    location = None
    for s, Pch in unit_p_ch.items():
        Pdis = unit_p_dis[s]
        pad = np.minimum(Pch, Pdis)
        i0, t0 = np.unravel_index(int(np.argmax(pad)), pad.shape)
        if float(pad[i0, t0]) > max_pad:
            max_pad = float(pad[i0, t0])
            location = (s, int(t0), int(i0))
    return dict(
        max_pad_mw=max_pad, location=location,
        annual_won_implied=ANNUAL_WON_PER_PAD_MW * max_pad,
    )


# ============================================================
# 3) AVG_DAYS 스케줄 계산 (avg 전용 - peak은 3차에서 baselines['unit_p_zero']로 대체)
# ============================================================

def _compute_schedule(entry, S, E, bus, loss_table, scenarios, v_sq_line_table=None,
                      ac_flow_table=None):
    """scenarios(PM.AVG_DAYS)를 프로토타입 LP로 풀어 unit_p/unit_q를 확정하고, 순수
    solve 시간(컴파일 제외)을 별도로 측정한다. v_sq_line_table이 주어지면(작업 B)
    method='qp'의 손실항에 그 시나리오의 기저 전압 제곱(branch, from-bus 기준)으로
    보정을 적용한다(QP_V2_CORRECTION 플래그를 따름 - _set_params 참조).

    ★ 4차 개정 - 타이밍 설계 수정: 워밍업 1회(컴파일 포함, 결과값 확정) +
    N_TIMING_REPS(5)회 재-solve(컴파일 제외, 결과값은 버림) 후 **중앙값**을
    solve_time으로 보고한다. warmup_total(컴파일 포함 1회 총합)도 반환한다.

    반환: unit_p, unit_q, schedule_aux, solve_time_median, warmup_total, solver_names,
    inaccurate_flags, pad_stats. schedule_aux의 p_ch/p_dis/s_app/q_penalty는 모두 워밍업
    solve 직후 복사한 확정값이며 타이밍 반복 결과가 아니다.
    """
    unit_p, unit_q = {}, {}
    unit_p_ch, unit_p_dis = {}, {}
    unit_s_app, unit_q_penalty = {}, {}
    solver_names, inaccurate_flags = {}, {}

    def _set_and_solve(s):
        smp = PM.SMP_PER_MWH[s]
        profile = PM.LOAD[s]
        lhs_rows = None
        if entry['method'] == 'pwl':
            lhs_rows = _lhs_rows_for(loss_table, bus, s, entry['M'])
        v_sq = v_sq_line_table[s] if (v_sq_line_table is not None) else None
        qe_ac = None
        if ac_flow_table is not None:
            qe_ac = ac_flow_table[s]['q_from']
        _set_params(
            entry, S, E, [bus], profile, smp, lhs_row_values=lhs_rows, v_sq_line=v_sq,
            qe_base_ac_mvar=qe_ac,
        )
        return _solve_timed(entry)

    # ---- 워밍업: 컴파일 포함, 결과값 확정(.copy()로 이후 재-solve의 아리아싱 방지) ----
    warmup_total = 0.0
    for s in scenarios:
        elapsed, solver_name, inaccurate = _set_and_solve(s)
        warmup_total += elapsed
        solver_names[s] = solver_name
        inaccurate_flags[s] = inaccurate
        v = entry['vars']
        unit_p[s] = np.array(v['P_net'].value, copy=True)
        unit_q[s] = np.array(v['Q'].value, copy=True)
        unit_p_ch[s] = np.array(v['P_ch'].value, copy=True)
        unit_p_dis[s] = np.array(v['P_dis'].value, copy=True)
        unit_s_app[s] = (
            np.array(v['s_app'].value, copy=True) if v['s_app'] is not None
            else np.full_like(unit_p[s], np.nan)
        )
        unit_q_penalty[s] = (
            np.array(v['q_penalty'].value, copy=True) if v['q_penalty'] is not None
            else np.zeros_like(unit_p[s])
        )

    label = f"{entry['method']}/M={entry['M']}/bus={bus}"
    _assert_pcs_circle(unit_p, unit_q, np.atleast_1d(np.asarray(S, dtype=float)), label)

    # ---- 검산 2-2 (재설계, 작업 A) - 항상 계측, 임계 초과 시에만 중단 ----
    pad_stats = _compute_padding_stats(unit_p_ch, unit_p_dis)
    if pad_stats['max_pad_mw'] > PAD_ABORT_MW:
        loc = pad_stats['location']
        raise AssertionError(
            f"{label}: 패딩 크기 max_pad={pad_stats['max_pad_mw']:.3e} MW > "
            f"PAD_ABORT_MW={PAD_ABORT_MW} - 위치(scenario={loc[0]}, t={loc[1]}, "
            f"unit={loc[2]}), 함의 연간 영향={pad_stats['annual_won_implied']:.2f}원/년 "
            "(상세 근거는 모듈 docstring '검산 2-2' 절 참조)."
        )

    # ---- 타이밍 반복: 컴파일 제외(같은 Parameter, 재-solve만), 결과값은 버림 ----
    rep_totals = []
    for _ in range(N_TIMING_REPS):
        rep_total = 0.0
        for s in scenarios:
            elapsed, _, _ = _set_and_solve(s)
            rep_total += elapsed
        rep_totals.append(rep_total)
    solve_time_median = float(np.median(rep_totals))

    schedule_aux = dict(
        p_ch=unit_p_ch, p_dis=unit_p_dis, s_app=unit_s_app,
        q_penalty=unit_q_penalty,
    )
    return (unit_p, unit_q, schedule_aux, solve_time_median, warmup_total, solver_names,
            inaccurate_flags, pad_stats)


def _group_stats(unit_q, scenarios, tol=1e-6):
    """scenarios 그룹 안에서 0이 아닌 시각 수와 Q 합(Mvar)을 센다."""
    n_nonzero = 0
    total = 0.0
    for s in scenarios:
        arr = unit_q[s][0]
        n_nonzero += int(np.sum(np.abs(arr) > tol))
        total += float(np.sum(arr))
    return n_nonzero, total


def _compute_free_zone_stats(unit_q_lp, S_val, scenarios=None):
    """★ 4차 개정(QP 단위 수정 라운드) 지시서 2-5, 작업 C에서 필터 수정: 판정 없이
    수치만 계측한다.

    q_penalty는 각 phi=atan(Q/P)가 pi/POLY_N 이하인 부채꼴 전체에서 정확히 0이 된다
    (그 구간에서는 theta=0 다각형 제약(P<=s_app)만 걸리고 Q는 등장하지 않으므로
    s_app이 P까지 내려가고 q_penalty도 0이 된다). 이 구간에서는 PCS 손실이 계상되지
    않아 LP가 손실계수의 크기와 무관하게 다각형 꼭짓점(free_zone_width)까지 Q를
    밀어붙일 유인이 생긴다.
    - free_zone_width = S*sin(pi/POLY_N)          [Mvar] - 무료로 쓸 수 있는 Q의 폭
    - max_uncounted_loss = S*(1-cos(pi/POLY_N))   [MVA] - 그 구간 안에서 최대로
      계상되지 않는 피상전력 증분.

    ★ 작업 C 수정: 기존 필터(arr>0.0)는 "q_lp=0"과 "수치적으로 거의 0(예: 1e-12,
    부동소수점 잡음)"을 구분하지 못해 P1에서 71/72(사실상 전체)가 "무료구간 안"으로
    오판됐다. FREE_ZONE_NONZERO_TOL(1e-6, _group_stats와 동일 계보의 "비영" 기준)로
    엄격한 하한을 두고, 분모(비영 시각 수)와 그 비를 함께 낸다 - 지시서가 요구한
    "0 < q_lp <= free_zone_width"를 수치적으로 안전하게 구현한 것이다."""
    if scenarios is None:
        scenarios = PM.AVG_DAYS
    free_zone_width = float(S_val) * np.sin(np.pi / PM.POLY_N)
    max_uncounted_loss = float(S_val) * (1.0 - np.cos(np.pi / PM.POLY_N))
    n_nonzero = 0
    n_in_zone = 0
    n_at_vertex = 0
    for s in scenarios:
        arr = unit_q_lp[s][0]
        nonzero = arr > FREE_ZONE_NONZERO_TOL
        n_nonzero += int(np.sum(nonzero))
        in_zone = nonzero & (arr <= free_zone_width + 1e-9)
        n_in_zone += int(np.sum(in_zone))
        n_at_vertex += int(np.sum(np.abs(arr - free_zone_width) < 1e-6))
    frac_in_zone = (n_in_zone / n_nonzero) if n_nonzero > 0 else float('nan')
    return dict(free_zone_width=free_zone_width, max_uncounted_loss=max_uncounted_loss,
                n_in_zone=n_in_zone, n_at_vertex=n_at_vertex, n_nonzero=n_nonzero,
                frac_in_zone=frac_in_zone)


def _build_qstar_unit_q(point_id, qstar_full):
    """qstar_full(dict[(point_id,scenario,t)]->q_star_t)에서 이 point_id의 ALL_DAYS
    (1,T) 배열 딕셔너리를 재구성한다.

    ★ 3차 개정 통제 설계: PEAK_DAYS는 CSV에 값이 있어도 항상 0.0으로 둔다(원본
    probe_q_selective.py는 AVG/PEAK 구분 없이 매 시각 독립적으로 손실 채널만 보고
    q_star를 찾았으므로 PEAK_DAYS에도 큰 값이 들어 있다 - 그러나 그 손실저감이
    b_defer/b_loss에 반영되는지는 이 실험의 범위 밖(모듈 docstring "통제 설계" 절)이라
    기준해도 프로토타입 LP와 동일하게 Q=0으로 재구성해야 공정한 비교가 된다)."""
    unit_q_star = {s: np.zeros((1, PM.TIME_STEPS)) for s in PM.ALL_DAYS}
    for s in PM.AVG_DAYS:
        for t in range(PM.TIME_STEPS):
            unit_q_star[s][0, t] = qstar_full.get((point_id, s, t), 0.0)
    # PEAK_DAYS는 위 초기화(np.zeros)로 이미 0 - 명시적으로 다시 채우지 않는다(CSV 값을
    # 참조조차 하지 않는 것이 "무시했다"를 코드로 보여주는 가장 분명한 방법이다).
    return unit_q_star


def _raw_peak_qstar_magnitude(point_id, qstar_full):
    """참고용(3차 개정에서는 사용하지 않음) - 원본 CSV에 저장된 PEAK_DAYS q_star가
    실제로 얼마나 컸는지 보고하기 위한 값. _build_qstar_unit_q와 달리 CSV 값을 그대로
    읽는다."""
    n_nonzero, total = 0, 0.0
    for s in PM.PEAK_DAYS:
        for t in range(PM.TIME_STEPS):
            v = qstar_full.get((point_id, s, t), 0.0)
            if abs(v) > 1e-6:
                n_nonzero += 1
            total += v
    return n_nonzero, total


# ============================================================
# 4) 기준해(probe_q_selective.py) 로딩 + 통제점별 (a)/(c) 기준값
# ============================================================

def _find_latest_csv(prefix):
    candidates = []
    for d in (RESULTS_DIR, ROOT_RESULTS_DIR):
        candidates += glob.glob(os.path.join(d, f'{prefix}_*.csv'))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _load_selective_qstar_full(path):
    """point_id,scenario,t -> q_star_t. ALL_DAYS 전부 싣는다(PEAK_DAYS 무시 여부는
    _build_qstar_unit_q가 소비 시점에 결정 - 로딩 단계에서는 원본을 그대로 보존한다)."""
    table = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            table[(r['point_id'], r['scenario'], int(r['t']))] = float(r['q_star_t'])
    return table


def _print_qstar_avg_distribution(qstar_full):
    """3차 개정 지시서 요구사항: "probe_q_selective의 q_star 분포(AVG_DAYS)를 먼저
    출력해 실제 최적 Q가 어느 범위에 있는지 확인"한다. SEGMENT_BOUNDARIES가 이 분포와
    여전히 맞는지 실행마다 눈으로 확인할 수 있게 한다(코드 상수는 작성 시점 기준이라
    매 실행 데이터에 자동으로 맞춰지지 않는다 - 어긋나면 이 상수부터 재검토할 것)."""
    section('기준해 q_star의 AVG_DAYS 분포 (PWL 경계 설정 근거 확인용)')
    vals = []
    for point in POINTS:
        for s in PM.AVG_DAYS:
            for t in range(PM.TIME_STEPS):
                v = qstar_full.get((point['point_id'], s, t))
                if v is not None and v != 0.0:
                    vals.append(v)
    if not vals:
        print('  AVG_DAYS 비영 q_star가 하나도 없음 - PWL 경계 설정 근거 확인 불가', flush=True)
        return
    arr = np.array(vals)
    pct = np.percentile(arr, [0, 25, 50, 75, 90, 100])
    print(f"  비영 표본 수 = {len(arr)}", flush=True)
    print(f"  분포(Mvar): min={pct[0]:.4f} p25={pct[1]:.4f} median={pct[2]:.4f} "
          f"p75={pct[3]:.4f} p90={pct[4]:.4f} max={pct[5]:.4f}", flush=True)
    print(f"  SEGMENT_BOUNDARIES(M=2)={SEGMENT_BOUNDARIES[2]}, (M=4)={SEGMENT_BOUNDARIES[4]} "
          "- median이 첫 구간(또는 둘째 구간) 안에 있고 max가 최상단 경계(0.5)를 넘지 "
          "않아야 이 경계 설정이 여전히 유효하다.", flush=True)
    top_boundary = SEGMENT_BOUNDARIES[4][-1]
    if arr.max() > top_boundary:
        print(f"  ★ 경고: 실측 max({arr.max():.4f})가 PWL 최상단 경계({top_boundary})를 "
              "초과한다 - 해당 시각의 근사오차는 '표현 불가'로 해석해야 하며 M을 더 "
              "늘려도 해소되지 않는다.", flush=True)

    section('참고: 원본 CSV의 PEAK_DAYS q_star 크기 (3차 개정에서는 사용하지 않음)')
    for point in POINTS:
        n_peak_raw, sum_peak_raw = _raw_peak_qstar_magnitude(point['point_id'], qstar_full)
        print(f"  {point['point_id']}: 원본 PEAK q_star {n_peak_raw}시각/{sum_peak_raw:.3f} "
              "Mvar (이 실험에서는 0으로 재구성 - 피크일 손실이 b_defer에 미치는 영향은 "
              "편익함수 재구조화 후 별도 안건, 모듈 docstring '통제 설계' 절 참조)", flush=True)


def _pwl_predicted_reduction(loss_row, boundaries, q_level):
    """PWL(주어진 boundaries)이 "세그먼트를 순서대로(0번부터) 채운다"는 가정 하에
    Q=q_level에서 예측하는 손실 저감량. q_level이 boundaries의 경계점과 정확히 같으면
    실측 시컨트들의 텔레스코핑 합이라 실측과 대수적으로 정확히 같다(항등식). 경계
    "사이"면 그 세그먼트의 실측 기울기로 선형보간한 값이다 - 이것이 곧 PWL 모델
    자체가 예측하는 값이다(순서대로 채운다는 가정의 타당성은 _check_pwl_slope_
    monotonicity가 별도로 확인한다 - 지시서 2-4)."""
    slopes = _segment_slopes(loss_row, boundaries)
    reduction = 0.0
    remaining = q_level
    for (lo, hi), slope in zip(zip(boundaries[:-1], boundaries[1:]), slopes):
        width = hi - lo
        seg_q = min(max(remaining, 0.0), width)
        reduction += slope * seg_q
        remaining -= seg_q
        if remaining <= 0.0:
            break
    return reduction


def _q_prediction_ratio_percentiles(pred_arr, actual_arr):
    """(예측/실측) 비의 분포(min/p25/median/p75/max)를 낸다 - 지시서 B-5. actual=0인
    표본은 비가 정의되지 않으므로 제외하고 그 개수를 함께 반환한다(판정 없이 수치만)."""
    mask = actual_arr != 0.0
    n_excluded = int(np.sum(~mask))
    if not np.any(mask):
        return None, n_excluded
    ratio = pred_arr[mask] / actual_arr[mask]
    pct = np.percentile(ratio, [0, 25, 50, 75, 100])
    return pct, n_excluded


def _diagnose_q_prediction_gap(point, loss_table, q_level, v_sq_line_table,
                               ac_flow_table, alpha_by_point):
    """★ 4차 개정(QP 단위 수정 라운드 + 작업 B): "Q=q_level에서 두 방식이 예측하는
    손실 저감량을 각각 출력해 실측 조류계산 값과 대조"한다(P3에서 QP/PWL 사용량이
    2.5배 갈린 원인이 QP 자신의 손실추정에 있는지 분리해서 보기 위함).

    q_level이 SEGMENT_BOUNDARIES[4]의 경계점(0.05)과 정확히 같으면 PWL의 이 지점
    예측은 시컨트들의 텔레스코핑 합이라 실측(loss_table)과 **대수적으로 정확히 같다**
    (근사가 아니라 항등식). q_level이 경계 "사이"(0.0375)면 PWL도 그 세그먼트 내
    선형보간 오차를 그대로 드러낸다.

    ★ 작업 B(지시서 B-4): QP_V2_CORRECTION 플래그와 무관하게 이 진단 함수는 항상
    보정 전(V=1 근사)과 보정 후(v_sq_line_table 사용) 예측을 **나란히** 계산·출력한다
    - 보정이 실제로 오차를 줄였는지 사용자가 직접 봐야 하기 때문이다. 또한(B-5)
    각 방식의 (예측/실측) 비의 분포(min/p25/median/p75/max)를 낸다 - 중앙값만으로는
    "편향이 일정한 비율"인지 확인할 수 없다.

    자동판정 없음 - 수치만 출력한다(지시서 "자동판정 금지")."""
    bus = point['b']
    idx0 = Q_BOUNDARY_POINTS.index(0.0)
    idxq = Q_BOUNDARY_POINTS.index(q_level)
    boundaries4 = SEGMENT_BOUNDARIES[4]
    boundaries9 = SEGMENT_BOUNDARIES[9]
    is_exact_boundary4 = q_level in boundaries4
    is_exact_boundary9 = q_level in boundaries9
    topo = lower_lp._get_topology()
    D, r_pu, n_bus = topo['D'], topo['r_pu'], topo['n_bus']
    bus_onehot_np = np.zeros((1, n_bus))
    bus_onehot_np[0, bus] = 1.0
    _base_load_p_bus, base_load_q_bus = lower_lp.base_load_bus_arrays()

    actual_vals, pwl_vals, pwl9_vals = [], [], []
    qp_uncorr_vals, qp_corr_vals = [], []
    qp_ac_vals, qp_gross_vals, qp_both_vals = [], [], []
    alpha = alpha_by_point[point['point_id']][q_level]
    grossup = (1.0 / alpha) if np.isfinite(alpha) and alpha != 0.0 else np.nan
    for s in PM.AVG_DAYS:
        arr = loss_table[bus][s]   # (T, len(Q_BOUNDARY_POINTS)) - P_inj=0 고정으로 실측된 것
        profile = np.asarray(PM.LOAD[s], dtype=float)
        load_q_val = base_load_q_bus[:, None] * profile[None, :]        # (n_bus,T)
        Qe_base = D @ (load_q_val / PM.S_BASE_MVA)                       # (n_branch,T), pu
        q_fixed = np.full((1, PM.TIME_STEPS), q_level)
        dQe = D @ ((bus_onehot_np.T @ q_fixed) / PM.S_BASE_MVA)           # (n_branch,T), pu
        # QP 모델의 손실 저감 예측[MW, V=1 근사] = r_pu*(2*Qe_base*dQe-dQe^2)*S_BASE_MVA
        # (P는 0으로 고정된 실측과 맞춰 P_e_base/dP_e 항은 생략).
        per_branch_uncorr = r_pu[:, None] * (2.0 * Qe_base * dQe - dQe ** 2) * PM.S_BASE_MVA
        qp_uncorr_t = per_branch_uncorr.sum(axis=0)  # (T,), MW
        # ★ 작업 B: V^2 보정판 - 기저 조류계산의 from-bus 전압 제곱(v_sq_line_table)으로
        # 나눈다(추가 조류계산 없음 - _measure_loss_table이 이미 캐시한 값 재사용).
        v_sq = v_sq_line_table[s]   # (n_branch, T)
        qp_corr_t = (per_branch_uncorr / v_sq).sum(axis=0)
        # 작업 4-3: ac_flow_table의 기저 송단 q_from_mvar를 절댓값 처리해
        # Baran-Wu load-positive Qe_base와 맞춘다. 코드 위치:
        # _diagnose_q_prediction_gap -> ac_flow_table[s]['q_from'].
        Qe_base_ac = np.abs(ac_flow_table[s]['q_from']) / PM.S_BASE_MVA
        per_branch_ac = (
            r_pu[:, None] * (2.0 * Qe_base_ac * dQe - dQe ** 2) * PM.S_BASE_MVA
        )
        qp_ac_t = (per_branch_ac / v_sq).sum(axis=0)
        qp_gross_t = qp_corr_t * grossup
        qp_both_t = qp_ac_t * grossup
        for t in range(PM.TIME_STEPS):
            actual = float(arr[t, idx0] - arr[t, idxq])
            actual_vals.append(actual)
            pwl_vals.append(_pwl_predicted_reduction(arr[t], boundaries4, q_level))
            pwl9_vals.append(_pwl_predicted_reduction(arr[t], boundaries9, q_level))
            qp_uncorr_vals.append(float(qp_uncorr_t[t]))
            qp_corr_vals.append(float(qp_corr_t[t]))
            qp_ac_vals.append(float(qp_ac_t[t]))
            qp_gross_vals.append(float(qp_gross_t[t]))
            qp_both_vals.append(float(qp_both_t[t]))

    actual_arr = np.array(actual_vals)
    pwl_arr = np.array(pwl_vals)
    pwl9_arr = np.array(pwl9_vals)
    qp_uncorr_arr = np.array(qp_uncorr_vals)
    qp_corr_arr = np.array(qp_corr_vals)
    qp_ac_arr = np.array(qp_ac_vals)
    qp_gross_arr = np.array(qp_gross_vals)
    qp_both_arr = np.array(qp_both_vals)
    pwl_err = pwl_arr - actual_arr
    pwl9_err = pwl9_arr - actual_arr
    qp_uncorr_err = qp_uncorr_arr - actual_arr
    qp_corr_err = qp_corr_arr - actual_arr
    boundary_note4 = (
        "텔레스코핑 항등식 - 0이 아니면 구현을 의심할 것" if is_exact_boundary4
        else "경계 사이 - PWL 자신의 세그먼트 내 선형보간 오차(실측과 다를 수 있음, 정상)"
    )
    boundary_note9 = (
        "텔레스코핑 항등식 - 0이 아니면 구현을 의심할 것" if is_exact_boundary9
        else "경계 사이 - PWL 자신의 세그먼트 내 선형보간 오차(실측과 다를 수 있음, 정상)"
    )
    print(f"  {point['point_id']}(bus={bus}) Q={q_level} 손실저감 예측 대조 "
          f"(AVG_DAYS {len(actual_arr)}개 시각):", flush=True)
    print(f"    실측(loss_table)      : median={np.median(actual_arr):.6f} MW  "
          f"mean={np.mean(actual_arr):.6f} MW", flush=True)
    print(f"    PWL(M=4) 예측         : median={np.median(pwl_arr):.6f} MW  "
          f"실측 대비 오차 median={np.median(pwl_err):+.6f} MW, "
          f"max|오차|={np.max(np.abs(pwl_err)):.6f} MW ({boundary_note4})", flush=True)
    print(f"    PWL(M=9) 예측         : median={np.median(pwl9_arr):.6f} MW  "
          f"실측 대비 오차 median={np.median(pwl9_err):+.6f} MW, "
          f"max|오차|={np.max(np.abs(pwl9_err)):.6f} MW ({boundary_note9})", flush=True)
    print(f"    QP 예측(보정 전,V=1)  : median={np.median(qp_uncorr_arr):.6f} MW  "
          f"실측 대비 오차 median={np.median(qp_uncorr_err):+.6f} MW, "
          f"max|오차|={np.max(np.abs(qp_uncorr_err)):.6f} MW", flush=True)
    print(f"    QP 예측(보정 후,V^2)  : median={np.median(qp_corr_arr):.6f} MW  "
          f"실측 대비 오차 median={np.median(qp_corr_err):+.6f} MW, "
          f"max|오차|={np.max(np.abs(qp_corr_err)):.6f} MW", flush=True)
    print("    Qe_base AC 교체 규약: ac_flow_table[s]['q_from']의 절댓값 "
          "([_diagnose_q_prediction_gap])", flush=True)
    print(f"    그로스업: 이 실행·통제점·Q={q_level}의 median alpha=A/dLoss="
          f"{alpha:.6f}, 1/alpha={grossup:.6f}", flush=True)

    # ---- 작업 4-2: 네 변형의 median 오차와 (예측/실측) 비 분포 ----
    variants = (
        ('(a) 현행[V^2]', qp_corr_arr),
        ('(b) +Qe_base AC', qp_ac_arr),
        ('(c) +grossup', qp_gross_arr),
        ('(d) +Qe_base AC+grossup', qp_both_arr),
    )
    for label, pred_arr in variants:
        err = pred_arr - actual_arr
        print(f"    변형[{label}]: 예측 median={np.median(pred_arr):.6f} MW, "
              f"실측 대비 median 오차={np.median(err):+.6f} MW", flush=True)
        pct, n_excl = _q_prediction_ratio_percentiles(pred_arr, actual_arr)
        if pct is None:
            print(f"      비율분포: 전 표본 actual=0 - 비 정의 불가", flush=True)
        else:
            print(f"      비율분포(예측/실측, actual=0 {n_excl}건 제외): "
                  f"min={pct[0]:.4f} p25={pct[1]:.4f} median={pct[2]:.4f} "
                  f"p75={pct[3]:.4f} max={pct[4]:.4f}", flush=True)

    # 기존 대조 출력도 보존한다.
    report_rows = []
    for label, pred_arr in (
        ('PWL(M=4)', pwl_arr), ('PWL(M=9)', pwl9_arr),
        ('QP(보정전)', qp_uncorr_arr),
    ):
        pct, n_excl = _q_prediction_ratio_percentiles(pred_arr, actual_arr)
        if pct is None:
            print(f"    비율분포[{label}]: 전 표본 actual=0 - 비 정의 불가", flush=True)
        else:
            print(f"    비율분포[{label}](예측/실측, actual=0 {n_excl}건 제외): "
                  f"min={pct[0]:.4f} p25={pct[1]:.4f} median={pct[2]:.4f} "
                  f"p75={pct[3]:.4f} max={pct[4]:.4f}", flush=True)
    for label, pred_arr in (
        ('PWL(M=4)', pwl_arr), ('PWL(M=9)', pwl9_arr),
        ('QP current', qp_corr_arr), ('QP +Qe_base AC', qp_ac_arr),
        ('QP +grossup', qp_gross_arr), ('QP +both', qp_both_arr),
    ):
        pct, n_excl = _q_prediction_ratio_percentiles(pred_arr, actual_arr)
        report_rows.append(dict(
            point_id=point['point_id'], bus=bus, q_level=q_level, variant=label,
            actual_median_mw=float(np.median(actual_arr)),
            predicted_median_mw=float(np.median(pred_arr)),
            error_median_mw=float(np.median(pred_arr - actual_arr)),
            ratio_min=(float(pct[0]) if pct is not None else np.nan),
            ratio_p25=(float(pct[1]) if pct is not None else np.nan),
            ratio_median=(float(pct[2]) if pct is not None else np.nan),
            ratio_p75=(float(pct[3]) if pct is not None else np.nan),
            ratio_max=(float(pct[4]) if pct is not None else np.nan),
            n_excluded=n_excl,
        ))
    return report_rows


def _verify_q_zero_anchor(point):
    """★ 4차 개정(QP 단위 수정 라운드) 검산 2-3: Q를 0으로 고정해 풀면 q_penalty가
    정확히(수치적으로) 0이어야 한다 - 그래야 "Q를 도입해도 Q=0 해가 기존 LP
    (force_q_zero=True)와 동일하다"는 회귀 앵커가 성립한다.

    QP entry(bus_onehot이 build-time 상수라 이 검증에 딱 맞음 - PWL/QP가 공유하는
    s_app/q_penalty/다각형 로직은 method 무관이므로 QP 하나로 그 공유 로직을 검증하면
    충분하다)를 만들고, entry['problem']은 이미 지어진 뒤라 제약을 追加할 수 없으므로
    같은 Variable/Parameter를 참조하는 새 cp.Problem(목적함수는 그대로, 제약에 Q==0만
    추가)을 만들어 별도로 푼다. AVG_DAYS의 첫 시나리오 하나로 충분하다(Q=0 anchor는
    시나리오 무관 - s_app/q_penalty 관계식 자체의 성질이므로)."""
    bus = int(point['b'])
    entry = _build_problem_proto('qp', n=1, T=PM.TIME_STEPS, bus_idx=bus)
    s = PM.AVG_DAYS[0]
    # v_sq_line을 넘기지 않는다(작업 B의 V^2 보정 미적용) - q_penalty=0 앵커는
    # s_app/폴리곤 제약만의 구조적 성질이라 손실항(rsmp)의 V^2 보정 여부와 무관하게
    # 항상 성립해야 한다(Q=0으로 고정되면 loss_term의 크기와 무관하게 s_app이 |P_net|
    # 까지 내려가는 것이 최적이므로).
    _set_params(
        entry, point['S'], point['E'], [bus], PM.LOAD[s], PM.SMP_PER_MWH[s],
        apply_qp_corrections=False,
    )

    q_zero_constraints = list(entry['problem'].constraints) + [entry['vars']['Q'] == 0]
    test_problem = cp.Problem(entry['problem'].objective, q_zero_constraints)
    test_problem.solve(solver=cp.CLARABEL, max_iter=2000)
    assert test_problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE), (
        f"{point['point_id']}: Q=0 고정 회귀 앵커 검증 solve 실패(status={test_problem.status})"
    )
    q_penalty_val = entry['vars']['q_penalty'].value
    assert np.all(np.abs(q_penalty_val) <= Q_ZERO_ANCHOR_TOL), (
        f"{point['point_id']}: Q=0으로 고정했는데 q_penalty!=0 "
        f"(최대={np.max(np.abs(q_penalty_val)):.3e} > tol={Q_ZERO_ANCHOR_TOL}) - "
        "회귀 앵커 위반. s_app<=S*cos(pi/N) 제약이나 다각형 부등식을 재검토할 것."
    )


def _compute_point_baselines(point, qstar_full):
    """(a) Q=0과 (c) Q=q_star(AVG_DAYS만, PEAK_DAYS는 0)의 j_net을 한 번에 계산한다 -
    둘 다 force_q_zero=True LP의 P를 공유하므로 evaluate_particle을 한 번만 호출하면
    된다. unit_p_zero(ALL_DAYS)도 함께 반환한다 - _process가 PEAK_DAYS 스케줄로
    그대로 재사용한다(3차 개정: 프로토타입 LP가 더 이상 PEAK_DAYS를 풀지 않으므로)."""
    x = np.array([point['b'], point['S'], point['E']], dtype=float)
    detail_zero = _evaluate_with_force_q(x, True)
    if detail_zero.get('diverged'):
        return dict(j_net_a=None, j_net_c=None, bus=None, unit_p_zero=None)

    j_net_a = detail_zero['j_net']
    unit_p = detail_zero['unit_p']   # dict[ALL_DAYS] -> (1,T)
    unit_q_zero = detail_zero['unit_q']
    b_arr, S_arr, E_arr = detail_zero['b'], detail_zero['S'], detail_zero['E']
    bus = int(b_arr[0])

    unit_q_star = _build_qstar_unit_q(point['point_id'], qstar_full)
    result_c = _reinject_and_evaluate(b_arr, S_arr, E_arr, unit_p, unit_q_star)
    j_net_c = result_c['j_net'] if not result_c.get('diverged') else None

    pcs_zero = {
        s: C_PCS * (
            np.sqrt(np.asarray(unit_p[s]) ** 2 + np.asarray(unit_q_zero[s]) ** 2)
            - np.abs(np.asarray(unit_p[s]))
        ) for s in PM.ALL_DAYS
    }
    loss_line_zero = {
        s: np.asarray(detail_zero['loss_ess'][s])
        - np.asarray(pcs_zero[s]).sum(axis=0)
        for s in PM.ALL_DAYS
    }
    return dict(
        j_net_a=j_net_a, j_net_c=j_net_c, bus=bus, unit_p_zero=unit_p,
        unit_q_zero=unit_q_zero, loss_line_zero=loss_line_zero,
        b_energy_a=detail_zero['b_energy'],
    )


# ============================================================
# 5) 통제점 x 방식 1회 처리 (AVG_DAYS만 LP로 풀고, PEAK_DAYS는 baselines 재사용)
# ============================================================

def _reinject_and_capture_line_loss(b, S, E, unit_p, unit_q):
    """기존 AC 평가의 조류계산을 그대로 쓰면서 성공한 각 시각의 선로손실만 읽는다."""
    original_run_pf = evaluate._run_pf_with_retry
    captured = []

    def _wrapped_run_pf(net, *args, **kwargs):
        ok = original_run_pf(net, *args, **kwargs)
        if ok:
            captured.append(float(net.res_line.pl_mw.sum()))
        return ok

    evaluate._run_pf_with_retry = _wrapped_run_pf
    try:
        result = _reinject_and_evaluate(b, S, E, unit_p, unit_q)
    finally:
        evaluate._run_pf_with_retry = original_run_pf

    line_loss = {}
    offset = 0
    for s in PM.ALL_DAYS:
        line_loss[s] = np.asarray(captured[offset:offset + PM.TIME_STEPS], dtype=float)
        offset += PM.TIME_STEPS
    return result, line_loss


def _annual_schedule_components(unit_p, unit_q, loss_line_ess, b_energy_value,
                                loss_table, bus, schedule_aux=None):
    """작업 2의 연간 원장. (5)는 AC 평가에서 캡처한 실제 선로손실을 직접 쓴다."""
    arb_proxy = q_loss_measured = pcs_true_cost = pcs_charged_cost = 0.0
    actual_line_loss_reduction = 0.0
    base_flow = evaluate._get_base_flow()
    for s in PM.AVG_DAYS:
        smp = np.asarray(PM.SMP_PER_MWH[s], dtype=float)
        weight = float(PM.N_WEEKDAYS[s]) * PM.DT_HOURS
        p_net = np.asarray(unit_p[s], dtype=float).sum(axis=0)
        q_units = np.asarray(unit_q[s], dtype=float)
        pcs_true_units = C_PCS * (
            np.sqrt(np.asarray(unit_p[s]) ** 2 + q_units ** 2)
            - np.abs(np.asarray(unit_p[s]))
        )
        pcs_true = pcs_true_units.sum(axis=0)
        if schedule_aux is not None:
            pcs_charged = C_PCS * np.asarray(
                schedule_aux['q_penalty'][s], dtype=float
            ).sum(axis=0)
        else:
            pcs_charged = np.zeros(PM.TIME_STEPS)

        arr = loss_table[bus][s]
        measured_reduction = np.array([
            np.interp(q_units[0, t], Q_BOUNDARY_POINTS, arr[t, 0] - arr[t, :])
            for t in range(PM.TIME_STEPS)
        ])
        arb_proxy += float(np.sum(smp * p_net)) * weight
        q_loss_measured += float(np.sum(smp * measured_reduction)) * weight
        pcs_true_cost += float(np.sum(smp * pcs_true)) * weight
        pcs_charged_cost += float(np.sum(smp * pcs_charged)) * weight
        actual_line_loss_reduction += float(np.sum(
            smp * (np.asarray(base_flow['loss'][s]) - np.asarray(loss_line_ess[s]))
        )) * weight

    ledger_rhs = arb_proxy - pcs_true_cost + actual_line_loss_reduction
    return dict(
        arb_proxy=arb_proxy,
        q_loss_measured=q_loss_measured,
        pcs_true_cost=pcs_true_cost,
        pcs_charged_cost=pcs_charged_cost,
        actual_line_loss_reduction=actual_line_loss_reduction,
        b_energy=b_energy_value,
        ledger_rhs=ledger_rhs,
        ledger_residual=b_energy_value - ledger_rhs,
        actual_minus_p0=actual_line_loss_reduction - q_loss_measured,
        net_true=arb_proxy + q_loss_measured - pcs_true_cost,
        pcs_gap=pcs_true_cost - pcs_charged_cost,
    )


def _process(point, method, M, avg_entry, loss_table, qstar_full, baselines, v_sq_line_table,
             ac_flow_table=None):
    S, E, bus = point['S'], point['E'], point['b']
    b_arr = np.array([bus], dtype=float)
    S_arr = np.array([S], dtype=float)
    E_arr = np.array([E], dtype=float)

    (unit_p_avg, unit_q_avg, schedule_aux, solve_time_median, warmup_total, solvers_avg,
     inacc_avg, pad_stats) = _compute_schedule(
        avg_entry, S, E, bus, loss_table, PM.AVG_DAYS,
        v_sq_line_table=v_sq_line_table, ac_flow_table=ac_flow_table,
    )

    # ---- PEAK_DAYS: 3차 개정 통제 설계 - force_q_zero=True LP의 P를 그대로, Q=0 ----
    unit_p_zero = baselines['unit_p_zero']
    unit_p_lp = dict(unit_p_avg, **{s: unit_p_zero[s] for s in PM.PEAK_DAYS})
    unit_q_lp = dict(unit_q_avg, **{s: np.zeros((1, PM.TIME_STEPS)) for s in PM.PEAK_DAYS})
    for s in PM.PEAK_DAYS:   # 방어적 점검 - 위 construct가 실제로 Q=0을 만들었는지
        assert np.all(unit_q_lp[s] == 0.0), f'{point["point_id"]}/{s}: PEAK Q가 0이 아님(버그)'

    result_b, loss_line_b = _reinject_and_capture_line_loss(
        b_arr, S_arr, E_arr, unit_p_lp, unit_q_lp
    )
    j_net_b = result_b['j_net'] if not result_b.get('diverged') else None

    j_net_a = baselines['j_net_a']
    j_net_c = baselines['j_net_c']

    def _delta(x_, y_):
        return (x_ - y_) if (x_ is not None and y_ is not None) else None

    # ---- 무료 Q 구간 계측 (지시서 2-5 - 판정 없이 수치만, (method,M,point) 단위로
    # 계산해 아래 각 ts_row에 그대로 반복 기입한다) ----
    free_zone = _compute_free_zone_stats(unit_q_lp, S, scenarios=PM.AVG_DAYS)

    # ---- 시각별 오차 - AVG_DAYS만 (지시서: "시각별 |q_lp-q_star|(AVG_DAYS만)") ----
    ts_rows = []
    abs_errs = []
    for s in PM.AVG_DAYS:
        for t in range(PM.TIME_STEPS):
            q_lp = float(unit_q_lp[s][0, t])
            p_ch = float(schedule_aux['p_ch'][s][0, t])
            p_dis = float(schedule_aux['p_dis'][s][0, t])
            p_net = p_dis - p_ch
            s_app = float(schedule_aux['s_app'][s][0, t])
            q_penalty = float(schedule_aux['q_penalty'][s][0, t])
            pcs_true = C_PCS * (np.hypot(p_net, q_lp) - abs(p_net))
            pcs_charged = C_PCS * q_penalty
            key = (point['point_id'], s, t)
            q_star = qstar_full.get(key)
            if q_star is None:
                continue
            abs_err = abs(q_lp - q_star)
            rel_err = (abs_err / q_star) if q_star != 0.0 else ''
            abs_errs.append(abs_err)
            pad_loc = pad_stats['location']
            ts_rows.append(dict(
                method=method, M=(M if M is not None else ''),
                point_id=point['point_id'], scenario=s, t=t,
                q_lp=q_lp, q_star=q_star, abs_err=abs_err, rel_err=rel_err,
                inaccurate=inacc_avg[s],
                free_zone_width_mvar=free_zone['free_zone_width'],
                max_uncounted_loss_mva=free_zone['max_uncounted_loss'],
                n_free_zone_hours=free_zone['n_in_zone'],
                n_vertex_stuck_hours=free_zone['n_at_vertex'],
                n_nonzero_q_hours=free_zone['n_nonzero'],
                frac_in_free_zone=free_zone['frac_in_zone'],
                max_pad_mw=pad_stats['max_pad_mw'],
                max_pad_scenario=(pad_loc[0] if pad_loc else ''),
                max_pad_t=(pad_loc[1] if pad_loc else ''),
                max_pad_unit=(pad_loc[2] if pad_loc else ''),
                max_pad_annual_won_implied=pad_stats['annual_won_implied'],
                qp_v2_correction=QP_V2_CORRECTION,
                p_ch=p_ch, p_dis=p_dis, p_net=p_net, s_app=s_app,
                q_penalty=q_penalty, pcs_true=pcs_true, pcs_charged=pcs_charged,
            ))

    annual_components = _annual_schedule_components(
        unit_p_lp, unit_q_lp, loss_line_b, result_b['b_energy'],
        loss_table, bus, schedule_aux=schedule_aux,
    )
    unit_q_star = _build_qstar_unit_q(point['point_id'], qstar_full)
    n_avg_lp, sum_avg_lp = _group_stats(unit_q_lp, PM.AVG_DAYS)
    n_avg_star, sum_avg_star = _group_stats(unit_q_star, PM.AVG_DAYS)

    return dict(
        ts_rows=ts_rows, abs_errs=abs_errs, result_b=result_b,
        j_net_a=j_net_a, j_net_b=j_net_b, j_net_c=j_net_c,
        b_minus_a=_delta(j_net_b, j_net_a), c_minus_a=_delta(j_net_c, j_net_a),
        c_minus_b=_delta(j_net_c, j_net_b),
        solve_time_avg=solve_time_median, warmup_time_avg=warmup_total,
        dpp_preserved_avg=avg_entry['dpp_preserved'],
        solvers_avg=solvers_avg, inaccurate_avg=inacc_avg,
        q_avg_lp=(n_avg_lp, sum_avg_lp), q_avg_star=(n_avg_star, sum_avg_star),
        free_zone=free_zone, pad_stats=pad_stats, annual_components=annual_components,
    )


# ============================================================
# 계측 추가 라운드 - 작업 C: 기준시간 측정 (현행 LP와의 apples-to-apples 비교)
# ============================================================

def _measure_baseline_timing(label, entry, loss_table):
    """지시서 작업C-2/C-3: 현재 방식들과 **완전히 동일한 타이밍 하네스**(_compute_schedule -
    워밍업 1회 후 N_TIMING_REPS회 반복, 컴파일 제외, 중앙값)로 기준 구성 하나를 3개
    통제점 전부에 대해 측정한다. loss_table을 받지만 method='none'/'pcs_only'는 그 안의
    lhs 관련 로직을 타지 않으므로(_compute_schedule/_set_params가 method=='pwl'일 때만
    사용) 실질적으로 쓰이지 않는다 - 하네스를 다른 방식과 동일하게 유지하기 위해
    시그니처만 맞춘다. 판정 없음 - 수치만."""
    section(f'기준 측정: {label}')
    print(f"DPP 유지: {entry['dpp_preserved']}", flush=True)
    times = {}
    for point in POINTS:
        S, E, bus = point['S'], point['E'], point['b']
        (_up, _uq, _aux, solve_time_median, warmup_total, _sn, _inf,
         _pad) = _compute_schedule(
            entry, S, E, bus, loss_table, PM.AVG_DAYS
        )
        times[point['point_id']] = solve_time_median
        print(f"  {point['point_id']}: solve_time(median, {N_TIMING_REPS}회, 컴파일 제외)="
              f"{solve_time_median:.4f}초, warmup(컴파일 포함,1회)={warmup_total:.4f}초",
              flush=True)
    return times


# ============================================================
# stdout 요약 (자동판정 없음 - 수치만 제시)
# ============================================================

def _fmt_won(v):
    return 'N/A' if v is None else f'{v:,.2f}원'


def _print_method_summary(method, M, per_point_outcomes, avg_entries_for_diag,
                           pwl_build_time, qp_build_times, baseline1_times):
    label = f"{method.upper()}" + (f" M={M}" if M is not None else "")
    section(f'방식 {label}')

    dpp_avg = per_point_outcomes[0][1]['dpp_preserved_avg']
    print(f"DPP 유지: solve_avg 변형={dpp_avg}", flush=True)
    if not dpp_avg:
        for entry in avg_entries_for_diag:
            _diagnose_dpp_terms(entry)

    # ---- Problem 빌드 비용 (지시서: "실제 배포 시에도 그런지 명시, 필요하면 별도 항목 보고") ----
    if method == 'pwl':
        print(f"Problem 빌드(1회, {len(per_point_outcomes)}포인트 공유) = "
              f"{pwl_build_time:.4f}초", flush=True)
    else:
        print("Problem 빌드(포인트마다 재구축 - bus_onehot을 build-time 상수로 굽어야 "
              "DPP가 유지되므로, 3포인트 공유가 불가능하다):", flush=True)
        for pid, bt in qp_build_times.items():
            print(f"  {pid}: {bt:.4f}초", flush=True)
        # ★ 작업E(계측 추가 라운드): 아래 문구는 이전 버전(0.022~0.030초 빌드시간을 전제로
        # "지배적 병목이 될 수 있다"는 판정성 결론)을 수정한 것이다 - 직전 실행 실측
        # 빌드시간은 P2 0.0093초/P3 0.0094초(P1의 0.0646초는 첫 빌드라 콜드스타트 포함)로
        # 훨씬 작았고, 판정에 쓰인 "입자수x세대수x run수" 외삽도 버스 정의역이 1~32
        # 32개뿐이라는 사실(워커당 32개 사전 빌드 캐시가 가능 - 32x0.0093=~0.30초 일회성)을
        # 반영하지 않은 채였다. 코드에 박아 둔 판정 문구는 근거 수치가 바뀌어도 자동으로
        # 갱신되지 않는다 - 이것이 이 스크립트가 stdout에 판정성 문장을 넣지 않는 이유의
        # 실례다(모듈 docstring 참조).
        print("  QP는 bus를 build-time 상수로 굽기 때문에 버스마다 Problem을 새로 지어야 "
              "한다. 측정된 빌드 시간은 위와 같다. 버스 정의역의 크기와 캐싱 전략에 따라 "
              "실배포 비용이 달라지므로, 이 수치를 근거로 사용자가 판단한다.", flush=True)

    all_abs_errs = []
    signed_errs = []
    for point, outcome in per_point_outcomes:
        all_abs_errs += outcome['abs_errs']
        for row in outcome['ts_rows']:
            signed_errs.append(row['q_lp'] - row['q_star'])

        print(f"\n  {point['point_id']}:", flush=True)
        print(f"    j_net(a:Q=0)={_fmt_won(outcome['j_net_a'])}  "
              f"j_net(b:Q=q_lp)={_fmt_won(outcome['j_net_b'])}  "
              f"j_net(c:Q=q_star)={_fmt_won(outcome['j_net_c'])}", flush=True)
        print(f"    (b-a)={_fmt_won(outcome['b_minus_a'])}  "
              f"(c-a)={_fmt_won(outcome['c_minus_a'])}  "
              f"(c-b)={_fmt_won(outcome['c_minus_b'])}", flush=True)

        n_avg, sum_avg = outcome['q_avg_lp']
        n_star, sum_star = outcome['q_avg_star']
        print(f"    q_lp(AVG_DAYS) 분포: {n_avg}시각/{sum_avg:.3f} Mvar  vs  "
              f"q_star(AVG_DAYS): {n_star}시각/{sum_star:.3f} Mvar", flush=True)

        annual = outcome['annual_components']
        print("    연간 원화 계측 (가중: PM.AVG_DAYS, PM.N_WEEKDAYS, "
              "PM.SMP_PER_MWH, PM.DT_HOURS):", flush=True)
        print(f"      (1) 차익 대리={annual['arb_proxy']:,.2f}원  "
              f"(3) 실제 PCS 비용={annual['pcs_true_cost']:,.2f}원", flush=True)
        print(f"      (5) 실제 총 선로손실 저감={annual['actual_line_loss_reduction']:,.2f}원  "
              f"(6) b_energy={annual['b_energy']:,.2f}원", flush=True)
        print(f"      (1)-(3)+(5)={annual['ledger_rhs']:,.2f}원  "
              f"(6)-[(1)-(3)+(5)]={annual['ledger_residual']:+.6f}원", flush=True)
        print(f"      (2) P_inj=0 loss_table Q 손실저감={annual['q_loss_measured']:,.2f}원  "
              f"(5)-(2)={annual['actual_minus_p0']:+,.2f}원", flush=True)
        print(f"      (4) LP 계상 PCS 비용={annual['pcs_charged_cost']:,.2f}원  "
              f"(3)-(4)={annual['pcs_gap']:+,.2f}원", flush=True)

        fz = outcome['free_zone']
        print(f"    무료구간(지시서 2-5, 작업C 필터수정): 폭={fz['free_zone_width']:.6f} Mvar, "
              f"최대미계상피상전력={fz['max_uncounted_loss']:.6f} MVA, "
              f"비영시각수={fz['n_nonzero']}, 구간내시각수={fz['n_in_zone']} "
              f"(비율={fz['frac_in_zone']:.3f}), 꼭짓점고정시각수={fz['n_at_vertex']} "
              "(판정 없음 - 이 통제점의 판별력 참고자료)", flush=True)

        pad = outcome['pad_stats']
        loc = pad['location']
        loc_str = f"scenario={loc[0]},t={loc[1]},unit={loc[2]}" if loc else "N/A"
        print(f"    패딩(검산 2-2, 작업A 재설계): max_pad={pad['max_pad_mw']:.3e} MW "
              f"(위치: {loc_str}), 함의 연간 영향={pad['annual_won_implied']:.4f}원/년 "
              "(임계 없이 항상 보고 - PAD_WARN_MW/PAD_ABORT_MW는 각각 경고수집/중단 "
              "임계일 뿐 이 수치 자체의 판정이 아니다)", flush=True)

        ratio = (outcome['warmup_time_avg'] / outcome['solve_time_avg']
                 if outcome['solve_time_avg'] > 0 else float('nan'))
        # ★ 계측 추가 라운드 작업C-3: 배율의 기준을 REFERENCE_SOLVE_TIME_SEC(다른 머신·다른
        # 범위 참고값, 아래 실행 메타에서 1회만 별도 표시)에서 기준1(force_q_zero=True 등가,
        # 이 스크립트·같은 통제점·같은 하네스로 실측)로 교체했다 - 지시서 C-1/C-3 근거.
        b1 = baseline1_times.get(point['point_id']) if baseline1_times else None
        if b1 and b1 > 0:
            ratio_label = f"{outcome['solve_time_avg'] / b1:.2f}x 기준1[force_q_zero=True 등가]"
        else:
            ratio_label = "기준1 값 없음"
        print(f"    solve_time(avg, 컴파일 제외, {N_TIMING_REPS}회 중앙값)="
              f"{outcome['solve_time_avg']:.4f}초 ({ratio_label}), "
              f"warmup(컴파일 포함, 1회)={outcome['warmup_time_avg']:.4f}초 "
              f"(warmup/median={ratio:.2f}x)", flush=True)

    if all_abs_errs:
        arr = np.array(all_abs_errs)
        print(f"\n  |q_lp-q_star| 중앙값={np.median(arr):.4f} Mvar, 최댓값={np.max(arr):.4f} Mvar",
              flush=True)
    if signed_errs:
        print(f"  부호 편향(평균 q_lp-q_star) = {np.mean(signed_errs):+.4f} Mvar "
              f"(양수면 LP가 과다공급, 음수면 과소공급 경향)", flush=True)


def _print_padding_summary(pad_warn_events):
    """★ 작업 A-3(b): max_pad가 PAD_WARN_MW를 넘었지만 PAD_ABORT_MW는 넘지 않아
    중단되지 않고 수집된 (method,M,point) 조합을 실행 말미에 요약한다(지시서
    "중단하지 말고 위반 항목을 수집해 두었다가 실행 말미에 요약 출력하라" -
    _print_pwl_monotonicity_check와 같은 패턴). 원인을 단정하지 않는다 - 수치·위치만."""
    section(f'패딩 경고 요약 (max_pad > PAD_WARN_MW={PAD_WARN_MW}, 중단하지 않고 수집됨)')
    if not pad_warn_events:
        print(f"  0건 - 모든 (method,M,point) 조합에서 max_pad <= {PAD_WARN_MW} MW", flush=True)
        return
    for ev in pad_warn_events:
        loc = ev['location']
        loc_str = f"scenario={loc[0]},t={loc[1]},unit={loc[2]}" if loc else "N/A"
        print(f"  ⚠ {ev['method']}/M={ev['M']}/{ev['point_id']}: "
              f"max_pad={ev['max_pad_mw']:.3e} MW (위치: {loc_str}), "
              f"함의 연간 영향={ev['annual_won_implied']:.2f}원/년", flush=True)
    print(f"\n  총 {len(pad_warn_events)}건 (PAD_ABORT_MW={PAD_ABORT_MW} 미만이라 중단되지는 "
          "않았다 - 상세 근거는 모듈 docstring '검산 2-2' 절 참조)", flush=True)


def _print_solver_diagnostics(solver_usage, inaccurate_events, total_solves):
    section('솔버 진단 (지시서 "solver 정확도 문제" 절)')
    print(f"총 LP 호출 수 = {total_solves}", flush=True)
    for name, count in sorted(solver_usage.items(), key=lambda kv: -kv[1]):
        print(f"  선택된 솔버 {name}: {count}회 ({count / total_solves * 100:.1f}%)", flush=True)

    n_inacc = len(inaccurate_events)
    print(f"OPTIMAL_INACCURATE로 마무리된 호출 수 = {n_inacc} "
          f"({n_inacc / total_solves * 100:.1f}%)", flush=True)
    if inaccurate_events:
        by_combo = {}
        for ev in inaccurate_events:
            key = (ev['kind'], ev['method'], ev['M'])
            by_combo[key] = by_combo.get(key, 0) + 1
        print("  조합별 발생 빈도(kind, method, M) -> 횟수:", flush=True)
        for key, count in sorted(by_combo.items(), key=lambda kv: -kv[1]):
            print(f"    {key} -> {count}회", flush=True)
        print("  ⚠ 위 조합의 q_lp/j_net(b)는 근사해 기반이다 - 3차 개정의 max_iter 강화 "
              "+ 완화 tolerance 재시도까지 거친 뒤에도 남은 것이므로, 이 조합은 결과에서 "
              "제외하거나(엄격) ts_rows CSV의 inaccurate=True 행을 걸러내고(관대) 해석할 "
              "것 - 지시서 대안 중 하나를 선택할 것.", flush=True)
    else:
        print("  OPTIMAL_INACCURATE 0건 - 3차 개정의 솔버 조정이 2차의 10%를 해소한 것으로 "
              "보인다(단 이 세션은 실행하지 않았으므로 실제 실행 결과로 재확인할 것).",
              flush=True)


def _print_interpretation():
    section('해석 지침 (자동판정 없음 - 수치를 보고 사람이 판단할 것)')
    print(
        "- (c-a)는 'Q만의 순효과 상한'(같은 P, q_star는 근사·외삽 없는 격자탐색 기준해,\n"
        "  AVG_DAYS만 - PEAK_DAYS는 3차 개정에서 양쪽 다 0으로 고정했다),\n"
        "  (b-a)는 '이 LP를 실제로 배포하면 얻는 순효과'(P도 함께 재최적화), (c-b)는 이\n"
        "  프로토타입이 기준해 대비 얼마나 못 미치는지의 격차다.\n"
        "- q_lp의 AVG_DAYS 분포를 q_star의 그것과 비교할 것(실행 초반 stdout에 q_star\n"
        "  전체 분포가 먼저 출력된다) - 두 그룹 시각 수(n)와 총량(Mvar)이 비슷할수록 LP가\n"
        "  기준해의 '언제 얼마나' 패턴을 잘 재현한다는 뜻이다.\n"
        "- 오차 크기(중앙값·최댓값)뿐 아니라 부호 편향도 볼 것. PWL은 오목함수(손실 체감)를\n"
        "  구간별 시컨트(직선)로 근사하므로 실제 곡선보다 아래에 있어 Q를 과소평가하는 편향이,\n"
        "  QP는 V^2~=1 근사가 전압이 낮은 지점에서 손실을 과대평가해 Q를 과소공급하는 편향이\n"
        "  예상된다 - 실측 부호가 다르면 그 자체가 보고할 발견이다.\n"
        "- M을 늘릴 때(1->2->4) |q_lp-q_star|가 실제로 줄어드는지가 PWL 경계 재설정이\n"
        "  효과가 있었는지의 직접 지표다(2차에서는 개선되지 않았는데, 그것이 경계 설정\n"
        "  탓이었는지 이번에 판별된다 - 위 'q_star 분포' 절과 함께 볼 것).\n"
        "- QP는 DPP를 위해 포인트마다 Problem을 새로 짓는다(PWL은 M별로 한 번, 3포인트\n"
        "  재사용) - solve_time(median, 컴파일 제외)과는 별도로 '방식 요약'에 찍히는\n"
        "  Problem 빌드 시간을 볼 것. 실배포(PSO 통합) 시 QP가 입자마다 이 비용을\n"
        "  반복 지불해야 하는지가 정확도 못지않게 중요한 채택 기준이다.\n"
        "- OPTIMAL_INACCURATE 비율이 높은 조합은 위 수치 자체의 신뢰도가 낮다는 뜻이니\n"
        "  '솔버 진단' 절과 함께 읽을 것.",
        flush=True,
    )


# ============================================================
# 메인
# ============================================================

def _make_path():
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(RESULTS_DIR, f'probe_lp_loss_proto_{hostname}_{ts}.csv')


def _alpha_path(main_csv_path):
    stem, ext = os.path.splitext(main_csv_path)
    return f'{stem}_alpha{ext}'


def _alpha_raw_path(main_csv_path):
    stem, ext = os.path.splitext(main_csv_path)
    return f'{stem}_alpha_raw{ext}'


def _report_path(main_csv_path):
    stem, _ext = os.path.splitext(main_csv_path)
    return f'{stem}_report.md'


def _md_num(value, digits=6):
    if value is None:
        return 'N/A'
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return str(value)
    return f'{value:.{digits}f}'


def _write_numeric_report(path, metadata, timings, alpha_constant_rows,
                          baseline_rows, method_rows, prediction_rows, alpha_rows):
    """이번 지시의 핵심 stdout 수치만 Markdown 표로 복제한다. 판정·해석 문장은 넣지 않는다."""
    lines = ['# probe_lp_loss_proto numeric report', '']
    lines += ['## Metadata', '', '| key | value |', '|---|---:|']
    for key, value in metadata:
        lines.append(f'| {key} | {value} |')

    lines += ['', '## Timing and calls', '',
              '| item | expected_calls | actual_calls | wall_time_sec | sec_per_call |',
              '|---|---:|---:|---:|---:|']
    for row in timings:
        lines.append(
            f"| {row['item']} | {_md_num(row.get('expected_calls'))} | "
            f"{_md_num(row.get('actual_calls'))} | {_md_num(row.get('wall_time_sec'))} | "
            f"{_md_num(row.get('sec_per_call'))} |"
        )

    lines += ['', '## Gross-up alpha', '',
              '| point | measured_alpha | QP_GROSSUP_ALPHA | difference |',
              '|---|---:|---:|---:|']
    for row in alpha_constant_rows:
        lines.append(
            f"| {row['point_id']} | {_md_num(row['measured_alpha'])} | "
            f"{_md_num(QP_GROSSUP_ALPHA)} | {_md_num(row['difference'])} |"
        )

    def _ledger_table(title, rows):
        lines.extend([
            '', f'## {title}', '',
            '| label | point | j_net_a | j_net_b | j_net_c | q_hours | q_sum_mvar | solve_time_sec | '
            'dpp | (1) arb | (3) pcs | (5) line_loss | (6) b_energy | residual | (2) p0_q_loss | (5)-(2) |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        ])
        for row in rows:
            lines.append(
                f"| {row['label']} | {row['point_id']} | {_md_num(row.get('j_net_a'), 2)} | "
                f"{_md_num(row.get('j_net_b'), 2)} | {_md_num(row.get('j_net_c'), 2)} | "
                f"{_md_num(row.get('q_hours'))} | "
                f"{_md_num(row.get('q_sum_mvar'))} | {_md_num(row.get('solve_time_sec'))} | "
                f"{_md_num(row.get('dpp'))} | {_md_num(row['arb_proxy'], 2)} | "
                f"{_md_num(row['pcs_true_cost'], 2)} | "
                f"{_md_num(row['actual_line_loss_reduction'], 2)} | "
                f"{_md_num(row['b_energy'], 2)} | {_md_num(row['ledger_residual'])} | "
                f"{_md_num(row['q_loss_measured'], 2)} | "
                f"{_md_num(row['actual_minus_p0'], 2)} |"
            )

    _ledger_table('Baseline (a)', baseline_rows)
    _ledger_table('Methods', method_rows)

    lines += ['', '## Loss prediction', '',
              '| point | q | variant | actual_median_mw | predicted_median_mw | '
              'error_median_mw | ratio_min | ratio_p25 | ratio_median | ratio_p75 | ratio_max |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in prediction_rows:
        lines.append(
            f"| {row['point_id']} | {_md_num(row['q_level'], 4)} | {row['variant']} | "
            f"{_md_num(row['actual_median_mw'])} | {_md_num(row['predicted_median_mw'])} | "
            f"{_md_num(row['error_median_mw'])} | {_md_num(row['ratio_min'])} | "
            f"{_md_num(row['ratio_p25'])} | {_md_num(row['ratio_median'])} | "
            f"{_md_num(row['ratio_p75'])} | {_md_num(row['ratio_max'])} |"
        )

    lines += ['', '## Alpha P-Q sweep', '',
              '| point | p | q | feasible | alpha_median | dloss_ratio_to_p0_median | identity_max_mw |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in alpha_rows:
        lines.append(
            f"| {row['point_id']} | {_md_num(row['p_level'])} | {_md_num(row['q_level'], 4)} | "
            f"{row['feasible']} | {_md_num(row['alpha_median'])} | "
            f"{_md_num(row['dloss_ratio_to_p0_median'])} | "
            f"{_md_num(row['identity_abs_max_mw'], 9)} |"
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"핵심 수치 Markdown 저장: {path}", flush=True)


def _open_csv_writer(path):
    """★ 작업 A-5: CSV를 한 번에 다 쓰지 않고 파일을 실행 내내 열어 둔 채 조합이
    끝날 때마다 즉시 append+flush한다(아래 _append_rows) - 어느 조합에서
    AssertionError로 중단되더라도 그 이전까지 완료된 조합의 결과가 CSV에 남는다."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    f = open(path, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=TS_CSV_FIELDS, extrasaction='ignore')
    writer.writeheader()
    f.flush()
    return f, writer


def _append_rows(f, writer, rows):
    if not rows:
        return
    writer.writerows(rows)
    f.flush()
    os.fsync(f.fileno())


if __name__ == '__main__':
    _check_env()

    # ---- 작업 D: 실행 메타 기록 ----
    run_started_at = datetime.datetime.now()
    hostname = socket.gethostname()
    section('실행 메타 + 정식화 상수 확인 (작업 D / 지시서 1-4 - 사후 역산·재현성 확인용)')
    print(f"실행 시작={run_started_at.isoformat()}  호스트={hostname}", flush=True)
    print(f"POLY_N={PM.POLY_N}  S_BASE_MVA={PM.S_BASE_MVA}  ETA_PCS={PM.ETA_PCS}  "
          f"C_PCS={C_PCS:.6f}  Q_DIAG_LEVELS={Q_DIAG_LEVELS}", flush=True)
    print(f"Q_BOUNDARY_POINTS={Q_BOUNDARY_POINTS}", flush=True)
    # ---- 작업 지시(9차세션) 참고 항목: POLY_N 변경과 함께 자동으로 바뀌는 값들이
    # 실제로 새 POLY_N을 반영해 나오는지 확인용(free_zone_width는 _compute_free_zone_stats와
    # 동일 공식 - S*sin(pi/POLY_N), 통제점별 S는 POINTS에서 그대로 가져온다) ----
    print(f"PCS_FACET_THRESHOLD={PCS_FACET_THRESHOLD:.6f}", flush=True)
    for _point in POINTS:
        _fzw = float(_point['S']) * np.sin(np.pi / PM.POLY_N)
        print(f"  free_zone_width[{_point['point_id']}](S={_point['S']}) = {_fzw:.6f} Mvar",
              flush=True)
    print(f"QP_V2_CORRECTION={QP_V2_CORRECTION}  QP_QE_BASE_AC={QP_QE_BASE_AC}  "
          f"QP_GROSSUP={QP_GROSSUP}  QP_GROSSUP_ALPHA={QP_GROSSUP_ALPHA}", flush=True)
    print("QP Qe_base AC source=_measure_loss_table의 q=0 기저 패스 "
          "ac_flow_table[scenario]['q_from']; sign=abs(q_from_mvar); "
          "optimization use=_compute_schedule->_set_params; 추가 조류계산=0", flush=True)
    print(f"PAD_WARN_MW={PAD_WARN_MW}  PAD_ABORT_MW={PAD_ABORT_MW}", flush=True)
    # ★ 계측 추가 라운드 작업C-4: 머신 간 비교용(추가 의존성 없이 얻을 수 있는 범위).
    print(f"CPU={platform.processor()}  논리코어수={os.cpu_count()}  "
          f"플랫폼={platform.platform()}", flush=True)
    # ★ 계측 추가 라운드 작업C-1/C-3: 이 값은 더 이상 solve_time 배율의 기준이 아니다
    # (다른 머신·다른 범위의 참고값일 뿐 - 상수 정의부 주석 참조). 배율은 아래 "기준
    # 측정: force_q_zero=True 등가"로 이 실행 안에서 새로 잰 값을 쓴다.
    print(f"REFERENCE_SOLVE_TIME_SEC={REFERENCE_SOLVE_TIME_SEC} (참고용 - 다른 머신·다른 "
          "범위(evaluate 전체)에서 측정된 값, solve_time 배율의 기준으로 쓰지 않는다)",
          flush=True)

    selective_path = _find_latest_csv('probe_q_selective')
    if selective_path is None:
        raise FileNotFoundError(
            f"{RESULTS_DIR}(또는 {ROOT_RESULTS_DIR})에 probe_q_selective_*.csv가 없다 - "
            "probe_q_selective.py를 먼저 실행할 것(이 스크립트의 기준해 입력)."
        )
    section(f'기준해 입력: {os.path.basename(selective_path)}')
    qstar_full = _load_selective_qstar_full(selective_path)
    print(f'ALL_DAYS q_star 매칭 항목 {len(qstar_full)}개 로드', flush=True)

    _print_qstar_avg_distribution(qstar_full)

    net, q_scale, p_total, q_total_before = _build_net_with_pf(TARGET_PF)
    base_p, base_q = _prepare_condition(net)

    section('검산 2-3: Q=0 고정 회귀 앵커 확인 (통제점별 1회, 위반 시 즉시 중단)')
    for point in POINTS:
        _verify_q_zero_anchor(point)
        print(f"  {point['point_id']}: q_penalty(Q=0)=0 확인됨", flush=True)

    section('기준값 j_net(a:Q=0) / j_net(c:Q=q_star, AVG_DAYS만) 재구성')
    baselines_by_point = {}
    for point in POINTS:
        b = _compute_point_baselines(point, qstar_full)
        baselines_by_point[point['point_id']] = b
        print(f"  {point['point_id']} (bus={b['bus']}): j_net(a)={_fmt_won(b['j_net_a'])}, "
              f"j_net(c)={_fmt_won(b['j_net_c'])}", flush=True)

    section(f'손실 테이블 실측 (전 32버스 x AVG_DAYS 3개 x 24시간 x 경계점 '
            f'{len(Q_BOUNDARY_POINTS)}개 '
            '+ 작업B: 기저 from-bus 전압^2/AC조류 캐시 + 작업A-1: 통제점 3개 전 q지점 '
            '전체상태 캐시, 추가 조류계산 없음)')
    # ---- 작업 지시(9차세션) 작업2: 벽시계 시간 + 실측 pp.runpp 호출횟수 계측
    # (계측 범위는 _measure_loss_table 호출 전후로만 한정 - 연산 순서 불변, 값 미변경) ----
    with _count_runpp_calls() as _runpp_counter:
        _loss_table_t0 = time.perf_counter()
        loss_table, v_sq_line_table, ac_flow_table, v_bus_table, ac_full_table = _measure_loss_table(
            net, base_p, base_q
        )
        loss_table_wall_time_sec = time.perf_counter() - _loss_table_t0
    loss_table_n_runpp_calls = _runpp_counter['n']
    loss_table_expected_calls = (
        len(ALL_BUSES) * len(PM.AVG_DAYS) * PM.TIME_STEPS * len(Q_BOUNDARY_POINTS)
    )
    loss_table_sec_per_call = (loss_table_wall_time_sec / loss_table_n_runpp_calls
                                if loss_table_n_runpp_calls > 0 else float('nan'))
    print(f"loss_table 벽시계 시간(초) = {loss_table_wall_time_sec:.6f}", flush=True)
    print(f"loss_table 예상 격자 호출 횟수 = {loss_table_expected_calls}", flush=True)
    print(f"loss_table 실측 pp.runpp 호출 횟수 = {loss_table_n_runpp_calls}", flush=True)
    print(f"loss_table 회당 시간(초) = {loss_table_sec_per_call:.6f}", flush=True)
    print("사실: 아래 연간 Q 손실저감 계측은 P_inj=0에서 실측한 loss_table을 사용한다. "
          "P!=0 실제 급전점과의 차이는 별도 alpha P×Q 스윕이 계측한다.", flush=True)

    _print_pwl_monotonicity_check(loss_table)

    # ---- 계측 추가 라운드(1차분): 작업 A (세그먼트 기울기 실측 노출) ----
    _print_pwl_segment_slopes(loss_table)
    _write_lhs_csv(_make_lhs_path(), loss_table)

    # ---- 계측 추가 라운드(1차분): 작업 B (Qe_base vs AC 실측 조류 대조) ----
    topo = lower_lp._get_topology()
    _print_sign_convention_check(ac_flow_table, topo)
    _print_qe_base_ac_comparison(ac_flow_table, topo)

    # ---- 계측 추가 라운드(1차분): 작업 C (손실공식·전압규약 항등성 검증) ----
    _print_loss_formula_identity_check(ac_flow_table, v_bus_table, net)

    # ---- 계측 추가 라운드(2차분): 작업 A-2/A-3/A-4 (AC 되먹임 성분 분해) ----
    _print_ac_feedback_decomposition(ac_full_table, loss_table, v_sq_line_table, net)
    _print_dqe_ratio(ac_full_table)
    alpha_by_point = _alpha_by_point_and_level(ac_full_table, loss_table, net)
    alpha_constant_rows = []
    section('QP_GROSSUP_ALPHA 실행 중 실측 대조')
    for point in POINTS:
        pid = point['point_id']
        measured_alpha = alpha_by_point[pid]['pooled']
        difference = measured_alpha - QP_GROSSUP_ALPHA
        alpha_constant_rows.append(dict(
            point_id=pid, measured_alpha=measured_alpha, difference=difference,
        ))
        print(f"  {pid}: measured_alpha={measured_alpha:.6f}  "
              f"QP_GROSSUP_ALPHA={QP_GROSSUP_ALPHA:.6f}  "
              f"difference={difference:+.6f}", flush=True)

    baseline_report_rows = []
    section('기준 (a:Q=0) 연간 원장')
    for point in POINTS:
        baseline = baselines_by_point[point['point_id']]
        if baseline['unit_p_zero'] is None:
            continue
        ledger_a = _annual_schedule_components(
            baseline['unit_p_zero'], baseline['unit_q_zero'],
            baseline['loss_line_zero'], baseline['b_energy_a'],
            loss_table, int(point['b']),
        )
        baseline['ledger_a'] = ledger_a
        print(f"  {point['point_id']}: (1)={ledger_a['arb_proxy']:,.2f}원  "
              f"(3)={ledger_a['pcs_true_cost']:,.2f}원  "
              f"(5)={ledger_a['actual_line_loss_reduction']:,.2f}원  "
              f"(6)={ledger_a['b_energy']:,.2f}원", flush=True)
        print(f"    (1)-(3)+(5)={ledger_a['ledger_rhs']:,.2f}원  "
              f"잔차={ledger_a['ledger_residual']:+.6f}원  "
              f"(2)={ledger_a['q_loss_measured']:,.2f}원  "
              f"(5)-(2)={ledger_a['actual_minus_p0']:+,.2f}원", flush=True)
        baseline_report_rows.append(dict(
            label='a', point_id=point['point_id'], j_net_a=baseline['j_net_a'],
            j_net_b=baseline['j_net_a'], j_net_c=baseline['j_net_c'],
            q_hours=0, q_sum_mvar=0.0,
            solve_time_sec=None, dpp=None, **ledger_a,
        ))

    # ---- 계측·진단 라운드 작업 3: 기존 P_inj=0 테이블과 분리된 P×Q 스윕 ----
    csv_path = _make_path()
    with _count_runpp_calls() as _alpha_runpp_counter:
        _alpha_t0 = time.perf_counter()
        alpha_rows, alpha_raw_rows, alpha_expected_calls = _measure_alpha_p_sweep(
            net, base_p, base_q
        )
        alpha_wall_time_sec = time.perf_counter() - _alpha_t0
    alpha_actual_calls = _alpha_runpp_counter['n']
    alpha_sec_per_call = (
        alpha_wall_time_sec / alpha_actual_calls if alpha_actual_calls > 0 else float('nan')
    )
    print(f"alpha P×Q 예상 격자 호출 횟수={alpha_expected_calls}", flush=True)
    print(f"alpha P×Q 실측 pp.runpp 호출 횟수={alpha_actual_calls}", flush=True)
    print(f"alpha P×Q 벽시계 시간(초)={alpha_wall_time_sec:.6f}", flush=True)
    print(f"alpha P×Q 회당 시간(초)={alpha_sec_per_call:.6f}", flush=True)
    _write_alpha_csv(_alpha_path(csv_path), alpha_rows)
    _write_alpha_raw_csv(_alpha_raw_path(csv_path), alpha_raw_rows)

    section('Q 수준 고정 손실저감 예측 대조 (PWL 텔레스코핑/보간 vs QP 근사 보정전/후, 지시서 요구)')
    prediction_report_rows = []
    for q_level in Q_DIAG_LEVELS:
        print(f"\n  -- Q={q_level} --", flush=True)
        for point in POINTS:
            if baselines_by_point[point['point_id']]['unit_p_zero'] is None:
                continue
            prediction_report_rows.extend(_diagnose_q_prediction_gap(
                point, loss_table, q_level, v_sq_line_table,
                ac_flow_table, alpha_by_point,
            ))

    # ---- 계측 추가 라운드(2차분) 작업C-2/C-3: 기준1/기준2 측정 (같은 하네스, 방식
    # 목록보다 먼저 - 지시서 "방식 목록의 맨 앞에 출력하라") ----
    none_entry = _build_problem_proto('none', n=1, T=PM.TIME_STEPS)
    pcs_only_entry = _build_problem_proto('pcs_only', n=1, T=PM.TIME_STEPS)
    baseline1_times = _measure_baseline_timing(
        "기준1 (force_q_zero=True 등가 - Q=0, 다각형/PCS 항 없음)", none_entry, loss_table
    )
    baseline2_times = _measure_baseline_timing(
        "기준2 (Q 자유 + PCS 비용 항만, 손실 편익 항 없음)", pcs_only_entry, loss_table
    )

    # ---- 작업 A-5: CSV를 실행 내내 열어 두고 조합마다 즉시 append(중단돼도 보존) ----
    csv_file, csv_writer = _open_csv_writer(csv_path)

    solver_usage = {}
    inaccurate_events = []
    pad_warn_events = []
    total_solves = 0
    method_report_rows = []

    try:
        for method, M in METHODS:
            shared_avg_entry = None
            pwl_build_time = None
            if method == 'pwl':
                t0 = time.perf_counter()
                shared_avg_entry = _build_problem_proto(method, n=1, T=PM.TIME_STEPS, M=M)
                pwl_build_time = time.perf_counter() - t0

            per_point_outcomes = []
            avg_entries_for_diag = []
            qp_build_times = {}
            for point in POINTS:
                baselines = baselines_by_point[point['point_id']]
                if baselines['unit_p_zero'] is None:
                    print(f"  {point['point_id']}: 기준(Q=0 강제) 평가 발산 -> "
                          "이 통제점 건너뜀", flush=True)
                    continue

                if method == 'pwl':
                    avg_entry = shared_avg_entry
                else:
                    t0 = time.perf_counter()
                    avg_entry = _build_problem_proto(method, n=1, T=PM.TIME_STEPS,
                                                      bus_idx=int(point['b']))
                    qp_build_times[point['point_id']] = time.perf_counter() - t0
                avg_entries_for_diag.append(avg_entry)

                outcome = _process(
                    point, method, M, avg_entry, loss_table, qstar_full,
                    baselines, v_sq_line_table, ac_flow_table,
                )
                per_point_outcomes.append((point, outcome))
                ledger = outcome['annual_components']
                q_hours, q_sum = outcome['q_avg_lp']
                method_report_rows.append(dict(
                    label=(f"PWL M={M}" if method == 'pwl' else 'QP'),
                    point_id=point['point_id'], j_net_a=outcome['j_net_a'],
                    j_net_b=outcome['j_net_b'], j_net_c=outcome['j_net_c'],
                    q_hours=q_hours, q_sum_mvar=q_sum,
                    solve_time_sec=outcome['solve_time_avg'],
                    dpp=outcome['dpp_preserved_avg'], **ledger,
                ))

                # ---- 작업 A-5: 이 조합이 끝나는 즉시 CSV에 반영 ----
                _append_rows(csv_file, csv_writer, outcome['ts_rows'])

                pad = outcome['pad_stats']
                if pad['max_pad_mw'] > PAD_WARN_MW:
                    pad_warn_events.append(dict(
                        method=method, M=M, point_id=point['point_id'],
                        max_pad_mw=pad['max_pad_mw'], location=pad['location'],
                        annual_won_implied=pad['annual_won_implied'],
                    ))

                for s, name in outcome['solvers_avg'].items():
                    solver_usage[name] = solver_usage.get(name, 0) + 1
                    total_solves += 1
                    if outcome['inaccurate_avg'][s]:
                        inaccurate_events.append(dict(
                            kind='avg', method=method, M=M,
                            point_id=point['point_id'], scenario=s,
                        ))

            if per_point_outcomes:
                _print_method_summary(method, M, per_point_outcomes, avg_entries_for_diag,
                                       pwl_build_time, qp_build_times, baseline1_times)
    finally:
        csv_file.close()
        print(f'CSV 저장(조합 완료마다 즉시 반영됨 - 작업 A-5): {csv_path}', flush=True)

    _restore_evaluate_state()

    _print_padding_summary(pad_warn_events)
    _print_solver_diagnostics(solver_usage, inaccurate_events, total_solves)
    report_metadata = [
        ('POLY_N', PM.POLY_N),
        ('QP_V2_CORRECTION', QP_V2_CORRECTION),
        ('QP_QE_BASE_AC', QP_QE_BASE_AC),
        ('QP_GROSSUP', QP_GROSSUP),
        ('QP_GROSSUP_ALPHA', QP_GROSSUP_ALPHA),
        ('Q_DIAG_LEVELS', Q_DIAG_LEVELS),
        ('Q_BOUNDARY_POINTS', Q_BOUNDARY_POINTS),
    ]
    report_timings = [
        dict(
            item='loss_table', expected_calls=loss_table_expected_calls,
            actual_calls=loss_table_n_runpp_calls,
            wall_time_sec=loss_table_wall_time_sec,
            sec_per_call=loss_table_sec_per_call,
        ),
        dict(
            item='alpha_pq', expected_calls=alpha_expected_calls,
            actual_calls=alpha_actual_calls, wall_time_sec=alpha_wall_time_sec,
            sec_per_call=alpha_sec_per_call,
        ),
    ]
    for pid, value in baseline1_times.items():
        report_timings.append(dict(
            item=f'baseline1_solve_{pid}', expected_calls=None, actual_calls=None,
            wall_time_sec=value, sec_per_call=None,
        ))
    for pid, value in baseline2_times.items():
        report_timings.append(dict(
            item=f'baseline2_solve_{pid}', expected_calls=None, actual_calls=None,
            wall_time_sec=value, sec_per_call=None,
        ))
    _write_numeric_report(
        _report_path(csv_path), report_metadata, report_timings,
        alpha_constant_rows, baseline_report_rows, method_report_rows,
        prediction_report_rows, alpha_rows,
    )
    _print_interpretation()
