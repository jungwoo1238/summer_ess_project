# params update

## 표 1

| 항목 | 변경 전 | 변경 후 | 변경 여부 |
|---|---:|---:|---|
| 부하 스케일 | K_SCALE=2.288670 | K_P=2.557200538358, K_Q=1.357608260521 | 변경 |
| 목표 피상전력(MVA) | 10.0 | 10.0 | 유지 |
| 목표 종합 역률 | 0.850241 | 0.95 | 변경 |
| 무효전력 경로 PCS 효율 | 0.97 | 0.975 | 변경 |
| 충전효율 ETA_C | 0.9487 | 0.9487 | 유지 |
| 방전효율 ETA_D | 0.9487 | 0.9487 | 유지 |
| 왕복효율 ETA_C×ETA_D | 0.90003169 | 0.90003169 | 유지 |
| 슬랙 전압(pu) | 1.02 | 1.02 | 유지 |
| VALIDATION loss_kw_scaled | 296.86 | 281.25 | 변경 |
| VALIDATION vmin_pu_scaled | 0.9620 | 0.9641 | 변경 |
| VALIDATION line0_current_a_scaled | 255.99 | 255.23 | 변경 |
| VALIDATION max_line_utilization_scaled | 0.6244 | 0.6225 | 변경 |
| VALIDATION v_violation_total_scaled | 0.0 | 0.0 | 유지 |

## 표 2

| 확인 항목 | 값 | 기준 | 확인 방식 |
|---|---:|---:|---|
| 원본 P 합(MW) | 3.715 | 3.715 | build_net 실행 시 assert |
| 원본 Q 합(Mvar) | 2.300 | 2.300 | build_net 실행 시 assert |
| 적용 후 P 합(MW) | 9.500000000000 | 9.500000 | params 산술 확인 |
| 적용 후 Q 합(Mvar) | 3.122498999199 | 3.122499 | params 산술 확인 |
| 적용 후 S 합(MVA) | 10.000000000000 | 10.000000 | params 산술 확인 |
| 적용 후 종합 역률 | 0.950000000000 | 0.950000 | params 산술 확인 |
| vm_pu=1.02 하한 위반 L1 합(pu) | 0.0 | 0.0 | probe_voltage_rescale_20260730_151104 |
| vm_pu=1.02 Vmin(pu) | 0.964080638619 | 0.95 이상 | probe_voltage_rescale_20260730_151104 |
| vm_pu=1.02 Vmin 버스 | 17 |  | probe_voltage_rescale_20260730_151104 |
| 조류계산 수 | 600 | 600 | probe_voltage_rescale_20260730_151104 |
| 조류계산 발산 수 | 0 | 0 | probe_voltage_rescale_20260730_151104 |
| ETA_PCS 읽힘 | 0.975 | 0.975 | params import |
| ETA_C×ETA_D | 0.90003169 | 약 0.90 | params import |
| SLACK_VM_PU 읽힘 | 1.02 | 1.02 | params import |

## 변경 diff

```diff
-K_SCALE = 2.288670
+TARGET_PF = 0.95
+CASE33_P_MW = 3.715
+CASE33_Q_MVAR = 2.300
+TARGET_P_MW = TARGET_MVA * TARGET_PF
+TARGET_Q_MVAR = TARGET_MVA * np.sqrt(1.0 - TARGET_PF ** 2)
+K_P = TARGET_P_MW / CASE33_P_MW
+K_Q = TARGET_Q_MVAR / CASE33_Q_MVAR

-net.load['p_mw'] *= K_SCALE
-net.load['q_mvar'] *= K_SCALE
+net.load['p_mw'] *= K_P
+net.load['q_mvar'] *= K_Q

-ETA_PCS = 0.97
+ETA_PCS = 0.975
```
