# 실행 환경 (검증 완료, 재진단 불필요)

- OS: Windows (win32-x64)
- Python: 3.11 (miniconda, conda env `ess`)
- 인터프리터: `C:\Users\samsung\miniconda3\envs\ess\python.exe`
- 프로젝트: `C:\Users\samsung\summer_ess_project`
- numpy 2.2.6 (conda-forge 빌드, BLAS/LAPACK 3.9.0)
- 설치 완료: pandapower, numpy, scipy, pandas, cvxpy, numba

## 환경변수

### 필수: MKL_THREADING_LAYER=SEQUENTIAL
이 numpy는 MKL 백엔드다(`np.show_config()`의 "openblas configuration: unknown").
VS Code 확장 세션에서 MKL 기본 스레딩(Intel OpenMP)이 워커 스레드 풀을
생성하는 시점에 프로세스가 무증상 종료된다(stdout/stderr 없이 종료코드만).
`MKL_NUM_THREADS=1`로는 해결되지 않는다 — 스레드 개수가 아니라 생성 자체가
막히는 문제이므로 백엔드를 순차 실행으로 교체해야 한다.

설정 위치: conda env config vars (`conda env config vars set MKL_THREADING_LAYER=SEQUENTIAL -n ess`).
`ess` 활성화 시 항상 설정되므로 VS Code든 터미널이든 로컬 실행이면 어디서나 적용되고,
환경과 함께 다닌다. 이미 설정 완료됨 — 코드나 실행 커맨드에서 별도로 지정할 필요 없음.

검증: 이 변수 설정 후 `pp.runpp` 정상 동작, 0번 검증값 전부 재현
(손실 310.06 kW / Vmin 0.9407 pu bus 17 / line0 261.51 A / 슬랙 8.8125 MW).

성능 영향: MKL 단일 스레드화. 33버스 희소 조류계산에는 이득 손실이 거의 없고,
Pool 병렬화 시에는 오히려 워커 간 스레드 경합을 막아 유리하다.

### 설정하지 말 것: OPENBLAS_CORETYPE, OPENBLAS_NUM_THREADS
이 numpy는 OpenBLAS 빌드가 아니므로 무효인 변수다.
SIMD 감지도 정상(AVX2·AVX512_ICL 검출)이라 CORETYPE 고정은 성능만 떨어뜨린다.

## 세션 시작 시 규칙

환경 진단(패키지 버전 확인, import 테스트, 행렬곱 테스트 등)을 하지 마라.
위 값을 신뢰하라. 실제로 ImportError나 크래시가 발생했을 때만 진단하라.
필요하면 `python -c "import sys; print(sys.executable)"` 한 줄로
conda env 활성화 여부만 확인하면 충분하다.
