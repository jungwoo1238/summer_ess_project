"""
PSO Core Implementation
=======================
ESS 최적 배치·용량 산정 프로젝트용 PSO 코어.

설계 원칙 (확정된 F 하이퍼파라미터 반영):
  - 관성 가중치 w: 0.9 -> 0.4 선형 감소 (Shi & Eberhart 1998)
  - c1 = c2 = 2.0
  - 입자 수 30, 최대 반복 100
  - 속도 제한: 차원별 v_max = k * (x_max - x_min), k = 0.2
  - 경계 처리: clamp
  - 정수 변수: 평가 시 반올림 (int_dims로 지정)

구조 설계 의도:
  - objective(X) 는 (n_particles, n_dims) 배열을 받아 (n_particles,) fitness 반환
    -> 나중에 조류계산 기반 목적함수를 그대로 끼워넣을 수 있게 벡터화 인터페이스 유지
  - 최소화 문제로 통일 (ESS 문제는 -순편익 + 페널티 형태로 넣으면 됨)
"""

import numpy as np


class PSO:
    def __init__(
        self,
        objective,          # callable: (n_particles, n_dims) -> (n_particles,)
        bounds,             # (n_dims, 2) array: [[lo, hi], ...]
        n_particles=30,
        n_iters=100,
        w_max=0.9,
        w_min=0.4,
        c1=2.0,
        c2=2.0,
        v_clamp_k=0.2,      # v_max = k * (hi - lo)
        int_dims=None,      # 정수로 반올림할 차원 인덱스 리스트 (예: 버스 번호)
        seed=None,
    ):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_dims = self.bounds.shape[0]
        self.n_particles = n_particles
        self.n_iters = n_iters
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2
        self.int_dims = list(int_dims) if int_dims else []
        self.rng = np.random.default_rng(seed)

        self.lo = self.bounds[:, 0]
        self.hi = self.bounds[:, 1]
        # 차원별 속도 제한 (스케일이 다른 변수들 때문에 필수)
        self.v_max = v_clamp_k * (self.hi - self.lo)

    def _decode(self, X):
        """평가 직전 정수 차원 반올림 (버스 번호 등)."""
        if not self.int_dims:
            return X
        Xd = X.copy()
        Xd[:, self.int_dims] = np.round(Xd[:, self.int_dims])
        return Xd

    def optimize(self, verbose=False):
        n, d = self.n_particles, self.n_dims

        # --- 초기화 ---
        X = self.rng.uniform(self.lo, self.hi, size=(n, d))
        V = self.rng.uniform(-self.v_max, self.v_max, size=(n, d))

        fit = self.objective(self._decode(X))
        pbest_X = X.copy()
        pbest_f = fit.copy()

        g_idx = np.argmin(pbest_f)
        gbest_X = pbest_X[g_idx].copy()
        gbest_f = pbest_f[g_idx]

        history = [gbest_f]

        # --- 반복 ---
        for t in range(self.n_iters):
            # 관성 가중치 선형 감소
            w = self.w_max - (self.w_max - self.w_min) * (t / max(self.n_iters - 1, 1))

            r1 = self.rng.random((n, d))
            r2 = self.rng.random((n, d))

            V = (w * V
                 + self.c1 * r1 * (pbest_X - X)
                 + self.c2 * r2 * (gbest_X - X))

            # 속도 제한
            V = np.clip(V, -self.v_max, self.v_max)

            # 위치 갱신 + 경계 clamp
            X = np.clip(X + V, self.lo, self.hi)

            # 평가
            fit = self.objective(self._decode(X))

            # pbest 갱신 (매 반복마다)
            improved = fit < pbest_f
            pbest_X[improved] = X[improved]
            pbest_f[improved] = fit[improved]

            # gbest 갱신
            g_idx = np.argmin(pbest_f)
            if pbest_f[g_idx] < gbest_f:
                gbest_f = pbest_f[g_idx]
                gbest_X = pbest_X[g_idx].copy()

            history.append(gbest_f)

            if verbose and (t + 1) % 20 == 0:
                print(f"  iter {t+1:3d}  w={w:.3f}  gbest={gbest_f:.6e}")

        return {
            "x": self._decode(gbest_X[None, :])[0],
            "f": gbest_f,
            "history": np.array(history),
        }


# ---------------------------------------------------------------
# 표준 벤치마크 함수 (전역최적 = 0, x* = 0 벡터)
# ---------------------------------------------------------------

def sphere(X):
    """단봉·볼록. 기본 수렴 확인용. f(0)=0"""
    return np.sum(X ** 2, axis=1)


def rastrigin(X):
    """다봉. 국소최적 다수 -> 탈출 능력 확인. f(0)=0"""
    A = 10.0
    n = X.shape[1]
    return A * n + np.sum(X ** 2 - A * np.cos(2 * np.pi * X), axis=1)


def rosenbrock(X):
    """좁은 골짜기. 수렴 정밀도 확인. f(1,...,1)=0"""
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2
                  + (1.0 - X[:, :-1]) ** 2, axis=1)
