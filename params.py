# 숫자는 전부 여기 (CLAUDE.md 부록 B 기준)

VN_KV = 22.9
K_SCALE = 2.288670

# 선로 정격 (KS C 3113, 40°C). CLAUDE.md 1절.
# {(from_bus, to_bus): max_i_ka}
LINE_RATINGS_A = {
    (0, 1): 410, (1, 2): 410,                      # ACSR-OC 160mm^2
    (2, 3): 296, (3, 4): 296, (4, 5): 296,          # ACSR-OC 95mm^2
    # 나머지 5~31 (27개): ACSR-OC 58mm^2 222A (build_net에서 default 처리)
}
LINE_RATING_DEFAULT_A = 222
