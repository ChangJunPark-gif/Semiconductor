# 4단계 특징 사전

WM-811K의 die 상태는 `0`=측정 대상 밖, `1`=pass, `2`=fail이다. 아래 비율의 분모에는 상태 `1`·`2`의 유효 die만 넣는다. 영역에 유효 die가 없으면 비율은 결측이다. 반경은 map 중심을 원점으로, 가로·세로의 중심~끝 거리를 각각 1로 정규화한 뒤 유클리드 거리로 계산한다.

| 변수 | 정의 | PCA 입력 |
| :--- | :--- | :---: |
| `yield` | pass die / 유효 die = 1 − `fail_rate` | 아니요 |
| `fail_rate` | fail die / 유효 die | 아니요 |
| `valid_dies`, `fail_dies` | 유효 die·fail die 개수 | 아니요 |
| `center_fail_rate` | 반경 < 0.35 영역의 fail 비율 | 아니요 |
| `edge_fail_rate` | 반경 ≥ 0.75 영역의 fail 비율 | 아니요 |
| `ring_0_fail_rate` | 반경 < 0.25 영역의 fail 비율 | 아니요 |
| `ring_1_fail_rate` | 0.25 ≤ 반경 < 0.50 영역의 fail 비율 | 아니요 |
| `ring_2_fail_rate` | 0.50 ≤ 반경 < 0.75 영역의 fail 비율 | 아니요 |
| `ring_3_fail_rate` | 반경 ≥ 0.75 영역의 fail 비율 | 아니요 |
| `center_enrichment`, `edge_enrichment`, `ring_0_enrichment`~`ring_3_enrichment` | 각 영역 fail 비율 − wafer 전체 `fail_rate` | 예 |
| `quadrant_0_fail_rate`~`quadrant_3_fail_rate` | 위쪽/아래쪽 × 왼쪽/오른쪽 사분면의 fail 비율. 번호는 위왼쪽 0, 위오른쪽 1, 아래왼쪽 2, 아래오른쪽 3 | 아니요 |
| `quadrant_asymmetry` | 유효 die가 있는 사분면 fail 비율의 최댓값 − 최솟값. 2개 미만이면 결측 | 예 |
| `component_count` | 상하좌우로 연결된 fail die 성분 개수 | 아니요 |
| `component_count_per_1000_valid` | `component_count` × 1000 / 유효 die 수 | 예 |
| `largest_component_size` | 최대 fail 연결 성분의 die 수. fail이 없으면 0 | 아니요 |
| `largest_component_fail_share` | 최대 fail 성분 die 수 / fail die 수. fail이 없으면 0 | 예 |
| `moran_i` | 3단계의 4방향 이웃에 대한 전역 Moran's I. 상수 map에서는 결측 | 예 |
| `fail_join_excess_per_edge` | (`fail_fail_edges` − `expected_fail_fail_edges`) / 이웃 edge 수. edge가 없으면 결측 | 예 |

PCA 입력은 표의 `예`인 11개 변수다. 결측은 train 중앙값으로 대체하며 결측 지시자도 입력에 추가한다. 이어서 train 평균·표준편차로 표준화하고 train에서만 PCA 축을 학습한다. `failure_type` 라벨과 `yield`는 PCA 적합에 사용하지 않는다. 3단계 permutation z-score는 전체 wafer가 아닌 표본에만 있으므로 입력에서 제외했다.
