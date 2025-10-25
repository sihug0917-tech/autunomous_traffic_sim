import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')   # 윈도우 기본 한글 폰트
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')     # 리눅스/Colab 등

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# --- 파라미터 ---
VF_KMH = 50.0
KJ_PER_LANE = 140.0
NUM_LANES = 3

def greenshields_k_to_v_kmh(k_per_lane, vf=VF_KMH, kj=KJ_PER_LANE):
    k = np.clip(k_per_lane, 0.0, kj)
    return vf * (1.0 - (k / kj))

K0_UNDERWOOD = KJ_PER_LANE / np.e
def underwood_k_to_v_kmh(k_per_lane, vf=VF_KMH, k0=K0_UNDERWOOD):
    return vf * np.exp(-k_per_lane / k0)

# --- 입력: 평균 밀도 리스트 (veh/km/lane) ---
# 예: 11개 항목이면 0%,10%,...100% 로 자동 라벨링
mean_k_list = [11.919, 11.807, 13.108, 13.406, 14.731, 15.580, 16.283, 17.578, 18.336, 20.975, 23.365]

k_vals = np.linspace(0.0, KJ_PER_LANE, 400)
vg_vals = greenshields_k_to_v_kmh(k_vals)
vu_vals = underwood_k_to_v_kmh(k_vals)

qg_per_lane = k_vals * vg_vals
qu_per_lane = k_vals * vu_vals

mean_k_arr = np.array(mean_k_list)
mean_v_g = greenshields_k_to_v_kmh(mean_k_arr)
mean_v_u = underwood_k_to_v_kmh(mean_k_arr)

mean_qg_per_lane = mean_k_arr * mean_v_g
mean_qu_per_lane = mean_k_arr * mean_v_u

# --- 퍼센트 라벨 생성 ---
n = len(mean_k_arr)
if n == 11:
    percent_labels = [f"{i*10}%" for i in range(11)]
else:
    percent_vals = np.linspace(0, 100, n)
    percent_labels = [f"{int(round(p))}%" for p in percent_vals]

# --- 플롯 ---
plt.figure(figsize=(9,5))
plt.plot(k_vals, qg_per_lane, label='Greenshields 모델에 의한 교통류', linewidth=1.5)
plt.plot(k_vals, qu_per_lane, label='Underwood 모델에 의한 교통류', linewidth=1.5)

# 점 표시 (두 모델 모두)
plt.scatter(mean_k_arr, mean_qg_per_lane, marker='o', s=80, label='자율주행차 비율별 교통류 (Greenshields)')
plt.scatter(mean_k_arr, mean_qu_per_lane, marker='s', s=80, label='자율주행차 비율별 교통류 (Underwood)')

# 퍼센트 라벨: 인덱스에 따라 위/아래 번갈아 배치
for i, pct in enumerate(percent_labels):
    x = mean_k_arr[i]
    y = mean_qg_per_lane[i]
    # 번갈아 위/아래로 오프셋
    if i % 2 == 0:
        xytext = (6, 6)     # 위쪽 우측
        va = 'bottom'
    else:
        xytext = (6, -12)   # 아래쪽 우측
        va = 'top'
    plt.annotate(pct, (x, y),
                 textcoords="offset points",
                 xytext=xytext,
                 fontsize=9,
                 va=va,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, edgecolor="none"))

plt.xlabel('밀도 k (veh/km/lane)')
plt.ylabel('교통류 q (veh/h per lane)')
plt.title('자율주행차 비율에 따른 교통류 변화')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
