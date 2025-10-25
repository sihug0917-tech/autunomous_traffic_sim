import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')   # 윈도우 기본 한글 폰트
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')     # 리눅스/Colab 등


p=[]
for i in range(11):
    p.append(str(i*10)+'%')
Greenshields_q = [
    545.21,
    540.56,
    594.04,
    606.11,
    659.05,
    692.31,
    719.46,
    768.55,
    796.73,
    891.63,
    973.28
]
Underwood_q = [
    472.83,
    469.41,
    508.13,
    516.68,
    553.33,
    575.65,
    593.47,
    624.76,
    642.18,
    697.92,
    742.19
]
plt.plot(p, Greenshields_q, label='Greenshields 모델에 의한 교통류', marker='o')
plt.plot(p, Underwood_q, label='Underwood 모델에 의한 교통류', marker='s')
plt.xlabel('자율주행차 비율 (%)')
plt.ylabel('교통류 q (veh/h per lane)')
plt.title('자율주행차 비율에 따른 교통류 변화')
plt.grid(True)
plt.legend()   
plt.show()