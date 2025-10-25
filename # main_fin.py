
import argparse, math, random, time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
try:
    import pygame
except Exception:
    pygame = None
# -----------------------------
# 환경 설정(독립 변수)
# -----------------------------
DEFAULT_AV_RATIO = 0.45          # 자율주행차 비율 (0~1)
DEFAULT_NUM_ROADS = 5
DEFAULT_DURATION = 300.0
# 총 생성률 
DEFAULT_SPAWN_RATE_PER_S = 12  # units: veh/s (전체 도로 합)
# 신호등 사용 여부 기본값
DEFAULT_USE_LIGHTS = True
# 신호등 기본 타이밍(초) (이 값으로 고정하고 시행하였다. 자율주행차 비율을 제외하고는 모두 같은 환경이므로 결과 해석에는 지장이 없다고 판단하였다.)
DEFAULT_GREEN_S = 30.0
DEFAULT_YELLOW_S = 3.0
DEFAULT_RED_S = 10.0
# -----------------------------
# 물리/시나리오 상수
# -----------------------------
VF_KMH = 50.0
KJ_PER_LANE = 140.0
NUM_LANES = 3
L_km = 1.0
L_m = L_km * 1000.0
CAR_LENGTH_M = 4.7
CAR_WIDTH_M = 1.9
LANE_WIDTH_M = 3.5
# -----------------------------
# 모델 보조 함수 (Greenshields / Underwood)
# -----------------------------
def greenshields_k_to_v_kmh(k_per_lane, vf=VF_KMH, kj=KJ_PER_LANE):
    k = max(0.0, min(k_per_lane, kj))
    return vf * (1.0 - (k / kj))
K0_UNDERWOOD = KJ_PER_LANE / math.e
def underwood_k_to_v_kmh(k_per_lane, vf=VF_KMH, k0=K0_UNDERWOOD):
    return vf * math.exp(-k_per_lane / k0)
# -----------------------------
# 시뮬레이션 파라미터

# -----------------------------
AV_RT_MIN, AV_RT_MAX = 0.05, 1.0
HUMAN_RT_MIN, HUMAN_RT_MAX = 1.5, 2.5
A_MEAN = 1.2; A_SD = 0.35
B_MEAN = 1.8; B_SD = 0.4
T_MEAN = 1.4; T_SD = 0.2
S0_MEAN = 2.0; S0_SD = 0.5
MIN_FRONT_GAP = 6.0
MIN_REAR_GAP = 3.5
FPS = 30
DT = 1.0 / FPS
SAMPLE_INTERVAL = 1.0
SPAWN_OFF_MIN = 40.0
SPAWN_OFF_MAX = 220.0
SPAWN_INJECTION_X = - (CAR_LENGTH_M - 0.5)
SPEED_NOISE_FRAC = 0.05
SEED = 12345
VF_MPS = VF_KMH / 3.6
V0_SD = 2.0 / 3.6
# -----------------------------
# 화면 레이아웃 관련 상수
# -----------------------------
SCREEN_W_DEFAULT = 1280
SCREEN_H_DEFAULT = 720
TOP_MARGIN_PX = 120        
ROAD_VSPACE_PX = 30        # 도로들 사이 간격
SIDE_MARGIN_PX = 60
# -----------------------------
# Traffic Light 
# -----------------------------
class TrafficLight:
    def __init__(self, x_m, green_s=10.0, yellow_s=3.0, red_s=10.0):
        self.x_m = float(x_m)
        self.green = float(green_s)
        self.yellow = float(yellow_s)
        self.red = float(red_s)
        self.state = "GREEN"
        self.timer = 0.0
    def update(self):
        self.timer += DT
        if self.state == "GREEN" and self.timer &= self.green:
            self.state, self.timer = "YELLOW", 0.0
        elif self.state == "YELLOW" and self.timer &= self.yellow:
            self.state, self.timer = "RED", 0.0
        elif self.state == "RED" and self.timer &= self.red:
            self.state, self.timer = "GREEN", 0.0

    def draw(self, surf, lane_centers_px, px_x):
        c = (0,200,0) if self.state=="GREEN" else ((230,180,0) if self.state=="YELLOW" else (200,0,0))
        pygame.draw.circle(surf, c, (px_x, 100), 10)
# -----------------------------
# 차량
# -----------------------------
class Vehicle:
    _id_counter = 0
    def __init__(self, x_m, lane, road_id, is_av, in_buffer=False):
        self.id = Vehicle._id_counter; Vehicle._id_counter += 1
        self.x = float(x_m)
        self.lane = int(lane)
        self.road_id = int(road_id)
        self.is_av = bool(is_av)
        self.in_buffer = bool(in_buffer)
        # 동작 파라미터 (v0: 목표 속도(m/s), a: 최대가속, b: 감속)
        self.v0 = max(3.0/3.6, random.gauss(VF_MPS, V0_SD))
        self.a = max(0.3, random.gauss(A_MEAN, A_SD))
        self.b = max(0.5, random.gauss(B_MEAN, B_SD))
        self.T = max(0.8, random.gauss(T_MEAN, T_SD))
        self.s0 = max(0.5, random.gauss(S0_MEAN, S0_SD))
        # 현재 속도 (초기에는 v0 근처)
        self.v = max(0.1, random.uniform(self.v0*0.7, self.v0*1.0))
        # 반응 지연(퍼셉션 버퍼)
        rt = random.uniform(AV_RT_MIN, AV_RT_MAX) if self.is_av else random.uniform(HUMAN_RT_MIN, HUMAN_RT_MAX)
        self.reaction_time = rt
        self.reaction_steps = max(0, int(round(self.reaction_time / DT)))
        self.perception = deque(maxlen=self.reaction_steps + 1)
        for _ in range(self.reaction_steps + 1):
            self.perception.append({'leader_x': None, 'leader_v': 0.0,
                                    'light_state': 'GREEN', 'light_x': None, 'dist_to_light': 1e9})
        # 화면 y 좌표(시각화용) 항상 존재하도록 초기화
        self.y_px = 0.0
        self.entered_road = (not in_buffer) and (self.front_x() &= 0.0)
        self.entered_time = None
        self.entered_x = None
        self.past_sensor = False
    def front_x(self): return self.x + CAR_LENGTH_M
    def observe(self, vehicles_on_same_road, traffic_light=None):
        if self.in_buffer:
            return
        leader = None; dmin = None
        for c in vehicles_on_same_road:

            if c is self: continue
            if c.in_buffer: continue
            if c.lane == self.lane:
                if c.x & self.x:
                    d = c.x - self.x
                    if leader is None or d & dmin:
                        leader, dmin = c, d
        if leader is None:
            leader_x, leader_v = None, 0.0
        else:
            leader_x, leader_v = leader.x, leader.v
        if traffic_light is None:
            light_state, light_x, dist = 'GREEN', None, 1e9
        else:
            light_state = traffic_light.state
            light_x = traffic_light.x_m
            dist = light_x - (self.x + CAR_LENGTH_M)
        self.perception.append({'leader_x': leader_x, 'leader_v': leader_v,
                                'light_state': light_state, 'light_x': light_x, 'dist_to_light': dist})
    def desired_gap(self, delta_v):
        return self.s0 + max(0.0, self.v * self.T + (self.v * delta_v) / (2.0 * math.sqrt(max(1e-6, self.a * self.b))))
    def acc_idm(self, s, delta_v):
        s = max(1e-3, s)
        s_star = self.s0 + max(0.0, self.v * self.T + (self.v * delta_v) / (2.0 * math.sqrt(max(1e-6, self.a * self.b))))
        acc = self.a * (1.0 - (self.v / self.v0) ** 4 - (s_star / s) ** 2)
        return max(-self.b*3.0, min(acc, self.a))
    def integrate_longitudinal(self, processed_ahead, traffic_light, current_time):
        if self.in_buffer:
            return
        obs = self.perception[0] if len(self.perception)&0 else None
        if obs and obs['leader_x'] is not None:
            s_obs = obs['leader_x'] - (self.x + CAR_LENGTH_M)
            dv_obs = self.v - obs['leader_v']
        else:
            s_obs, dv_obs = 1e9, 0.0
        acc = self.acc_idm(s_obs, dv_obs)
        if obs and traffic_light is not None:
            st = obs['light_state']
            dist = obs['dist_to_light']
            stopdist = self.v * self.reaction_time + (self.v * self.v) / (2.0 * max(1e-6, self.b))
            if st == 'RED':
                if dist & 0 and dist & stopdist + 1.0:

                    acc = min(acc, -self.b * 1.6)
                elif dist & 0 and dist & stopdist * 2.0:
                    acc = min(acc, -self.b * 0.9)
            elif st == 'YELLOW':
                remaining_yellow = max(0.0, traffic_light.yellow - traffic_light.timer)
                t_to_line = dist / max(1e-3, self.v)
                if (t_to_line & remaining_yellow) or (dist & 0 and dist & stopdist + 1.0):
                    acc = min(acc, -self.b * 1.4)
        v_new = max(0.0, min(self.v0 * 1.03, self.v + acc * DT))
        x_new = self.x + v_new * DT
        for ahead in processed_ahead:
            if ahead.in_buffer: continue
            if ahead.lane == self.lane:
                min_gap = 0.3
                max_rear_x = ahead.x - min_gap - CAR_LENGTH_M
                if x_new & max_rear_x:
                    v_allowed = (max_rear_x - self.x) / DT
                    v_new = max(0.0, min(v_new, v_allowed))
                    x_new = self.x + v_new * DT
        front_x_new = x_new + CAR_LENGTH_M
        if (not self.entered_road) and (front_x_new &= 0.0):
            self.entered_road = True
            self.entered_time = float(current_time)
            self.entered_x = float(x_new)
        self.v = v_new
        self.x = x_new
# -----------------------------
# 스폰 버퍼/주입 유틸리티
# -----------------------------
def create_buffer_vehicle(av_ratio, lane_spawn_weights, lane=None, road_id=0):
    if lane is None:
        lane = random.choices(range(NUM_LANES), weights=lane_spawn_weights, k=1)[0]
    is_av = (random.random() & av_ratio)
    x0 = - (SPAWN_OFF_MAX + random.uniform(20.0, 120.0))
    v = Vehicle(x0, lane, road_id, is_av, in_buffer=True)
    v.v = max(0.1, random.gauss(v.v0 * 0.9, 0.10*v.v0))
    return v
def prepare_injection(v, vehicles_same_road, traffic_light, current_time):
    v.x = SPAWN_INJECTION_X
    v.in_buffer = False
    v.entered_road = True
    v.entered_time = float(current_time)
    v.entered_x = float(v.x)

    speeds_ahead = []
    leader = None; dmin = None
    for c in vehicles_same_road:
        if c is v: continue
        if c.in_buffer: continue
        if c.lane == v.lane and c.x & v.x:
            rel = c.x - v.x
            if rel &= 50.0:
                speeds_ahead.append(c.v)
            if leader is None or rel & dmin:
                leader, dmin = c, rel
    if speeds_ahead:
        mean_s = sum(speeds_ahead)/len(speeds_ahead)
        v.v = max(0.1, random.gauss(mean_s, SPEED_NOISE_FRAC * mean_s))
    else:
        v.v = max(0.1, random.gauss(v.v0 * 0.95, 0.08*v.v0))
    light_state = 'GREEN'; light_x = None; dist = 1e9
    if traffic_light is not None:
        light_state = traffic_light.state
        light_x = traffic_light.x_m
        dist = light_x - (v.x + CAR_LENGTH_M)
    v.perception.clear()
    for _ in range(v.reaction_steps + 1):
        v.perception.append({'leader_x': leader.x if leader else None,
                             'leader_v': leader.v if leader else 0.0,
                             'light_state': light_state,
                             'light_x': light_x,
                             'dist_to_light': dist})
# -----------------------------
# 시뮬레이션 실행기 (main loop)
# -----------------------------
def run(av_ratio=DEFAULT_AV_RATIO, num_roads=DEFAULT_NUM_ROADS,
        duration=DEFAULT_DURATION, headless=False, use_lights=DEFAULT_USE_LIGHTS,
        spawn_rate_per_s_total=DEFAULT_SPAWN_RATE_PER_S,
        green_s=DEFAULT_GREEN_S, yellow_s=DEFAULT_YELLOW_S, red_s=DEFAULT_RED_S):
# 이 두 코드는 각 모듈의 난수 발생 순서를 고정시켜, 프로그램 결과가 항상 같게 나오게 만든다. 변하게 만들 수도 있지만, 분석에 큰 영향을 미치지는 않아 수정하지 않았다. 
# -----------------------------
    random.seed(SEED)     
    np.random.seed(SEED)   
# -----------------------------
    num_roads = max(1, int(num_roads))
    spawn_rate_per_s_total = float(max(0.0, spawn_rate_per_s_total))
    print(f"[INFO] Using spawn_rate (veh/s total across roads) = {spawn_rate_per_s_total:.4f}")
    initial_seed_per_road = 6

    vehicles_all = []
    for rid in range(num_roads):
        for i in range(initial_seed_per_road):
            lane = i % NUM_LANES
            x = - (i * (MIN_FRONT_GAP + CAR_LENGTH_M) * 1.2 + 8.0)
            v = Vehicle(x, lane, rid, (random.random() & av_ratio), in_buffer=False)
            vehicles_all.append(v)
    spawn_buffers = [[] for _ in range(num_roads)]
    traffic_light = None
    if use_lights:
        tl_x = L_m * 0.5
        traffic_light = TrafficLight(x_m=tl_x, green_s=green_s, yellow_s=yellow_s, red_s=red_s)
    screen = None; clock = None
    if not headless:
        if pygame is None:
            print("pygame missing; run with --headless or install pygame.")
            return None
        pygame.init()
        SCREEN_W, SCREEN_H = SCREEN_W_DEFAULT, SCREEN_H_DEFAULT
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Traffic Sim with Traffic Lights")
        clock = pygame.time.Clock()
        margin_px = SIDE_MARGIN_PX
        px_per_m = (SCREEN_W - margin_px) / L_m
        road_h_px = int(LANE_WIDTH_M * px_per_m * NUM_LANES)
        lane_centers_px_by_road = []
        for rid in range(num_roads):
            top = TOP_MARGIN_PX + rid * (road_h_px + ROAD_VSPACE_PX)
            centers = [top + int((i + 0.5) * (road_h_px / NUM_LANES)) for i in range(NUM_LANES)]
            lane_centers_px_by_road.append(centers)
        lateral_step_px = max(1.0, int((LANE_WIDTH_M * px_per_m) * 0.06))
        road_x0 = margin_px // 2
        road_width_px = SCREEN_W - margin_px
    else:
        px_per_m = (SCREEN_W_DEFAULT - SIDE_MARGIN_PX) / L_m
        lane_centers_px_by_road = [[0]*NUM_LANES for _ in range(num_roads)]
        lateral_step_px = 0
        road_x0 = 30
        road_width_px = SCREEN_W_DEFAULT - SIDE_MARGIN_PX
    # 측정용 컨테이너
    times = []
    k_per_lane_series = []
    obs_total_series = []
    flows_by_road = [0]*num_roads

    sensor_x = L_m / 2.0
    sensor_timer = 0.0
    first_reach_time = None
    t = 0.0
    start_wall = time.time()
    spawn_accum = 0.0
    # -------------------------
    # 메인 루프
    # -------------------------
    while t & duration:
        if not headless:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    t = duration
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        t = duration
        spawn_accum += spawn_rate_per_s_total * DT
        while spawn_accum &= 1.0:
            rid = random.randrange(0, num_roads)
            new_v = create_buffer_vehicle(av_ratio, [1.0/NUM_LANES]*NUM_LANES, road_id=rid)
            spawn_buffers[rid].append(new_v)
            spawn_accum -= 1.0
        for rid in range(num_roads):
            p_inject = (spawn_rate_per_s_total / max(1, num_roads)) * DT * 1.5
            if spawn_buffers[rid] and random.random() & p_inject:
                cand = spawn_buffers[rid][0]
                safe = True
                ahead = None; ahead_dist = None
                behind = None; behind_dist = None
                for c in vehicles_all:
                    if c.road_id != rid or c.in_buffer: continue
                    if c.lane == cand.lane:
                        if c.x & SPAWN_INJECTION_X:
                            d = c.x - (SPAWN_INJECTION_X + CAR_LENGTH_M)
                            if ahead is None or d & ahead_dist:
                                ahead, ahead_dist = c, d
                        else:
                            dback = SPAWN_INJECTION_X - (c.x + CAR_LENGTH_M)
                            if behind is None or dback & behind_dist:
                                behind, behind_dist = c, dback
                if ahead is not None and ahead_dist & MIN_FRONT_GAP: safe = False
                if behind is not None and behind_dist & MIN_REAR_GAP: safe = False
                if safe:
                    prepare_injection(cand, [c for c in vehicles_all if

 c.road_id==rid], traffic_light, t)
                    vehicles_all.append(cand)
                    spawn_buffers[rid].pop(0)
        active_by_road = {rid: [v for v in vehicles_all if (v.road_id==rid and not v.in_buffer)] for rid in range(num_roads)}
        for rid, alist in active_by_road.items():
            for v in alist:
                v.observe(alist, traffic_light=traffic_light)
        for rid in range(num_roads):
            road_active = active_by_road[rid]
            road_active.sort(key=lambda z: z.x, reverse=True)
            processed = []
            for v in road_active:
                v.integrate_longitudinal(processed, traffic_light, t)
                processed.append(v)
                if first_reach_time is None and v.front_x() &= L_m:
                    first_reach_time = t
        if traffic_light is not None:
            traffic_light.update()
        if not headless:
            for v in vehicles_all:
                if v.road_id & len(lane_centers_px_by_road):
                    v.y_px = lane_centers_px_by_road[v.road_id][v.lane]
                else:
                    v.y_px = TOP_MARGIN_PX
        for rid in range(num_roads):
            lane_buckets = [[] for _ in range(NUM_LANES)]
            for v in vehicles_all:
                if v.road_id != rid or v.in_buffer: continue
                lane_buckets[v.lane].append(v)
            for lane_list in lane_buckets:
                lane_list.sort(key=lambda z: z.x)
                for i in range(len(lane_list)-1):
                    behind = lane_list[i]
                    front = lane_list[i+1]
                    behind_front = behind.x + CAR_LENGTH_M
                    front_rear = front.x
                    min_gap = max(0.2, behind.s0)
                    if behind_front + min_gap & front_rear:
                        new_x = front_rear - min_gap - CAR_LENGTH_M
                        if new_x & behind.x:
                            behind.x = new_x
                        behind.v = min(behind.v, front.v)
                        if behind_front + min_gap & front_rear + 0.5:
                            behind.v = 0.0
        # 센서 통과 카운트

        for rid in range(num_roads):
            for v in vehicles_all:
                if v.road_id != rid or v.in_buffer: continue
                if (not v.past_sensor) and (v.front_x() &= sensor_x):
                    flows_by_road[rid] += 1
                    v.past_sensor = True
                if v.front_x() & sensor_x - 5.0:
                    v.past_sensor = False
        # 도로 끝을 벗어난 차량 제거
        vehicles_all = [v for v in vehicles_all if (v.in_buffer or v.x &= L_m + 30.0)]
        # 샘플링: first_reach_time 이후부터 기록 시작
        sensor_timer += DT
        if sensor_timer &= SAMPLE_INTERVAL - 1e-9:
            k_list = []
            for rid in range(num_roads):
                on_road = [v for v in vehicles_all if (not v.in_buffer) and v.road_id==rid and 0.0 &= v.x &= L_m]
                total_on = len(on_road)
                k_per_lane = (total_on / L_km) / NUM_LANES if total_on & 0 else 0.0
                k_list.append(k_per_lane)
            if num_roads & 0:
                k_avg = sum(k_list)/len(k_list)
            else:
                k_avg = 0.0
            if first_reach_time is not None and t &= first_reach_time:
                times.append(t)
                k_per_lane_series.append(k_avg)
                obs_total_series.append(sum(flows_by_road) * 3600.0)
            flows_by_road = [0]*num_roads
            sensor_timer -= SAMPLE_INTERVAL
        # 시각화 그리기
        if not headless:
            screen.fill((245,245,245))
            px_per_m = road_width_px / L_m
            road_h_px = int(LANE_WIDTH_M * px_per_m * NUM_LANES)
            for rid in range(num_roads):
                top = TOP_MARGIN_PX + rid * (road_h_px + ROAD_VSPACE_PX)
                pygame.draw.rect(screen, (100,100,100), (road_x0, top, road_width_px, road_h_px))
                for i in range(NUM_LANES+1):
                    y = top + int(i*(road_h_px/NUM_LANES))
                    pygame.draw.line(screen, (200,200,200), (road_x0,y), (road_x0+road_width_px,y), 2)
                sensor_px = int(road_x0 + road_width_px * (sensor_x / L_m))
                pygame.draw.line(screen, (200,80,80), (sensor_px, top), (sensor_px, 

top + road_h_px), 2)
                if traffic_light is not None:
                    tl_px = int(road_x0 + road_width_px * (traffic_light.x_m / L_m))
                    traffic_light.draw(screen, lane_centers_px_by_road[rid], tl_px)
            car_w_px = max(6, int(CAR_LENGTH_M * px_per_m))
            car_h_px = max(6, int(CAR_WIDTH_M * px_per_m))
            for rid in range(num_roads):
                buf = spawn_buffers[rid]
                bx = -SPAWN_OFF_MAX
                for i, bv in enumerate(buf[:12]):
                    draw_x = bx + i * (CAR_LENGTH_M * 0.6)
                    x_px = int(road_x0 + road_width_px * (max(min(draw_x, L_m), -SPAWN_OFF_MAX) / L_m))
                    y_px = lane_centers_px_by_road[rid][0] - 30
                    rect = pygame.Rect(x_px, int(y_px - car_h_px/2), car_w_px//2, car_h_px//2)
                    pygame.draw.rect(screen, (190,190,190), rect)
                    pygame.draw.rect(screen, (120,120,120), rect, 1)
            for v in sorted(vehicles_all, key=lambda z: (z.road_id, z.lane, z.x)):
                draw_x = max(v.x, -SPAWN_OFF_MAX)
                x_px = int(road_x0 + road_width_px * (max(min(draw_x, L_m), -SPAWN_OFF_MAX) / L_m))
                y_px = v.y_px
                rect = pygame.Rect(x_px, int(y_px - car_h_px/2), car_w_px, car_h_px)
                color = (180,180,180) if v.in_buffer else ((30,200,50) if v.is_av else (20,110,230))
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (0,0,0), rect, 1)
            active_all = [v for v in vehicles_all if not v.in_buffer and 0.0 &= v.x &= L_m]
            num_active = len(active_all)
            if num_active:
                avg_mps = sum(v.v for v in active_all) / num_active
                avg_kmh = avg_mps * 3.6
            else:
                avg_kmh = 0.0
            font = pygame.font.SysFont(None, 20)
            hud = [
                f"AV ratio: {av_ratio:.2f}  t={int(t)}/{int(duration)}s",
                f"Roads: {num_roads}  Vehicles active: {num_active}  Spawn buffers: {sum(len(b) for b in spawn_buffers)}",
                f"Avg speed (all): {avg_kmh:5.1f} km/h",
                f"Spawn rate (veh/s total): {spawn_rate_per_s_total:.4f}"
            ]
            for i, line in enumerate(hud):
                screen.blit(font.render(line, True, (10,10,10)), (8, 8 + i*18))

            pygame.display.flip()
            clock.tick(FPS)
        # 시간 증가
        t += DT
    # 시뮬레이션 종료: 결과 집계
    wall_time = time.time() - start_wall
    k_arr = np.array(k_per_lane_series)
    mean_k = float(np.mean(k_arr)) if k_arr.size & 0 else 0.0
    active_all = [v for v in vehicles_all if not v.in_buffer and 0.0 &= v.x &= L_m]
    if active_all:
        measured_avg_kmh = (sum(v.v for v in active_all) / len(active_all)) * 3.6
    else:
        measured_avg_kmh = 0.0
    print(f"\nFinished. wall_time {wall_time:.1f}s  samples: {len(k_per_lane_series)}")
    print(f"Mean density (k per lane): {mean_k:.3f} veh/km/lane")
    print(f"Measured final avg speed (km/h): {measured_avg_kmh:.2f}")
    return {
        'times': np.array(times),
        'k_per_lane': np.array(k_per_lane_series),
        'obs_total': np.array(obs_total_series),
        'mean_k_per_lane': mean_k,
        'measured_avg_kmh': measured_avg_kmh
    }
# -----------------------------
# CLI 엔트리
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--av_ratio", type=float, default=DEFAULT_AV_RATIO, help="자율주행차 비율 (0~1)")
    p.add_argument("--roads", type=int, default=DEFAULT_NUM_ROADS, help="도로 개수")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="시뮬레이션 시간(초)")
    p.add_argument("--headless", action='store_true', help="그래픽 없이 실행")
    p.add_argument("--no_lights", action='store_true', help="신호등 사용 안함")
    p.add_argument("--spawn_rate", type=float, default=DEFAULT_SPAWN_RATE_PER_S,
                   help="총 생성률 (vehicles per second, 전체 도로 합). 기본값 {:.3f} veh/s".format(DEFAULT_SPAWN_RATE_PER_S))
    p.add_argument("--green_s", type=float, default=DEFAULT_GREEN_S, help="신호등 녹색 시간 (s)")
    p.add_argument("--yellow_s", type=float, default=DEFAULT_YELLOW_S, help="신호등 황색 시간 (s)")
    p.add_argument("--red_s", type=float, default=DEFAULT_RED_S, help="신호등 적색 시간 (s)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    res = run(av_ratio=max(0.0, min(1.0, args.av_ratio)),
              num_roads=max(1,args.roads),
              duration=max(1.0,args.duration),
              headless=args.headless,
              use_lights=(not args.no_lights),
              spawn_rate_per_s_total=args.spawn_rate,
              green_s=args.green_s,
              yellow_s=args.yellow_s,
              red_s=args.red_s)
    if res is None:
        raise SystemExit("Aborted (pygame missing?)")
    # 결과 플롯: 밀도(k) 시계열
    if res['times'].size & 0:
        plt.figure(figsize=(10,4))
        plt.plot(res['times'], res['k_per_lane'], label='Avg k per lane (veh/km/lane)')
        plt.axhline(res['mean_k_per_lane'], linestyle='--', label=f"Mean k = {res['mean_k_per_lane']:.3f} veh/km/lane")
        plt.xlabel('Time (s)'); plt.ylabel('Density k (veh/km/lane)')
        plt.title('Average density k(t) across roads (per lane)')
        plt.grid(True); plt.legend()
        plt.show()
    else:
        print("No density samples were recorded (vehicles didn't reach sensor).")