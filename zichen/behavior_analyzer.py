import numpy as np
from collections import deque


# 官方命名：PerfectCombinedSystem_KinematicsBehaviorAnalyzer
class BehaviorAnalyzer:
    def __init__(self, max_history=150):
        """
        大腦記憶庫：儲存人員與物品的時序軌跡
        人員記憶格式: {id: deque([(cx, cy, w, h, kpts), ...], maxlen=max_history)}
        物件記憶格式: {id: deque([(cx, cy, w, h), ...], maxlen=60)}
        """
        self.max_history = max_history
        self.person_tracks = {}
        self.object_tracks = {}
        self.littering_timers = {}

        # 亂丟行為記憶：這裡不判斷「是不是垃圾」，只判斷「是否有人丟出物件」
        self.object_owner_memory = {}
        self.body_distance_history = {}
        self.object_moved_away_memory = {}
        self.throwing_event_objects = set()
        self.throwing_event_owner_memory = {}

    def update_person(self, pid, bbox, keypoints=None):
        """
        更新人員軌跡。
        主程式傳入 bbox 格式為 (x, y, w, h)，這裡轉存成中心點格式：
        (cx, cy, w, h, keypoints)
        """
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2

        if pid not in self.person_tracks:
            self.person_tracks[pid] = deque(maxlen=self.max_history)

        self.person_tracks[pid].append((cx, cy, w, h, keypoints))

    def update_object(self, oid, bbox):
        """
        更新物件軌跡。
        主程式傳入 bbox 格式為 (x, y, w, h)，這裡轉存成中心點格式：
        (cx, cy, w, h)
        """
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2

        if oid not in self.object_tracks:
            self.object_tracks[oid] = deque(maxlen=60)

        self.object_tracks[oid].append((cx, cy, w, h))

    def _get_head_foot_data(self, kpts):
        """
        工具函式：安全提取鼻子點與雙腳中心點。
        COCO keypoints:
        0 = nose
        15 = left ankle
        16 = right ankle
        """
        if kpts is None or len(kpts) <= 16:
            return None, None

        try:
            hx, hy = float(kpts[0][0]), float(kpts[0][1])
            la_x, la_y = float(kpts[15][0]), float(kpts[15][1])
            ra_x, ra_y = float(kpts[16][0]), float(kpts[16][1])

            if hx <= 0 or hy <= 0:
                return None, None

            if la_x <= 0 or la_y <= 0 or ra_x <= 0 or ra_y <= 0:
                return None, None

            fx = (la_x + ra_x) / 2
            fy = (la_y + ra_y) / 2

            return (hx, hy), (fx, fy)
        except Exception:
            return None, None

    # =========================================================================
    # 🛌 1. 昏倒判定：5 幀時序平均高度縮減 + 鼻子點快速下墜位移
    # =========================================================================
    def check_fall(self, pid):
        track = self.person_tracks.get(pid)

        if not track or len(track) < 6:
            return False

        kpts_now = track[-1][4]
        head_now, foot_now = self._get_head_foot_data(kpts_now)

        if not head_now or not foot_now:
            return False

        hx, hy = head_now
        fx, fy = foot_now

        current_height = fy - hy

        if current_height <= 0:
            return False

        # =====================================================
        # 1. 計算過去 5 幀平均頭腳垂直高度
        # =====================================================
        past_heights = []

        for i in range(-6, -1):
            kpts_past = track[i][4]
            head_past, foot_past = self._get_head_foot_data(kpts_past)

            if head_past and foot_past:
                past_height = foot_past[1] - head_past[1]

                if past_height > 0:
                    past_heights.append(past_height)

        if len(past_heights) < 3:
            return False

        avg_past_height = np.mean(past_heights)

        if avg_past_height <= 0:
            return False

        height_shrink_ratio = current_height / avg_past_height

        # =====================================================
        # 2. 取第 -6 幀頭點，計算頭部下墜與總位移
        # =====================================================
        kpts_first = track[-6][4]
        head_first, _ = self._get_head_foot_data(kpts_first)

        if not head_first:
            return False

        hx_p, hy_p = head_first

        head_total_movement = np.hypot(
            hx - hx_p,
            hy - hy_p
        )

        head_down_movement = hy - hy_p

        # =====================================================
        # 3. 計算目前身體垂直比例
        #    越接近 1 代表越直立
        #    越小代表身體越斜或倒下
        # =====================================================
        head_foot_distance = np.hypot(
            hx - fx,
            hy - fy
        )

        vertical_ratio = current_height / max(head_foot_distance, 1)

        # =====================================================
        # 4. 使用「身高比例」作為動態門檻
        # =====================================================
        height_collapsed = height_shrink_ratio < 0.65
        head_dropped = head_down_movement > avg_past_height * 0.12
        head_moved_fast = head_total_movement > avg_past_height * 0.10
        body_not_upright = vertical_ratio < 0.75

        fall_candidate = (
            height_collapsed
            and head_dropped
            and head_moved_fast
            and body_not_upright
        )

        # =====================================================
        # 5. 連續確認，避免單幀骨架飄掉誤判
        # =====================================================
        if not hasattr(self, "fall_confirm_counter"):
            self.fall_confirm_counter = {}

        if fall_candidate:
            self.fall_confirm_counter[pid] = self.fall_confirm_counter.get(pid, 0) + 1
        else:
            self.fall_confirm_counter[pid] = 0

        if self.fall_confirm_counter[pid] >= 2:
            return True

        return False
# =========================================================================
    # 🏃 2. 跑步判定：
    # 目前高度接近過去 10 幀平均高度 + 過去 10 段平均位移夠大
    # =========================================================================
    def check_running(self, pid, fps=30, movement_threshold=10):
        track = self.person_tracks.get(pid)

        # 需要至少 11 幀：
        # - 目前幀：track[-1]
        # - 過去 10 幀高度：track[-11:-1]
        # - 最近 10 段位移：track[-11] 到 track[-1] 之間的連續鼻點位移
        if not track or len(track) < 11:
            return False, 0

        # =====================================================
        # 取得目前幀頭點與腳點
        # =====================================================
        kpts_now = track[-1][4]
        head_now, foot_now = self._get_head_foot_data(kpts_now)

        if not head_now or not foot_now:
            return False, 0

        hx, hy = head_now
        fx, fy = foot_now

        current_height = fy - hy

        if current_height <= 0:
            return False, 0

        # =====================================================
        # 條件 1：目前高度 > 過去 10 幀平均高度 × 0.90
        # =====================================================
        past_heights = []

        # 過去 10 幀，不包含目前幀
        for item in list(track)[-11:-1]:
            kpts = item[4]
            head, foot = self._get_head_foot_data(kpts)

            if head and foot:
                vertical_height = foot[1] - head[1]

                if vertical_height > 0:
                    past_heights.append(vertical_height)

        # 過去 10 幀至少要有 5 幀有效高度
        if len(past_heights) < 5:
            return False, 0

        avg_past_height = np.mean(past_heights)

        if avg_past_height <= 0:
            return False, 0

        height_ok = current_height > avg_past_height * 0.90

        # =====================================================
        # 條件 2：最近 10 段鼻點平均位移 > movement_threshold
        # =====================================================
        recent_tracks = list(track)[-11:]

        head_points = []

        for item in recent_tracks:
            kpts = item[4]
            head, _ = self._get_head_foot_data(kpts)

            if head:
                head_points.append(head)

        # 需要至少 6 個有效頭點，才有足夠位移資料
        if len(head_points) < 6:
            return False, 0

        movements = []

        for i in range(1, len(head_points)):
            prev_x, prev_y = head_points[i - 1]
            curr_x, curr_y = head_points[i]

            movement = np.hypot(
                curr_x - prev_x,
                curr_y - prev_y
            )

            movements.append(movement)

        if len(movements) == 0:
            return False, 0

        avg_movement = np.mean(movements)

        movement_ok = avg_movement > movement_threshold

        # 換算成每秒像素速度，這裡是平均每幀位移 × FPS
        pixel_speed_per_second = avg_movement * fps

        # =====================================================
        # 最終判斷
        # =====================================================
        if height_ok and movement_ok:
            return True, pixel_speed_per_second

        return False, pixel_speed_per_second

    # =========================================================================
    # 🚶 3. 徘徊判定：頭腳高度比例 + 頭部點位移固定範圍內
    # =========================================================================
    def check_loitering(self, pid, range_threshold=30, min_frames=120):
        track = self.person_tracks.get(pid)
        if not track or len(track) < min_frames:
            return False

        recent_tracks = list(track)[-min_frames:]

        hx_coords = []
        hy_coords = []
        hf_distances = []

        for item in recent_tracks:
            kpts = item[4]
            head, foot = self._get_head_foot_data(kpts)

            if head:
                hx_coords.append(head[0])
                hy_coords.append(head[1])

            if head and foot:
                dist = np.hypot(head[0] - foot[0], head[1] - foot[1])
                hf_distances.append(dist)

        if len(hx_coords) < min_frames // 2 or len(hf_distances) < min_frames // 2:
            return False

        avg_height = np.mean(hf_distances)

        kpts_now = track[-1][4]
        head_now, foot_now = self._get_head_foot_data(kpts_now)

        if head_now and foot_now and avg_height > 0:
            current_height = np.hypot(head_now[0] - foot_now[0], head_now[1] - foot_now[1])
            height_ratio = current_height / avg_height

            if not (0.80 <= height_ratio <= 1.10):
                return False
        else:
            return False

        x_range = max(hx_coords) - min(hx_coords)
        y_range = max(hy_coords) - min(hy_coords)

        if x_range < range_threshold and y_range < range_threshold:
            return True

        return False

    # ===============   # 💥 4. 碰撞判定：多人同框 + 鼻子距離過近 + 移動方向劇烈改變
    # =========================================================================
    def check_collision(self, pid_a, pid_b, dist_threshold=35):
        track_a = self.person_tracks.get(pid_a)
        track_b = self.person_tracks.get(pid_b)
        if not track_a or not track_b or len(track_a) < 6 or len(track_b) < 6: 
            return False

        # 1. 提取兩人在當前幀與 5 幀前的鼻子數據
        head_a_now, foot_a = self._get_head_foot_data(track_a[-1][-1])
        head_b_now, foot_b = self._get_head_foot_data(track_b[-1][-1])
        head_a_past, _ = self._get_head_foot_data(track_a[-6][-1])
        head_b_past, _ = self._get_head_foot_data(track_b[-6][-1])
        
        if head_a_now and head_b_now and head_a_past and head_b_past and foot_a and foot_b:
            # 條件一：兩人的頭腳均保持一定距離（非躺地狀態）
            hf_a = np.sqrt((head_a_now[0]-foot_a[0])**2 + (head_a_now[1]-foot_a[1])**2)
            hf_b = np.sqrt((head_b_now[0]-foot_b[0])**2 + (head_b_now[1]-foot_b[1])**2)
            if hf_a < 45 or hf_b < 45: return False
            
            # 條件二：兩人的【鼻子空間距離過近】
            current_head_dist = np.sqrt((head_a_now[0] - head_b_now[0])**2 + (head_a_now[1] - head_b_now[1])**2)
            if current_head_dist < dist_threshold:
                
                # 條件三：📐 運動學向量分析（檢查位移方向是否發生劇烈突變）
                # 計算 A 過去 5 幀的移動向量
                va_x = head_a_now[0] - head_a_past[0]
                va_y = head_a_now[1] - head_a_past[1]
                
                # 同步去撈更早之前的歷史向量（10 幀前到 5 幀前），當作「碰撞前的原方向」
                if len(track_a) >= 11:
                    head_a_old, _ = self._get_head_foot_data(track_a[-11][-1])
                    if head_a_old:
                        v_orig_x = head_a_past[0] - head_a_old[0]
                        v_orig_y = head_a_past[1] - head_a_old[1]
                        
                        # 計算原向量與新向量的內積
                        mag_orig = np.sqrt(v_orig_x**2 + v_orig_y**2)
                        mag_new = np.sqrt(va_x**2 + va_y**2)
                        
                        if mag_orig > 2 and mag_new > 1:
                            cos_theta = (v_orig_x * va_x + v_orig_y * va_y) / (mag_orig * mag_new)
                            # 如果 cos_theta < 0.2，代表夾角大於 78 度（包含反彈、急停、或死角折返），視為劇烈碰撞
                            if cos_theta < 0.2:
                                return True
                return True # 若歷史資料不夠長，直接依據同框且鼻子過近觸發保險
        return False


# =========================================================================
    # 🚯 5. 丟垃圾 / 亂丟行為判定：
    # 3 幀 ≤ 新物件出現時間 ≤ 20 幀
    # 且新物件靠近 YOLO 左 / 右手腕點，就判定為亂丟行為
    # =========================================================================
    def check_littering(
        self,
        oid,
        hand_birth_threshold=15,
        min_birth_frames=10,
        max_birth_frames=20
    ):
        """
        判斷條件：
        1. 物件是新出現的物件，但至少要穩定出現 3 幀
        min_birth_frames <= len(obj_track) <= max_birth_frames

        2. 物件中心點靠近某人物的 YOLO 左手腕或右手腕

        3. 最近手腕距離 < hand_birth_threshold

        → 直接判定丟垃圾 / 亂丟行為
        """

        obj_track = self.object_tracks.get(oid)

        if not obj_track:
            return False

        # 初始化記憶資料，避免重複觸發同一個物件
        if not hasattr(self, "throwing_event_objects"):
            self.throwing_event_objects = set()

        if not hasattr(self, "throwing_event_owner_memory"):
            self.throwing_event_owner_memory = {}

        if not hasattr(self, "object_owner_memory"):
            self.object_owner_memory = {}

        # 同一個物件已經觸發過，就不重複判斷
        if oid in self.throwing_event_objects:
            return False

        # =====================================================
        # 條件 1：物件出現時間必須介於 3 幀到 20 幀之間
        # =====================================================
        obj_age = len(obj_track)

        if obj_age < min_birth_frames:
            return False

        if obj_age > max_birth_frames:
            return False

        # object_tracks 存的是中心點格式：(cx, cy, w, h)
        ocx, ocy, ow, oh = obj_track[-1]

        best_pid = None
        min_hand_dist = hand_birth_threshold

        # =====================================================
        # 條件 2：檢查物件是否靠近任一人物左 / 右手腕
        # =====================================================
        for pid, p_track in self.person_tracks.items():
            if not p_track:
                continue

            # person_tracks 存的是：(cx, cy, w, h, keypoints)
            kpts = p_track[-1][4]

            if kpts is None or len(kpts) <= 10:
                continue

            try:
                # COCO keypoints:
                # 9 = left wrist, 10 = right wrist
                lw_x, lw_y = float(kpts[9][0]), float(kpts[9][1])
                rw_x, rw_y = float(kpts[10][0]), float(kpts[10][1])

                hand_dists = []

                if lw_x > 0 and lw_y > 0:
                    hand_dists.append(
                        np.hypot(ocx - lw_x, ocy - lw_y)
                    )

                if rw_x > 0 and rw_y > 0:
                    hand_dists.append(
                        np.hypot(ocx - rw_x, ocy - rw_y)
                    )

                if not hand_dists:
                    continue

                closer_hand_dist = min(hand_dists)

                if closer_hand_dist < min_hand_dist:
                    min_hand_dist = closer_hand_dist
                    best_pid = pid

            except Exception:
                continue

        # =====================================================
        # 條件 3：最近手腕距離 < 15 像素，直接判定亂丟
        # =====================================================
        if best_pid is not None:
            self.object_owner_memory[oid] = best_pid
            self.throwing_event_objects.add(oid)
            self.throwing_event_owner_memory[oid] = best_pid

            return True

        return False


    def clear_dead_tracks(self, active_pids, active_oids):
        self.person_tracks = {k: v for k, v in self.person_tracks.items() if k in active_pids}
        self.object_tracks = {k: v for k, v in self.object_tracks.items() if k in active_oids}
        self.littering_timers = {k: v for k, v in self.littering_timers.items() if k in active_oids}