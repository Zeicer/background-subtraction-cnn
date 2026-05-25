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
    # 🚶 3. 徘徊判定：頭腳高度比例 + 頭部點位移固定範圍內
    # =========================================================================
    def check_loitering(self, pid, range_threshold=30, min_frames=150):
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

    # =========================================================================
    # 💥 4. 碰撞判定：人物框重疊 + 鼻點偏離原路徑過大
    # =========================================================================
    def check_collision(self, pid_a, pid_b, dev_ratio_threshold=0.25):
        track_a = self.person_tracks.get(pid_a)
        track_b = self.person_tracks.get(pid_b)

        if not track_a or not track_b or len(track_a) < 11 or len(track_b) < 11:
            return False

        # person_tracks 存的是中心點格式：(cx, cy, w, h, kpts)
        acx, acy, aw, ah = track_a[-1][0:4]
        bcx, bcy, bw, bh = track_b[-1][0:4]

        box_a = (
            acx - aw / 2,
            acy - ah / 2,
            acx + aw / 2,
            acy + ah / 2,
        )
        box_b = (
            bcx - bw / 2,
            bcy - bh / 2,
            bcx + bw / 2,
            bcy + bh / 2,
        )

        is_overlapping = not (
            box_a[2] < box_b[0]
            or box_a[0] > box_b[2]
            or box_a[3] < box_b[1]
            or box_a[1] > box_b[3]
        )

        if not is_overlapping:
            return False

        head_a_now, foot_a = self._get_head_foot_data(track_a[-1][4])
        head_b_now, foot_b = self._get_head_foot_data(track_b[-1][4])

        head_a_past, _ = self._get_head_foot_data(track_a[-6][4])
        head_b_past, _ = self._get_head_foot_data(track_b[-6][4])

        head_a_old, _ = self._get_head_foot_data(track_a[-11][4])
        head_b_old, _ = self._get_head_foot_data(track_b[-11][4])

        if not (
            head_a_now
            and head_b_now
            and head_a_past
            and head_b_past
            and head_a_old
            and head_b_old
            and foot_a
            and foot_b
        ):
            return False

        hf_a = np.hypot(head_a_now[0] - foot_a[0], head_a_now[1] - foot_a[1])
        hf_b = np.hypot(head_b_now[0] - foot_b[0], head_b_now[1] - foot_b[1])

        if hf_a <= 1 or hf_b <= 1:
            return False

        v_orig_a_x = head_a_past[0] - head_a_old[0]
        v_orig_a_y = head_a_past[1] - head_a_old[1]

        expected_a_x = head_a_past[0] + v_orig_a_x
        expected_a_y = head_a_past[1] + v_orig_a_y

        dev_a = np.hypot(
            head_a_now[0] - expected_a_x,
            head_a_now[1] - expected_a_y,
        )

        v_orig_b_x = head_b_past[0] - head_b_old[0]
        v_orig_b_y = head_b_past[1] - head_b_old[1]

        expected_b_x = head_b_past[0] + v_orig_b_x
        expected_b_y = head_b_past[1] + v_orig_b_y

        dev_b = np.hypot(
            head_b_now[0] - expected_b_x,
            head_b_now[1] - expected_b_y,
        )

        thresh_a = hf_a * dev_ratio_threshold
        thresh_b = hf_b * dev_ratio_threshold

        if dev_a > thresh_a or dev_b > thresh_b:
            return True

        return False

    # 保留別名，避免其他程式有呼叫舊名稱
    def check_collision_by_deviation(self, pid_a, pid_b, dev_ratio_threshold=0.25):
        return self.check_collision(pid_a, pid_b, dev_ratio_threshold)

# =========================================================================
# 🚯 5. 丟垃圾 / 亂丟行為判定：
# 只要新物件在 YOLO 手腕點附近產生，就判定為亂丟行為
# =========================================================================
    def check_littering(
        self,
        oid,
        hand_birth_threshold=60,
        birth_frames=20
    ):
        """
        判斷條件：
        1. 物件是新出現的物件，也就是軌跡長度 <= birth_frames
        2. 物件中心點靠近某人物的 YOLO 左手腕或右手腕
        3. 距離小於 hand_birth_threshold
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

        # 只判斷「新物件」
        # 如果物件已經存在超過 birth_frames 幀，就不再當作剛產生的物件
        if len(obj_track) > birth_frames:
            return False

        # object_tracks 存的是中心點格式：(cx, cy, w, h)
        ocx, ocy, ow, oh = obj_track[-1]

        best_pid = None
        min_hand_dist = hand_birth_threshold

        # 檢查所有人物的 YOLO 手腕點
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
                    hand_dists.append(np.hypot(ocx - lw_x, ocy - lw_y))

                if rw_x > 0 and rw_y > 0:
                    hand_dists.append(np.hypot(ocx - rw_x, ocy - rw_y))

                if not hand_dists:
                    continue

                closer_hand_dist = min(hand_dists)

                if closer_hand_dist < min_hand_dist:
                    min_hand_dist = closer_hand_dist
                    best_pid = pid

            except Exception:
                continue

        # 只要新物件靠近任一人物手腕點，就立刻判定亂丟
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
