import numpy as np
from collections import deque

# 官方命名：PerfectCombinedSystem_KinematicsBehaviorAnalyzer
class BehaviorAnalyzer:
    def __init__(self, max_history=150):
        """
        大腦記憶庫：儲存人員與物品的時序軌跡
        人員記憶格式: { id: deque([(cx, cy, w, h, kpts), ...], maxlen=150) }
        """
        self.person_tracks = {}
        self.object_tracks = {}
        self.littering_timers = {}

    def update_person(self, pid, bbox, keypoints=None):
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        if pid not in self.person_tracks:
            self.person_tracks[pid] = deque(maxlen=150)
        self.person_tracks[pid].append((cx, cy, w, h, keypoints))

    def update_object(self, oid, bbox):
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        if oid not in self.object_tracks:
            self.object_tracks[oid] = deque(maxlen=60)
        self.object_tracks[oid].append((cx, cy, w, h))

    def _get_head_foot_data(self, kpts):
        """
        工具函式：安全提取鼻子(頭)與雙腳中心點(腳)的座標
        """
        if kpts is None or len(kpts) <= 16:
            return None, None
        try:
            # 鼻子點 (Keypoint 0)
            hx, hy = kpts[0][0], kpts[0][1]
            # 雙腳踝中心點 (Keypoint 15 左腳踝, 16 右腳踝)
            fx = (kpts[15][0] + kpts[16][0]) / 2
            fy = (kpts[15][1] + kpts[16][1]) / 2
            return (hx, hy), (fx, fy)
        except:
            return None, None

    # =========================================================================
    # 🛌 1. 昏倒/跌倒判定：頭腳距離塌陷 + 鼻子點高加速度
    # =========================================================================
    def check_fall(self, pid):
        track = self.person_tracks.get(pid)
        if not track or len(track) < 5: return False
        
        # 1. 拿到當前幀與歷史幀的骨架
        _, _, _, _, kpts_now = track[-1]
        _, _, _, _, kpts_past = track[-4] # 3 幀前
        
        head_now, foot_now = self._get_head_foot_data(kpts_now)
        head_past, _ = self._get_head_foot_data(kpts_past)
        
        if head_now and foot_now and head_past:
            hx, hy = head_now
            fx, fy = foot_now
            hx_p, hy_p = head_past
            
            # A. 幾何特徵：計算當前頭與腳的絕對歐幾里得距離
            hf_dist = np.sqrt((hx - fx)**2 + (hy - fy)**2)
            
            # B. 動態特徵：計算鼻子點的【瞬時位移速度與方向（下墜感）】
            # Y 軸向下為正，hy - hy_p > 0 代表頭部正在向下墜落
            head_v_y = hy - hy_p 
            head_speed = np.sqrt((hx - hx_p)**2 + (hy - hy_p)**2)
            
            # 🌟 自適應動態門檻：利用正常的頭腳距離當基準，如果頭腳距離縮短到原本正常高度的 0.45 倍以下
            # 且此時鼻子點正在快速位移 (下墜)，則判定為昏倒/跌倒
            # 正常人頭到腳長度在 320x240 畫面上大約 80~120 像素，昏倒躺平時會縮短到 35 像素以下
            if hf_dist < 80 and head_speed > 5 and head_v_y > 0:
                return True
                
        # 傳統保險：高寬比扁平化過度嚴重
        if track[-1][3] / max(track[-1][2], 1) < 0.65 and (track[-1][1] - track[-5][1]) > 15:
            return True
        return False

    # =========================================================================
    # 🏃 2. 跑步判定：頭腳保持正常距離 + 鼻子點快速位移
    # =========================================================================
    def check_running(self, pid, fps=30):
        track = self.person_tracks.get(pid)
        if not track or len(track) < 6: return False, 0
        
        _, _, _, _, kpts_now = track[-1]
        _, _, _, _, kpts_past = track[-6] # 5 幀前
        
        head_now, foot_now = self._get_head_foot_data(kpts_now)
        head_past, _ = self._get_head_foot_data(kpts_past)
        
        if head_now and foot_now and head_past:
            hx, hy = head_now
            fx, fy = foot_now
            hx_p, hy_p = head_past
            
            # A. 幾何特徵：頭腳必須保持一定的站立距離 (大於 50 像素，代表沒躺下)
            hf_dist = np.sqrt((hx - fx)**2 + (hy - fy)**2)
            
            # B. 動態特徵：計算鼻子點 5 幀內的總位移
            head_pixel_dist = np.sqrt((hx - hx_p)**2 + (hy - hy_p)**2)
            
            # 💡 核心自適應換算：將位移除以頭腳距離，算出相對速度
            relative_speed = (head_pixel_dist / max(hf_dist, 1)) * (fps / 5)
            
            # 如果頭腳保持距離，且相對速度大於 1.6 倍人體高度/秒，判定為跑步
            if hf_dist >= 50 and relative_speed > 1.6:
                return True, relative_speed * 100
            return False, relative_speed * 100
            
        return False, 0

    # =========================================================================
    # 🚶 3. 徘徊判定：頭腳保持正常距離 + 鼻子點在固定範圍內打轉
    # =========================================================================
    def check_loitering(self, pid, range_threshold=55, min_frames=90):
        track = self.person_tracks.get(pid)
        if not track or len(track) < min_frames: return False
        
        # 檢查最新一幀是否站立
        _, _, _, _, kpts_now = track[-1]
        head_now, foot_now = self._get_head_foot_data(kpts_now)
        if head_now and foot_now:
            hf_dist = np.sqrt((head_now[0] - foot_now[0])**2 + (head_now[1] - foot_now[1])**2)
            if hf_dist < 50: return False # 躺在地上的人不算徘徊
            
        # 蒐集最近 3 秒內所有歷史鼻子點的 X 和 Y 座標
        hx_coords = []
        hy_coords = []
        
        for i in range(len(track)):
            _, _, _, _, kpts = track[i]
            head, _ = self._get_head_foot_data(kpts)
            if head:
                hx_coords.append(head[0])
                hy_coords.append(head[1])
                
        if len(hx_coords) < min_frames // 2: return False
        
        # 💡 幾何核心：計算鼻子點移動的外包邊界
        x_range = max(hx_coords) - min(hx_coords)
        y_range = max(hy_coords) - min(hy_coords)
        
        # 如果時間待得夠久，且鼻子點移動的長寬都在 55 像素以內，代表在原地打轉徘徊
        if x_range < range_threshold and y_range < range_threshold:
            return True
        return False

    # =========================================================================
    # 💥 4. 碰撞判定：多人同框 + 鼻子距離過近 + 移動方向劇烈改變
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
    # 🚯 5. 丟垃圾判定：物體從手點(左/右手腕)分離 + 物體原地定格達標
    # =========================================================================
    def check_littering(self, oid, separation_threshold=55, static_frames_needed=30):
        obj_track = self.object_tracks.get(oid)
        if not obj_track or len(obj_track) < 5: return False
        
        ocx, ocy, _, _ = obj_track[-1]
        ocx_past, ocy_past, _, _ = obj_track[-5]
        
        # A. 檢查物件本身是否在原地定格
        is_static = np.sqrt((ocx - ocx_past)**2 + (ocy - ocy_past)**2) < 3
        if is_static:
            self.littering_timers[oid] = self.littering_timers.get(oid, 0) + 1
        else:
            self.littering_timers[oid] = 0
            return False

        # B. 東西不動時間達標（滿 1 秒），開始回溯起源手部關節
        if self.littering_timers[oid] >= static_frames_needed:
            hand_close_at_birth = False
            all_people_far = True
            
            for pid, p_track in self.person_tracks.items():
                if len(p_track) > 0:
                    pcx, pcy, _, _, kpts = p_track[-1]
                    dist_to_center = np.sqrt((ocx - pcx)**2 + (ocy - pcy)**2)
                    
                    # 如果此時有人離它很近，視為暫放
                    if dist_to_center < separation_threshold:
                        all_people_far = False
                        break
                    
                    # 💡 核心創新：提取手腕點（Keypoint 9 左手腕, 10 右手腕）
                    if kpts is not None and len(kpts) > 10:
                        try:
                            lw_x, lw_y = kpts[9][0], kpts[9][1]
                            rw_x, rw_y = kpts[10][0], kpts[10][1]
                            
                            # 只要這個垃圾剛出生時，距離全場任何一人的左手或右手腕小於 40 像素，即認定為「分離事件」
                            if (np.sqrt((ocx - lw_x)**2 + (ocy - lw_y)**2) < 20 or 
                                np.sqrt((ocx - rw_x)**2 + (ocy - rw_y)**2) < 20):
                                hand_close_at_birth = True
                        except:
                            pass
            
            # 完美咬合：人走遠了，且該物件當初確實是在某個人的手點位置被拋棄/分離出來的
            if all_people_far and hand_close_at_birth:
                return True
        return False

    def clear_dead_tracks(self, active_pids, active_oids):
        self.person_tracks = {k: v for k, v in self.person_tracks.items() if k in active_pids}
        self.object_tracks = {k: v for k, v in self.object_tracks.items() if k in active_oids}
        self.littering_timers = {k: v for k, v in self.littering_timers.items() if k in active_oids}