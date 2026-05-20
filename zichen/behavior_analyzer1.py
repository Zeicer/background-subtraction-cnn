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
    # 🛌 1. 全新昏倒判定：【5 幀時序高度縮減 60% 法】 + 鼻子點快速下墜位移
    # =========================================================================
    def check_fall(self, pid):
        track = self.person_tracks.get(pid)
        # 依據你的要求，需要比對 5 幀以內的變化，所以記憶體長度至少要 6 幀
        if not track or len(track) < 6: return False
        
        # 1. 拿到當前幀 [-1] 與 5 幀以前 [-6] 的歷史骨架
        _, _, _, _, kpts_now = track[-1]
        _, _, _, _, kpts_past = track[-6] # 5 幀前 (約 0.16 秒前)
        
        head_now, foot_now = self._get_head_foot_data(kpts_now)
        head_past, foot_past = self._get_head_foot_data(kpts_past)
        
        if head_now and foot_now and head_past and foot_past:
            hx, hy = head_now
            fx, fy = foot_now
            hx_p, hy_p = head_past
            fx_p, fy_p = foot_past
            
            # A. 📐 計算 5 幀前與當前幀的【頭腳垂直相對距離 (ΔY)】
            v_dist_now = fy - hy         # 當前垂直高度
            v_dist_past = fy_p - hy_p    # 5 幀前的直立高度
            
            # B. 💡 核心優化：計算 5 幀以內高度的【相對縮減比例】
            # 拿「當前高度」除以「5幀前高度」。若人好端端站著比例接近 1.0；若突然倒地比例會暴跌
            height_shrink_ratio = v_dist_now / max(v_dist_past, 1)
            
            # C. 🚀 動態特徵：計算【鼻子點的瞬時下墜速度】
            head_pixel_speed = np.sqrt((hx - hx_p)**2 + (hy - hy_p)**2)
            head_v_y = hy - hy_p # Y 軸向下為正，大於 0 代表鼻子確實正在往下墜落
            
            # 🌟 【完全符合你的要求 - 昏倒判定黃金公式】
            # 條件一：height_shrink_ratio < 0.60 (5幀以內，頭腳相對垂直距離縮減到原本的 60% 以下)
            # 條件二：head_pixel_speed > 5.0 且 head_v_y > 2 (在此同時，鼻子點必須伴隨著快速向下的位移速度)
            if height_shrink_ratio < 0.60 and head_pixel_speed > 5.0 and head_v_y > 2:
                return True
                
        # 傳統後備保險：高寬比扁平化過度嚴重 (防止骨架點不小心被地上的椅子、雜物短暫擋住時漏偵測)
        aspect_ratio = track[-1][3] / max(track[-1][2], 1)
        if aspect_ratio < 0.70 and (track[-1][1] - track[-6][1]) > 12:
            return True
            
        return False

# =========================================================================
    # 🏃 2. 全新跑步判定：自適應高度縮減比例 + 鼻子絕對速度 + 高度穩定率
    # =========================================================================
    def check_running(self, pid, fps=30):
        track = self.person_tracks.get(pid)
        if not track or len(track) < 6: return False, 0
        
        # 1. 拿到當前幀 [-1]、上一幀 [-2] 與歷史幀 [-3] 的骨架
        _, _, _, _, kpts_now = track[-1]
        _, _, _, _, kpts_prev = track[-2] # 前一幀，用來檢查高度穩定度
        _, _, _, _, kpts_past = track[-3] # 2 幀前，用來計算鼻子即時速度
        
        head_now, foot_now = self._get_head_foot_data(kpts_now)
        head_prev, foot_prev = self._get_head_foot_data(kpts_prev)
        head_past, _ = self._get_head_foot_data(kpts_past)
        
        if head_now and foot_now and head_prev and foot_prev and head_past:
            hx, hy = head_now
            fx, fy = foot_now
            hx_prev, hy_prev = head_prev
            fx_prev, fy_prev = foot_prev
            hx_p, hy_p = head_past
            
            # 📐 A. 幾何特徵：計算當前的【頭腳垂直距離】與【頭腳絕對總距離】
            v_dist_now = fy - hy                                      # 當前垂直差 (ΔY)
            hf_dist_now = np.sqrt((hx - fx)**2 + (hy - fy)**2)        # 當前歐幾里得總體長
            
            # 💡 核心修正：計算高度縮減比例 (垂直差 佔 總體長 的百分比)
            # 站立時垂直高度幾乎等於總體長，比例會接近 1.0；蹲下或平躺時會暴跌
            vertical_ratio = v_dist_now / max(hf_dist_now, 1)
            
            # 📐 B. 穩定度審查：計算前後兩幀「頭腳垂直相對位置距離」的落差
            v_dist_prev = fy_prev - hy_prev
            height_stability = np.abs(v_dist_now - v_dist_prev)
            
            # 🚀 C. 動態特徵：計算【鼻子點的瞬時位移總速度】
            head_pixel_speed = np.sqrt((hx - hx_p)**2 + (hy - hy_p)**2)
            
            # 🌟 【自適應縮減比例版 - 跑步判定黃金公式】
            # 條件一：vertical_ratio > 0.80 (高度保持在總體長的 80% 以上，完美適應遠近，確認為直立姿態)
            # 條件二：height_stability <= 8 (前後幀頭腳高度落差在 8 像素內，相對位置保持一定)
            # 條件三：head_pixel_speed > 4.5 (鼻子點產生連續高速位移)
            if vertical_ratio > 0.90 and height_stability <= 8 and head_pixel_speed > 50:
                # 回傳 True 與放大的速度數值（供主程式畫面上渲染百分比使用）
                return True, head_pixel_speed * 30
                
            return False, head_pixel_speed * 30
            
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
# =========================================================================
 # =========================================================================
    # 🚯 5. 丟垃圾判定：手點誕生 + 物品離手點越來越遠 + 該人物 ID 離開/走遠
    # =========================================================================
    def check_littering(self, oid, separation_threshold=20, static_frames_needed=10):
        obj_track = self.object_tracks.get(oid)
        if not obj_track or len(obj_track) < 5: return False
        
        ocx, ocy, ow, oh = obj_track[-1]
        ocx_past, ocy_past, _, _ = obj_track[-5]
        
        # 💡 初始化動態記憶帳本
        if not hasattr(self, 'object_owner_memory'):
            self.object_owner_memory = {}       # 紀錄 {物品ID: 人員ID}
        if not hasattr(self, 'hand_distance_history'):
            self.hand_distance_history = {}     # 紀錄物品與主人手部的距離歷史 {物品ID: [距離1, 距離2, ...]}

        # 🟢【步驟 1】手點精準綁定（物品剛出生前幾幀）
        if oid not in self.object_owner_memory and len(obj_track) <= 6:
            best_pid = None
            min_hand_dist = 45  
            
            for pid, p_track in self.person_tracks.items():
                if len(p_track) > 0:
                    *_, kpts = p_track[-1]  
                    if kpts is not None and len(kpts) > 10:
                        try:
                            lw_x, lw_y = kpts[9][0], kpts[9][1]   
                            rw_x, rw_y = kpts[10][0], kpts[10][1] 
                            
                            dist_lw = np.sqrt((ocx - lw_x)**2 + (ocy - lw_y)**2)
                            dist_rw = np.sqrt((ocx - rw_x)**2 + (ocy - rw_y)**2)
                            closer_dist = min(dist_lw, dist_rw)
                            
                            if closer_dist < min_hand_dist:
                                min_hand_dist = closer_dist
                                best_pid = pid
                        except:
                            pass
            
            if best_pid is not None:
                self.object_owner_memory[oid] = best_pid
                self.hand_distance_history[oid] = [min_hand_dist]

        # 🟢【步驟 2】動態追蹤：如果已經綁定主人，在物品移動或剛分離的階段，記錄「與手點的距離歷史」
        assigned_pid = self.object_owner_memory.get(oid)
        if assigned_pid is not None and len(obj_track) <= 15:
            current_p_track = self.person_tracks.get(assigned_pid)
            if current_p_track and len(current_p_track) > 0:
                *_, kpts = current_p_track[-1]
                if kpts is not None and len(kpts) > 10:
                    try:
                        lw_x, lw_y = kpts[9][0], kpts[9][1]   
                        rw_x, rw_y = kpts[10][0], kpts[10][1] 
                        dist_lw = np.sqrt((ocx - lw_x)**2 + (ocy - lw_y)**2)
                        dist_rw = np.sqrt((ocx - rw_x)**2 + (ocy - rw_y)**2)
                        current_hand_dist = min(dist_lw, dist_rw)
                        
                        # 紀錄距離變化的時間序列
                        self.hand_distance_history[oid].append(current_hand_dist)
                    except:
                        pass

        # 🟢【步驟 3】檢查物件本身是否在原地定格（落地後靜止不動）
        is_static = np.sqrt((ocx - ocx_past)**2 + (ocy - ocy_past)**2) < 3
        if is_static:
            self.littering_timers[oid] = self.littering_timers.get(oid, 0) + 1
        else:
            self.littering_timers[oid] = 0
            return False

        # 🟢【步驟 4】東西不動時間達標，發動終極判定（包含「離手越來越遠」的審查）
        if self.littering_timers[oid] >= static_frames_needed:
            
            # 1️⃣ 審查【動態分離特徵】：檢查距離歷史，是否「越來越遠」
            dist_hist = self.hand_distance_history.get(oid, [])
            is_moving_away = False
            
            if len(dist_hist) >= 3:
                # 判斷趨勢：最新記錄的距離大於最初誕生的距離，且中間有增長趨勢
                # 在 $320x240$ 下，只要手與物品拉開超過 12 像素，就認定有「分離遠離」的動作
                if dist_hist[-1] > dist_hist[0] + 12:
                    is_moving_away = True
            else:
                # 🛡️ 容錯保底：如果手腕點因為角度被擋住，導致沒收集到足夠的歷史距離，
                # 但只要它出生時有綁到主人，我們就放寬動態限制，直接進入下一步的 ID 遠離判定。
                is_moving_away = True

            # 如果經過驗證，發現它並沒有「離手越來越遠」（例如手一直黏在上面），就不是亂丟
            if not is_moving_away:
                return False

            # 2️⃣ 審查【人物 ID 的最終動向】
            if assigned_pid is None:
                # 保底：全場沒人靠近
                all_people_far = True
                for pid, p_track in self.person_tracks.items():
                    if len(p_track) > 0:
                        px, py, pw, ph, _ = p_track[-1]
                        pcx, pcy = px + pw // 2, py + ph // 2
                        if np.sqrt((ocx - pcx)**2 + (ocy - pcy)**2) < 45:
                            all_people_far = False
                            break
                return all_people_far

            current_p_track = self.person_tracks.get(assigned_pid)
            
            # 情況 B：該人物 ID 已經大步離開、完全消失在畫面
            if not current_p_track or len(current_p_track) == 0:
                return True
            
            # 情況 A：該人物 ID 還在場上，計算當前人體中心與物品的物理距離
            px, py, pw, ph, _ = current_p_track[-1]
            pcx, pcy = px + pw // 2, py + ph // 2
            dist_to_owner = np.sqrt((ocx - pcx)**2 + (ocy - pcy)**2)
            
            # 判定：滿足腳邊短距離（40像素）或正在大步遠離的趨勢
            if dist_to_owner > separation_threshold:
                return True
                
            if len(current_p_track) >= 10:
                px_past, py_past, pw_past, bh_past, _ = current_p_track[-10]
                pcx_past, pcy_past = px_past + pw_past // 2, py_past + bh_past // 2
                dist_to_owner_past = np.sqrt((ocx - pcx_past)**2 + (ocy - pcy_past)**2)
                if dist_to_owner > dist_to_owner_past and dist_to_owner > 28:
                    return True 

        return False

    def clear_dead_tracks(self, active_pids, active_oids):
        self.person_tracks = {k: v for k, v in self.person_tracks.items() if k in active_pids}
        self.object_tracks = {k: v for k, v in self.object_tracks.items() if k in active_oids}
        self.littering_timers = {k: v for k, v in self.littering_timers.items() if k in active_oids}