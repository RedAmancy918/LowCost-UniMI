# import os
# import cv2
# import numpy as np
# import json

# # --- 配置区 ---
# DEMO_PATH = "demos/demo_20260323_155409"  # 替换成实际路径
# # ----------------

# def draw_imu_plot(times, values, current_t, width, height):
#     """
#     绘制 IMU 曲线图
#     """
#     plot_img = np.zeros((height, width, 3), dtype=np.uint8) + 40
#     if not times: 
#         return plot_img

#     max_v = max(max(values), 20)
#     min_v = min(min(values), -20)
#     range_v = max_v - min_v if max_v != min_v else 1

#     def to_y(v): 
#         return int(height - (v - min_v) / range_v * height)
#     def to_x(t): 
#         return int((t / max(times)) * width) if max(times) > 0 else 0

#     # 背景参考线 (0 刻度)
#     zero_y = to_y(0)
#     cv2.line(plot_img, (0, zero_y), (width, zero_y), (100, 100, 100), 1)

#     # 曲线
#     pts = [[to_x(t), to_y(v)] for t, v in zip(times, values)]
#     if len(pts) > 1:
#         cv2.polylines(plot_img, [np.array(pts, np.int32)], False, (100, 100, 255), 1)

#     # 当前播放指示线
#     curr_x = to_x(current_t)
#     cv2.line(plot_img, (curr_x, 0), (curr_x, height), (255, 150, 0), 2)

#     cv2.putText(plot_img, f"IMU Z-Accel | Time: {current_t:.3f}s", (10, 20), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#     return plot_img

# def load_imu_data(imu_file_path):
#     """
#     加载 JSONL 格式的 IMU 数据
#     """
#     times = []
#     accel_z = []
#     if not os.path.exists(imu_file_path):
#         print("[WARN] 找不到 IMU 文件:", imu_file_path)
#         return times, accel_z

#     with open(imu_file_path, 'r') as f:
#         for line in f:
#             d = json.loads(line)
#             times.append(d['timestamp'])
#             accel_z.append(d['accelerometer'][2])
#     return times, accel_z

# def play_sync_check():
#     # 视频路径
#     v0_path = os.path.join(DEMO_PATH, "camera_0", "raw_video.mp4")
#     v1_path = os.path.join(DEMO_PATH, "camera_1", "raw_video.mp4")
#     imu_path = os.path.join(DEMO_PATH, "imu_data.jsonl")

#     # 加载 IMU 数据
#     imu_times, accel_z = load_imu_data(imu_path)

#     # 初始化视频
#     cap0 = cv2.VideoCapture(v0_path)
#     cap1 = cv2.VideoCapture(v1_path)
#     fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0

#     win_name = "UMI Interactive Sync Checker"
#     cv2.namedWindow(win_name)
#     print("[INFO] 按 'SPACE' 暂停/播放，按 'Q' 退出")

#     is_paused = False
#     current_frame = 0

#     while True:
#         if not is_paused:
#             ret0, frame0 = cap0.read()
#             ret1, frame1 = cap1.read()

#             if not ret0 or not ret1:
#                 # 视频播放完重头开始
#                 current_frame = 0
#                 cap0.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 continue

#             # 当前时间
#             curr_time = current_frame / fps
#             current_frame += 1

#             # 绘制 IMU 曲线
#             width = frame0.shape[1] * 2
#             imu_view = draw_imu_plot(imu_times, accel_z, curr_time, width, 200)

#             # 标注视频
#             cv2.putText(frame0, f"Wrist (CAM0) | Time: {curr_time:.2f}s", (20, 40), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(frame1, f"Chest (CAM1) | Time: {curr_time:.2f}s", (20, 40), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # 拼接显示
#             combined = np.vstack((np.hstack((frame0, frame1)), imu_view))
#             cv2.imshow(win_name, combined)

#         delay = int(1000 / fps) if not is_paused else 30
#         key = cv2.waitKey(delay) & 0xFF

#         if key == ord(' '):
#             is_paused = not is_paused
#         elif key == ord('q'):
#             break

#     cap0.release()
#     cap1.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     play_sync_check()

import os
import cv2
import numpy as np
import json
import time

# --- 配置区 ---
DEMO_PATH = "demos/demo_20260323_164940"  # 替换成实际路径
# ----------------

def draw_imu_plot(times, values, current_t, width, height):
    plot_img = np.zeros((height, width, 3), dtype=np.uint8) + 40
    if not times: 
        return plot_img

    max_v = max(max(values), 20)
    min_v = min(min(values), -20)
    range_v = max_v - min_v if max_v != min_v else 1

    def to_y(v): 
        return int(height - (v - min_v) / range_v * height)
    def to_x(t): 
        return int((t / max(times)) * width) if max(times) > 0 else 0

    zero_y = to_y(0)
    cv2.line(plot_img, (0, zero_y), (width, zero_y), (100, 100, 100), 1)

    pts = [[to_x(t), to_y(v)] for t, v in zip(times, values)]
    if len(pts) > 1:
        cv2.polylines(plot_img, [np.array(pts, np.int32)], False, (100, 100, 255), 1)

    curr_x = to_x(current_t)
    cv2.line(plot_img, (curr_x, 0), (curr_x, height), (255, 150, 0), 2)
    cv2.putText(plot_img, f"IMU Z-Accel | Time: {current_t:.3f}s", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return plot_img

def load_imu_data(imu_file_path):
    times = []
    accel_z = []
    if not os.path.exists(imu_file_path):
        print("[WARN] 找不到 IMU 文件:", imu_file_path)
        return times, accel_z

    with open(imu_file_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            times.append(d['timestamp'])
            accel_z.append(d['accelerometer'][2])
    return times, accel_z

def play_sync_check():
    # 视频路径
    v0_path = os.path.join(DEMO_PATH, "camera_0", "raw_video.mp4")
    v1_path = os.path.join(DEMO_PATH, "camera_1", "raw_video.mp4")
    imu_path = os.path.join(DEMO_PATH, "imu_data.jsonl")

    imu_times, accel_z = load_imu_data(imu_path)

    cap0 = cv2.VideoCapture(v0_path)
    cap1 = cv2.VideoCapture(v1_path)
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / fps

    win_name = "UMI Smooth Sync Playback"
    cv2.namedWindow(win_name)
    print("[INFO] 按 'SPACE' 暂停/播放，按 'Q' 退出")

    is_paused = False
    frame_idx = 0
    start_time = time.perf_counter()

    while True:
        # 当前视频时间（秒）
        elapsed = time.perf_counter() - start_time
        target_frame = int(elapsed * fps)

        if not is_paused and target_frame > frame_idx:
            # 拉帧到对应时间
            cap0.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            if not ret0 or not ret1:
                break  # 播放结束
            frame_idx = target_frame

            curr_time = frame_idx / fps
            imu_view = draw_imu_plot(imu_times, accel_z, curr_time, frame0.shape[1]*2, 200)

            cv2.putText(frame0, f"Wrist (CAM0) | Time: {curr_time:.2f}s", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame1, f"Chest (CAM1) | Time: {curr_time:.2f}s", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            combined = np.vstack((np.hstack((frame0, frame1)), imu_view))
            cv2.imshow(win_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            is_paused = not is_paused
            if not is_paused:
                # 调整 start_time 保持流畅播放
                start_time = time.perf_counter() - frame_idx / fps
        elif key == ord('q'):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_sync_check()