import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time
import threading
import serial
import numpy as np
import json
import shutil
from collections import deque
from threading import Lock
from datetime import datetime

# =========================
# Camera Config
# =========================
CAM_CHEST_ID = 2
CAM_WRIST_ID = 0

RES_WIDTH = 640
RES_HEIGHT = 480
TARGET_FPS = 30

# =========================
# IMU Config
# =========================
IMU_PORT = "/dev/serial0"
IMU_BAUD = 115200

# =========================
# Global Variables
# =========================
TIME_ZERO = 0.0
is_recording = False
is_running = True
curr_dir = None

latest_frames = {
    CAM_CHEST_ID: None,
    CAM_WRIST_ID: None
}

frame_queues = {
    CAM_CHEST_ID: deque(),
    CAM_WRIST_ID: deque()
}

frame_timestamps = {
    CAM_CHEST_ID: [],
    CAM_WRIST_ID: []
}

imu_data_list = []

video_writers = {
    CAM_CHEST_ID: None,
    CAM_WRIST_ID: None
}

frame_lock = Lock()

# =========================
# IMU protocol
# =========================
CmdPacket_Begin = 0x49
CmdPacket_End = 0x4D
CmdPacketMaxDatSizeRx = 73

CS = 0
i = 0
RxIndex = 0
cmdLen = 0
buf = bytearray(5 + CmdPacketMaxDatSizeRx)
ser = None

def Cmd_RxUnpack(unpack_buf, DLen):
    global imu_data_list

    if not is_recording:
        return

    scaleAccel = 0.00478515625
    scaleAngleSpeed = 0.06103515625

    if unpack_buf[0] == 0x11:
        ctl = (unpack_buf[2] << 8) | unpack_buf[1]
        current_relative_t = time.perf_counter() - TIME_ZERO
        L = 7
        ax = ay = az = gx = gy = gz = 0.0

        if (ctl & 0x0002):
            ax = int.from_bytes(unpack_buf[L:L+2], byteorder='little', signed=True) * scaleAccel * 9.81
            ay = int.from_bytes(unpack_buf[L+2:L+4], byteorder='little', signed=True) * scaleAccel * 9.81
            az = int.from_bytes(unpack_buf[L+4:L+6], byteorder='little', signed=True) * scaleAccel * 9.81
            L += 6

        if (ctl & 0x0004):
            gx = int.from_bytes(unpack_buf[L:L+2], byteorder='little', signed=True) * scaleAngleSpeed * np.pi / 180.0
            gy = int.from_bytes(unpack_buf[L+2:L+4], byteorder='little', signed=True) * scaleAngleSpeed * np.pi / 180.0
            gz = int.from_bytes(unpack_buf[L+4:L+6], byteorder='little', signed=True) * scaleAngleSpeed * np.pi / 180.0

        imu_data_list.append({
            "timestamp": float(current_relative_t),
            "accelerometer": [float(ax), float(ay), float(az)],
            "gyroscope": [float(gx), float(gy), float(gz)]
        })

def Cmd_GetPkt(byte):
    global CS, i, RxIndex, buf, cmdLen
    CS += byte
    if RxIndex == 0:
        if byte == CmdPacket_Begin:
            i = 0
            buf[i] = CmdPacket_Begin
            i += 1
            CS = 0
            RxIndex = 1
    elif RxIndex == 1:
        buf[i] = byte; i += 1; RxIndex = 2
    elif RxIndex == 2:
        buf[i] = byte; i += 1; cmdLen = byte; RxIndex = 3
    elif RxIndex == 3:
        buf[i] = byte; i += 1
        if i >= cmdLen + 3: RxIndex = 4
    elif RxIndex == 4:
        CS -= byte; buf[i] = byte; i += 1; RxIndex = 5
    elif RxIndex == 5:
        RxIndex = 0
        if byte == CmdPacket_End:
            Cmd_RxUnpack(buf[3:i-1], i-4)
            return 1
    return 0

def Cmd_PackAndTx(pDat, DLen):
    tx_buf = bytearray([0x00]*46) + bytearray([0x00,0xff,0x00,0xff,0x49,0xFF,DLen]) + bytearray(pDat[:DLen])
    CS_tx = sum(tx_buf[51:51+DLen+2]) & 0xFF
    tx_buf.append(CS_tx)
    tx_buf.append(0x4D)
    ser.write(tx_buf)

# =========================
# IMU Thread
# =========================
def imu_worker():
    global ser
    try:
        ser = serial.Serial(IMU_PORT, IMU_BAUD, timeout=0.1)
        time.sleep(1.0)
        print("[INFO] IMU 200Hz start")
        Cmd_PackAndTx(bytearray([0x12,5,255,0,4,200,1,3,5,0x06,0x00]), 11)
        while is_running:
            if ser.in_waiting > 0:
                raw_byte = ser.read(1)
                if raw_byte:
                    Cmd_GetPkt(raw_byte[0])
    except Exception as e:
        print("[IMU ERROR]", e)

# =========================
# Camera Thread
# =========================
def camera_worker(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    while is_running:
        ret, frame = cap.read()
        
        # 移除了画面旋转逻辑，保持原始物理画面
        if ret:
            ts = time.perf_counter() - TIME_ZERO
            with frame_lock:
                latest_frames[cam_id] = frame.copy()
                if is_recording:
                    frame_queues[cam_id].append((ts, frame.copy()))
                    frame_timestamps[cam_id].append(ts)
    cap.release()

# =========================
# Writer Thread
# =========================
def writer_worker():
    while is_running:
        for cam_id in [CAM_CHEST_ID, CAM_WRIST_ID]:
            if is_recording and video_writers[cam_id] is not None:
                if frame_queues[cam_id]:
                    ts, frame = frame_queues[cam_id].popleft()
                    video_writers[cam_id].write(frame)
        time.sleep(0.001)

# =========================
# Recording
# =========================
def start_record(session_dir):
    global TIME_ZERO, is_recording

    os.makedirs(os.path.join(session_dir, "camera_0"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "camera_1"), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writers[CAM_WRIST_ID] = cv2.VideoWriter(
        os.path.join(session_dir, "camera_0/raw_video.mp4"),
        fourcc, TARGET_FPS, (RES_WIDTH, RES_HEIGHT)
    )

    video_writers[CAM_CHEST_ID] = cv2.VideoWriter(
        os.path.join(session_dir, "camera_1/raw_video.mp4"),
        fourcc, TARGET_FPS, (RES_WIDTH, RES_HEIGHT)
    )

    frame_queues[CAM_CHEST_ID].clear()
    frame_queues[CAM_WRIST_ID].clear()
    frame_timestamps[CAM_CHEST_ID].clear()
    frame_timestamps[CAM_WRIST_ID].clear()
    imu_data_list.clear()

    TIME_ZERO = time.perf_counter()
    is_recording = True

    print(f"\n[>>> 开始录制 <<<] 数据存放于: {session_dir}")

def stop_and_save(session_dir):
    global is_recording
    is_recording = False
    time.sleep(0.2)

    for k in video_writers:
        if video_writers[k]:
            video_writers[k].release()
            video_writers[k] = None

    with open(os.path.join(session_dir, "imu_data.json"), 'w') as f:
        json.dump(imu_data_list, f, indent=2)

    with open(os.path.join(session_dir, "camera_0/timestamps.json"), 'w') as f:
        json.dump(frame_timestamps[CAM_WRIST_ID], f, indent=2)

    with open(os.path.join(session_dir, "camera_1/timestamps.json"), 'w') as f:
        json.dump(frame_timestamps[CAM_CHEST_ID], f, indent=2)

    print(f"\n[### 录制已保存 ###] 数据已写入: {session_dir}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    threading.Thread(target=imu_worker, daemon=True).start()
    threading.Thread(target=camera_worker, args=(CAM_CHEST_ID,), daemon=True).start()
    threading.Thread(target=camera_worker, args=(CAM_WRIST_ID,), daemon=True).start()
    threading.Thread(target=writer_worker, daemon=True).start()

    cv2.namedWindow("UMI_System")

    print("\n" + "="*45)
    print(" 树莓派采集系统已启动 (GUI 模式)")
    print("="*45)
    print(" 请确保点击激活了视频画面窗口后，再按下以下按键：")
    print(" [空格] : 开始录制新数据")
    print(" [ s ]  : 停止并保存当前录制")
    print(" [ q ]  : 放弃本次录制，并直接删除废弃数据")
    print(" [ESC]  : 退出整个程序")
    print("="*45 + "\n")

    try:
        while is_running:
            f_c = latest_frames.get(CAM_CHEST_ID)
            f_w = latest_frames.get(CAM_WRIST_ID)

            if f_c is not None and f_w is not None:
                combined = np.hstack((f_c, f_w))
                
                # 在画面上绘制状态提示
                status_color = (0, 0, 255) if is_recording else (0, 255, 0)
                status_text = "RECORDING" if is_recording else "IDLE"
                
                cv2.putText(
                    combined,
                    f"Status: {status_text} | IMU: {len(imu_data_list)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2
                )
                
                # 动态提示键位
                if is_recording:
                    cv2.putText(
                        combined,
                        "Press 's' to SAVE, 'q' to ABORT",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
                    )
                else:
                    cv2.putText(
                        combined,
                        "Press 'SPACE' to START, 'ESC' to QUIT",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                    )

                cv2.imshow("UMI_System", combined)

            # 监听键盘输入
            key = cv2.waitKey(1) & 0xFF

            # 空格键: 开始录制
            if key == ord(' '):
                if not is_recording:
                    curr_dir = os.path.join("demos", f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    start_record(curr_dir)
                else:
                    print("\n[警告] 正在录制中...请按 's' 保存或按 'q' 放弃。")

            # s 键: 保存
            elif key == ord('s'):
                if is_recording:
                    stop_and_save(curr_dir)
                    print("\n程序继续运行，等待下一次指令 (空格:开始, ESC:退出)...")
                else:
                    print("\n[提示] 当前没有录制任务。按 [空格] 开始。")

            # q 键: 放弃本次录制，删文件，不退出程序
            elif key == ord('q'):
                if is_recording:
                    is_recording = False
                    time.sleep(0.2) # 等待线程缓冲
                    
                    # 释放写入器，释放文件锁
                    for k in video_writers:
                        if video_writers[k]:
                            video_writers[k].release()
                            video_writers[k] = None
                            
                    # 彻底删除生成的无用文件夹和视频碎片
                    if curr_dir and os.path.exists(curr_dir):
                        shutil.rmtree(curr_dir)
                        
                    print(f"\n[!!! 录制已废弃 !!!] 数据已被彻底删除: {curr_dir}")
                    print("程序继续运行，按 [空格] 可开始新录制。")
                else:
                    print("\n[提示] 当前没有录制任务可以放弃。")

            # ESC 键: 彻底退出程序
            elif key == 27: 
                is_running = False
                if is_recording:
                    is_recording = False
                    time.sleep(0.2)
                    for k in video_writers:
                        if video_writers[k]:
                            video_writers[k].release()
                    if curr_dir and os.path.exists(curr_dir):
                        shutil.rmtree(curr_dir)
                print("\n[INFO] 系统安全退出。")
                break

    except KeyboardInterrupt:
        is_running = False
        print("\n[INFO] 检测到 Ctrl+C，强制退出。")
    finally:
        cv2.destroyAllWindows()