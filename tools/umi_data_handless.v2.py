import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time
import threading
import serial
import numpy as np
import json
import shutil
import select
from threading import Lock
from datetime import datetime
import sys
import tty
import termios
import traceback


# Camera Config
CAM_CHEST_ID = 2 # 胸部相机
CAM_WRIST_ID = 0 # 腕部相机

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

frame_timestamps = {
    CAM_CHEST_ID: [],
    CAM_WRIST_ID: []
}

captured_frames = {
    CAM_CHEST_ID: 0,
    CAM_WRIST_ID: 0
}

written_frames = {
    CAM_CHEST_ID: 0,
    CAM_WRIST_ID: 0
}

video_writers = {
    CAM_CHEST_ID: None,
    CAM_WRIST_ID: None
}

imu_file = None
imu_buffer = []

frame_lock = Lock()
writer_lock = Lock()
imu_lock = Lock()

# =========================
# Non-blocking Keyboard
# =========================
def get_char_nonblock():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
        if rlist:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# =========================
# IMU Protocol
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

def flush_imu():
    global imu_buffer, imu_file
    if imu_file and imu_buffer:
        with imu_lock:
            for sample in imu_buffer:
                imu_file.write(json.dumps(sample) + "\n")
            imu_file.flush()
            imu_buffer.clear()

def Cmd_RxUnpack(unpack_buf, DLen):
    global imu_buffer
    if not is_recording:
        return

    scaleAccel = 0.00478515625
    scaleAngleSpeed = 0.06103515625

    if unpack_buf[0] == 0x11:
        ctl = (unpack_buf[2] << 8) | unpack_buf[1]
        current_relative_t = time.perf_counter() - TIME_ZERO
        L = 7
        ax = ay = az = gx = gy = gz = 0.0

        if ctl & 0x0002:
            ax = int.from_bytes(unpack_buf[L:L+2], 'little', signed=True) * scaleAccel * 9.81
            ay = int.from_bytes(unpack_buf[L+2:L+4], 'little', signed=True) * scaleAccel * 9.81
            az = int.from_bytes(unpack_buf[L+4:L+6], 'little', signed=True) * scaleAccel * 9.81
            L += 6

        if ctl & 0x0004:
            gx = int.from_bytes(unpack_buf[L:L+2], 'little', signed=True) * scaleAngleSpeed * np.pi / 180.0
            gy = int.from_bytes(unpack_buf[L+2:L+4], 'little', signed=True) * scaleAngleSpeed * np.pi / 180.0
            gz = int.from_bytes(unpack_buf[L+4:L+6], 'little', signed=True) * scaleAngleSpeed * np.pi / 180.0

        imu_buffer.append({
            "timestamp": float(current_relative_t),
            "accelerometer": [float(ax), float(ay), float(az)],
            "gyroscope": [float(gx), float(gy), float(gz)]
        })

        if len(imu_buffer) >= 50:
            flush_imu()

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
        buf[i] = byte
        i += 1
        RxIndex = 2
    elif RxIndex == 2:
        buf[i] = byte
        i += 1
        cmdLen = byte
        RxIndex = 3
    elif RxIndex == 3:
        buf[i] = byte
        i += 1
        if i >= cmdLen + 3:
            RxIndex = 4
    elif RxIndex == 4:
        CS -= byte
        buf[i] = byte
        i += 1
        RxIndex = 5
    elif RxIndex == 5:
        RxIndex = 0
        if byte == CmdPacket_End:
            Cmd_RxUnpack(buf[3:i-1], i-4)

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
        print("[INFO] IMU started at 200Hz")
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

    if not cap.isOpened():
        print(f"[ERROR] Camera {cam_id} failed to open")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"[INFO] Camera {cam_id} started")

    while is_running:
        ret, frame = cap.read()
        if ret:
            ts = time.perf_counter()
            with frame_lock:
                latest_frames[cam_id] = (frame, ts)
                captured_frames[cam_id] += 1
        else:
            time.sleep(0.01)

    cap.release()

# =========================
# Writer Thread
# =========================
def writer_worker():
    frame_interval = 1.0 / TARGET_FPS
    next_tick = time.perf_counter()

    while is_running:
        if is_recording:
            now = time.perf_counter()
            if now >= next_tick:
                with writer_lock:
                    for cam_id in [CAM_CHEST_ID, CAM_WRIST_ID]:
                        if video_writers[cam_id]:
                            with frame_lock:
                                packet = latest_frames[cam_id]
                            if packet:
                                frame, ts = packet
                                video_writers[cam_id].write(frame)
                                frame_timestamps[cam_id].append(float(ts - TIME_ZERO))
                                written_frames[cam_id] += 1
                next_tick += frame_interval
            else:
                time.sleep(0.002)
        else:
            time.sleep(0.01)

# =========================
# Metadata
# ========================= 
def save_metadata(session_dir):
    metadata = {
        "fps": TARGET_FPS,
        "resolution": [RES_WIDTH, RES_HEIGHT],
        "imu_hz": 200,
        "frames": {
            "captured": captured_frames,
            "written": written_frames
        }
    }

    with open(os.path.join(session_dir, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

# =========================
# Recording
# =========================

# .avi
# def start_record(session_dir):
#     global TIME_ZERO, is_recording, imu_file

#     os.makedirs(os.path.join(session_dir, "camera_0"), exist_ok=True)
#     os.makedirs(os.path.join(session_dir, "camera_1"), exist_ok=True)

#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')

#     with writer_lock:
#         video_writers[CAM_WRIST_ID] = cv2.VideoWriter(
#             os.path.join(session_dir, "camera_0/raw_video.avi"),
#             fourcc, TARGET_FPS, (RES_WIDTH, RES_HEIGHT)
#         )

#         video_writers[CAM_CHEST_ID] = cv2.VideoWriter(
#             os.path.join(session_dir, "camera_1/raw_video.avi"),
#             fourcc, TARGET_FPS, (RES_WIDTH, RES_HEIGHT)
#         )

#     imu_file = open(os.path.join(session_dir, "imu_data.jsonl"), "w")

#     frame_timestamps[CAM_CHEST_ID].clear()
#     frame_timestamps[CAM_WRIST_ID].clear()

#     TIME_ZERO = time.perf_counter()
#     is_recording = True
#     print(f"\n[>>> 开始录制 <<<] {session_dir}")


# .mp4
def start_record(session_dir):
    global TIME_ZERO, is_recording, imu_file

    os.makedirs(os.path.join(session_dir, "camera_0"), exist_ok=True)
    os.makedirs(os.path.join(session_dir, "camera_1"), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    with writer_lock:
        video_writers[CAM_WRIST_ID] = cv2.VideoWriter(
            os.path.join(session_dir, "camera_0/raw_video.mp4"),
            fourcc,
            TARGET_FPS,
            (RES_WIDTH, RES_HEIGHT)
        )

        video_writers[CAM_CHEST_ID] = cv2.VideoWriter(
            os.path.join(session_dir, "camera_1/raw_video.mp4"),
            fourcc,
            TARGET_FPS,
            (RES_WIDTH, RES_HEIGHT)
        )

    imu_file = open(os.path.join(session_dir, "imu_data.jsonl"), "w")

    frame_timestamps[CAM_CHEST_ID].clear()
    frame_timestamps[CAM_WRIST_ID].clear()

    TIME_ZERO = time.perf_counter()
    is_recording = True

    print(f"\n[>>> 开始录制 MP4 <<<] {session_dir}")


# =========================
# Stop Save
# =========================
def stop_and_save(session_dir):
    global is_recording, imu_file

    is_recording = False
    flush_imu()

    with writer_lock:
        for k in video_writers:
            if video_writers[k]:
                video_writers[k].release()
                video_writers[k] = None

    if imu_file:
        imu_file.close()
        imu_file = None

    with open(os.path.join(session_dir, "camera_0/timestamps.json"), "w") as f:
        json.dump(frame_timestamps[CAM_WRIST_ID], f)

    with open(os.path.join(session_dir, "camera_1/timestamps.json"), "w") as f:
        json.dump(frame_timestamps[CAM_CHEST_ID], f)

    save_metadata(session_dir)
    print(f"\n[### 保存完成 ###] {session_dir}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    threading.Thread(target=imu_worker, daemon=True).start()
    threading.Thread(target=camera_worker, args=(CAM_CHEST_ID,), daemon=True).start()
    threading.Thread(target=camera_worker, args=(CAM_WRIST_ID,), daemon=True).start()
    threading.Thread(target=writer_worker, daemon=True).start()

    print("\n" + "="*45)
    print(" 树莓派无头同步采集系统已启动")
    print("="*45)
    print(" [空格] : 开始录制")
    print(" [ s ]  : 停止并保存")
    print(" [ q ]  : 放弃录制")
    print(" [ESC]  : 退出")
    print("="*45 + "\n")

    try:
        while is_running:
            key = get_char_nonblock()

            if key == ' ':
                if not is_recording:
                    curr_dir = os.path.join("demos", f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    start_record(curr_dir)

            elif key == 's':
                if is_recording:
                    stop_and_save(curr_dir)

            elif key == 'q':
                if is_recording:
                    is_recording = False
                    time.sleep(0.1)
                    shutil.rmtree(curr_dir)
                    print("[已删除废弃录制]")

            elif key == '\x1b':
                is_running = False
                print("[安全退出]")
                break

    except KeyboardInterrupt:
        is_running = False
        print("[Ctrl+C退出]")

    except Exception:
        traceback.print_exc()
