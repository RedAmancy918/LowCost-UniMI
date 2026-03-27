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

# hardware Config (

CAM_WRIST_LEFT_ID = 2  # 左手相机ID
CAM_CHEST_ID = 4      # 胸部相机ID
CAM_WRIST_RIGHT_ID = 0  # 右手相机ID
CAM_IDS = [CAM_WRIST_LEFT_ID, CAM_CHEST_ID, CAM_WRIST_RIGHT_ID]

CAM_MAPPING = {
    CAM_WRIST_LEFT_ID: "camera_0",
    CAM_CHEST_ID: "camera_1",
    CAM_WRIST_RIGHT_ID: "camera_2"
}

IMU_LEFT_PORT = "/dev/ttyUSB0" # 左手imu串口
IMU_RIGHT_PORT = "/dev/serial0" # 右手IMU串口
IMU_BAUD = 115200 # 波特率


# video cofig
RES_WIDTH = 640 
RES_HEIGHT = 480 
TARGET_FPS = 30 

# state machine
TIME_ZERO = 0.0
is_recording = False
is_running = True
current_task_name = ""

latest_frames = {k: None for k in CAM_IDS}
frame_timestamps = {k: [] for k in CAM_IDS}
captured_frames = {k: 0 for k in CAM_IDS}
written_frames = {k: 0 for k in CAM_IDS}
video_writers = {k: None for k in CAM_IDS}

frame_lock = Lock()
writer_lock = Lock()

class IMUHandler:
    def __init__(self, port, baud, name):
        self.port = port
        self.baud = baud
        self.name = name
        self.ser = None
        self.imu_file = None
        self.imu_buffer = []
        self.lock = Lock()

        self.CS = 0
        self.i = 0
        self.RxIndex = 0
        self.cmdLen = 0
        self.buf = bytearray(200)

        self.CmdPacket_Begin = 0x49
        self.CmdPacket_End = 0x4D

    def flush_to_file(self):
        if self.imu_file and self.imu_buffer:
            with self.lock:
                for sample in self.imu_buffer:
                    self.imu_file.write(json.dumps(sample) + "\n")
                self.imu_file.flush()
                self.imu_buffer.clear()

    def unpack_payload(self, unpack_buf):
        if not is_recording: return

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

            self.imu_buffer.append({
                "timestamp": float(current_relative_t),
                "accelerometer": [float(ax), float(ay), float(az)],
                "gyroscope": [float(gx), float(gy), float(gz)]
            })

            if len(self.imu_buffer) >= 20: # 稍微调小一点，确保快速写入
                self.flush_to_file()

    def parse_byte(self, byte):
        self.CS += byte
        if self.RxIndex == 0:
            if byte == self.CmdPacket_Begin:
                self.i = 0
                self.buf[self.i] = self.CmdPacket_Begin
                self.i += 1
                self.CS = 0
                self.RxIndex = 1
        elif self.RxIndex == 1:
            self.buf[self.i] = byte
            self.i += 1
            if byte == 255:
                self.RxIndex = 0
            else:
                self.RxIndex += 1
        elif self.RxIndex == 2:
            self.buf[self.i] = byte
            self.i += 1
            if byte > 73 or byte == 0:
                self.RxIndex = 0
            else:
                self.RxIndex += 1
                self.cmdLen = byte
        elif self.RxIndex == 3:
            self.buf[self.i] = byte
            self.i += 1
            if self.i >= self.cmdLen + 3:
                self.RxIndex += 1
        elif self.RxIndex == 4:
            self.CS -= byte
            if (self.CS & 0xFF) == byte:
                self.buf[self.i] = byte
                self.i += 1
                self.RxIndex += 1
            else:
                self.RxIndex = 0
        elif self.RxIndex == 5:
            self.RxIndex = 0
            if byte == self.CmdPacket_End:
                self.buf[self.i] = byte
                self.i += 1
                self.unpack_payload(self.buf[3 : self.i-2])

    def send_init_cmd(self, pDat):
        DLen = len(pDat)
        tx_buf = bytearray([0x00]*46) + bytearray([0x00,0xff,0x00,0xff,0x49,0xFF,DLen]) + bytearray(pDat)
        CS_tx = sum(tx_buf[51:51+DLen+2]) & 0xFF
        tx_buf.append(CS_tx)
        tx_buf.append(0x4D)
        if self.ser and self.ser.is_open:
            self.ser.write(tx_buf)

    def worker(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(1.0)
            self.ser.reset_input_buffer()
            

            # --- 配置 -> 唤醒 -> 开启主动上报 ---
            
            # 1. 发送配置参数 
            config_params = [0x12, 5, 255, 0, 4, 100, 1, 3, 5, 0x7F, 0x00]
            self.send_init_cmd(bytearray(config_params))
            time.sleep(0.2)
            
            # 2. 唤醒传感器 ，官方的 [0x03]
            self.send_init_cmd(bytearray([0x03]))
            time.sleep(0.2)
            
            # 3. 开启主动上报 ，官方的 [0x19]
            self.send_init_cmd(bytearray([0x19]))
            time.sleep(0.2)
            

            while is_running:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    for raw_byte in data:
                        self.parse_byte(raw_byte)
                else:
                    time.sleep(0.002)

        except Exception as e:
            print(f"[IMU {self.name} ERROR]", e)

imu_left = IMUHandler(IMU_LEFT_PORT, IMU_BAUD, "left")
imu_right = IMUHandler(IMU_RIGHT_PORT, IMU_BAUD, "right")

# =========================
# Helpers & Workers
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

def camera_worker(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    print(f"[INFO] Camera {cam_id} started")

    while is_running:
        ret, frame = cap.read()
        if ret:
            ts = time.perf_counter()
            with frame_lock:
                latest_frames[cam_id] = (frame, ts)

    cap.release()

def writer_worker():
    frame_interval = 1.0 / TARGET_FPS
    next_tick = time.perf_counter()

    while is_running:
        if is_recording:
            now = time.perf_counter()
            if now >= next_tick:
                with writer_lock:
                    for cam_id in CAM_IDS:
                        if video_writers[cam_id]:
                            with frame_lock:
                                packet = latest_frames[cam_id]
                            if packet:
                                frame, ts = packet
                                video_writers[cam_id].write(frame)
                                frame_timestamps[cam_id].append(float(ts - TIME_ZERO))
                next_tick += frame_interval
            else:
                time.sleep(0.002)
        else:
            time.sleep(0.01)

# =========================
# Recording Control
# =========================
def start_record(session_dir):
    global TIME_ZERO, is_recording

    os.makedirs(session_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    with writer_lock:
        for cam_id in CAM_IDS:
            logical_name = CAM_MAPPING[cam_id]
            cam_dir = os.path.join(session_dir, logical_name)
            os.makedirs(cam_dir, exist_ok=True)

            video_writers[cam_id] = cv2.VideoWriter(
                os.path.join(cam_dir, "raw_video.mp4"),
                fourcc, TARGET_FPS, (RES_WIDTH, RES_HEIGHT)
            )

            frame_timestamps[cam_id].clear()

    imu_left.imu_file = open(os.path.join(session_dir, "imu_left.jsonl"), "w")
    imu_right.imu_file = open(os.path.join(session_dir, "imu_right.jsonl"), "w")

    TIME_ZERO = time.perf_counter()
    is_recording = True

    print(f"\n[>>> 录制开始 <<<] {os.path.basename(session_dir)}")

def stop_and_save(session_dir):
    global is_recording
    is_recording = False

    imu_left.flush_to_file()
    imu_right.flush_to_file()

    with writer_lock:
        for cam_id in CAM_IDS:
            if video_writers[cam_id]:
                video_writers[cam_id].release()
                video_writers[cam_id] = None

            logical_name = CAM_MAPPING[cam_id]
            with open(os.path.join(session_dir, f"{logical_name}/timestamps.json"), "w") as f:
                json.dump(frame_timestamps[cam_id], f)

    if imu_left.imu_file:
        imu_left.imu_file.close()
        imu_left.imu_file = None

    if imu_right.imu_file:
        imu_right.imu_file.close()
        imu_right.imu_file = None

    metadata = {
        "fps": TARGET_FPS,
        "res": [RES_WIDTH, RES_HEIGHT],
        "imu_hz": 200
    }

    with open(os.path.join(session_dir, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[### 录制保存 ###] {session_dir}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    current_task_name = input("请输入任务名 (demo): ").strip() or "demo"
    base_task_dir = os.path.abspath(current_task_name)
    os.makedirs(base_task_dir, exist_ok=True)

    threading.Thread(target=imu_left.worker, daemon=True).start()
    threading.Thread(target=imu_right.worker, daemon=True).start()

    for cam_id in CAM_IDS:
        threading.Thread(target=camera_worker, args=(cam_id,), daemon=True).start()

    threading.Thread(target=writer_worker, daemon=True).start()

    curr_dir = None
    print("\n双 IMU 同步采集： | 空格:录制 | s:停止 | q:废弃 | ESC:退出")

    try:
        while is_running:
            key = get_char_nonblock()

            if key == ' ':
                if not is_recording:
                    curr_dir = os.path.join(
                        base_task_dir,
                        f"{current_task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    start_record(curr_dir)

            elif key == 's' and is_recording:
                stop_and_save(curr_dir)

            elif key == 'q' and is_recording:
                is_recording = False
                time.sleep(0.1) 
                shutil.rmtree(curr_dir)
                print(f"[已删除废弃数据] {curr_dir}")

            elif key == '\x1b':
                is_running = False
                break

    except KeyboardInterrupt:
        is_running = False