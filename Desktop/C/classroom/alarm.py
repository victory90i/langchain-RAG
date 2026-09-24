import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

ROOT = Path(__file__).resolve().parent


class IncidentLogger:
    def __init__(self, csv_path="classroom_alerts.csv"):
        self.csv_path = ROOT / csv_path
        self.last_logged = {}
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "student_id", "suspicion_score", "head_yaw_deg", "torso_lean_deg", "object_near_hand"])

    def log_incident(self, student_id, score, head_yaw, torso_lean, object_near):
        now = datetime.now()
        now_ts = now.timestamp()
        if student_id in self.last_logged and (now_ts - self.last_logged[student_id]) < 5.0:
            return False
        self.last_logged[student_id] = now_ts
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime("%Y-%m-%d %H:%M:%S"),
                student_id,
                round(float(score), 3),
                round(float(head_yaw), 1) if head_yaw is not None else "N/A",
                round(float(torso_lean), 1) if torso_lean is not None else "N/A",
                bool(object_near),
            ])
        print(f"[CSV LOGGED] Student #{student_id} at {now.strftime('%Y-%m-%d %H:%M:%S')}")
        return True


class AlarmController:
    """Audio alarm + dashboard overlay."""

    def __init__(self, sound_path="alarm.mp3", cooldown_sec=2.0):
        self.sound_path = ROOT / sound_path
        self.cooldown_sec = cooldown_sec
        self.last_played_time = 0.0
        self.audio_ready = False
        self.sound = None

        if pygame is not None and self.sound_path.exists():
            try:
                pygame.mixer.init()
                self.sound = pygame.mixer.Sound(str(self.sound_path))
                self.audio_ready = True
                print(f"[ALARM] Real audio ready: {self.sound_path}")
            except Exception as exc:
                print(f"[ALARM WARN] Audio init failed: {exc}")
        elif not self.sound_path.exists():
            print(f"[ALARM WARN] Sound file missing: {self.sound_path}")
        else:
            print("[ALARM WARN] pygame is not installed; using console fallback")

    def trigger(self):
        now = time.time()
        if now - self.last_played_time < self.cooldown_sec:
            return
        self.last_played_time = now
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self):
        if self.audio_ready and self.sound is not None:
            try:
                self.sound.play()
            except Exception as exc:
                print(f"[ALARM ERROR] {exc}")
        else:
            print("\a[ALARM BEEP] Suspicious behavior detected")


def draw_flag_overlay(frame, bbox, student_id, score, is_flagged):
    x1, y1, x2, y2 = bbox
    color = (0, 0, 255) if is_flagged else (0, 255, 0)
    thickness = 3 if is_flagged else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    status = "FLAGGED" if is_flagged else "NORMAL"
    label = f"ID #{student_id} | {status} ({score:.2f})"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    lbl_w, lbl_h = label_size
    lbl_y1 = max(0, y1 - lbl_h - 10)
    cv2.rectangle(frame, (x1, lbl_y1), (x1 + lbl_w + 10, lbl_y1 + lbl_h + 8), color, -1)
    cv2.putText(frame, label, (x1 + 5, lbl_y1 + lbl_h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def apply_spotlight(frame, flagged_bboxes, dim_factor=0.35):
    if not flagged_bboxes:
        return frame
    mask = np.zeros_like(frame, dtype=np.uint8)
    for x1, y1, x2, y2 in flagged_bboxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    dimmed = (frame * dim_factor).astype(np.uint8)
    return np.where(mask == 255, frame, dimmed)


def draw_dashboard(frame, fps, student_count, alarm_active):
    h, w, _ = frame.shape
    hud = frame[0:40, 0:w]
    frame[0:40, 0:w] = (hud * 0.2).astype(np.uint8)
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Students: {student_count}", (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    status = "ALARM ACTIVE" if alarm_active else "SYSTEM SECURE"
    color = (0, 0, 255) if alarm_active else (0, 255, 0)
    cv2.putText(frame, f"Status: {status}", (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
