import os
from pathlib import Path

import cv2
import mediapipe as mp
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


class ClassroomDetector:
    """YOLO + MediaPipe detector for people and phones."""

    def __init__(
        self,
        yolo_model_name="yolov8n.pt",
        pose_model_name="pose_landmarker_lite.task",
        face_model_name="face_landmarker.task",
        confidence_threshold=0.35,
        yolo_input_width=640,
        crop_padding=0.08,
        crop_size_px=256,
        landmark_every_n_frames=2,
    ):
        self.confidence_threshold = confidence_threshold
        self.crop_padding = crop_padding
        self.yolo_input_width = yolo_input_width
        self.crop_size_px = crop_size_px
        self.landmark_every_n_frames = landmark_every_n_frames
        self._frame_count = 0

        self.yolo = YOLO(str(ROOT / yolo_model_name))

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ROOT / pose_model_name)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_detector = PoseLandmarker.create_from_options(pose_options)

        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ROOT / face_model_name)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_detector = FaceLandmarker.create_from_options(face_options)

    def _crop_with_padding(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * self.crop_padding)
        pad_y = int((y2 - y1) * self.crop_padding)
        cx1 = max(int(x1) - pad_x, 0)
        cy1 = max(int(y1) - pad_y, 0)
        cx2 = min(int(x2) + pad_x, w)
        cy2 = min(int(y2) + pad_y, h)
        if cx2 <= cx1 or cy2 <= cy1:
            return None, (0, 0)
        return frame[cy1:cy2, cx1:cx2], (cx1, cy1)

    def _run_landmarks(self, crop, crop_ts):
        h, w = crop.shape[:2]
        if w > self.crop_size_px:
            scale = self.crop_size_px / w
            crop = cv2.resize(crop, (self.crop_size_px, int(h * scale)), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_out = []
        face_out = []

        try:
            pose_result = self.pose_detector.detect_for_video(mp_image, crop_ts)
            if pose_result and pose_result.pose_landmarks:
                for lm in pose_result.pose_landmarks[0]:
                    pose_out.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": getattr(lm, "visibility", 1.0),
                    })
        except Exception:
            pass

        try:
            face_result = self.face_detector.detect_for_video(mp_image, crop_ts)
            if face_result and face_result.face_landmarks:
                for lm in face_result.face_landmarks[0]:
                    face_out.append({"x": lm.x, "y": lm.y, "z": lm.z})
        except Exception:
            pass

        return pose_out, face_out

    def process_frame(self, frame_bgr, timestamp_ms):
        self._frame_count += 1
        do_landmarks = self._frame_count % self.landmark_every_n_frames == 0
        h_full, w_full = frame_bgr.shape[:2]

        if w_full > self.yolo_input_width:
            scale = self.yolo_input_width / w_full
            yolo_input = cv2.resize(
                frame_bgr,
                (self.yolo_input_width, int(h_full * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            yolo_input = frame_bgr

        results = self.yolo(yolo_input, conf=self.confidence_threshold, classes=[0, 67], verbose=False)[0]
        yolo_h, yolo_w = yolo_input.shape[:2]

        people = []
        phones = []

        for det_idx, box in enumerate(results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0].item()) if box.cls is not None else 0

            x1n = x1 * w_full / yolo_w
            x2n = x2 * w_full / yolo_w
            y1n = y1 * h_full / yolo_h
            y2n = y2 * h_full / yolo_h

            bbox = {
                "x_min": x1n / w_full,
                "y_min": y1n / h_full,
                "x_max": x2n / w_full,
                "y_max": y2n / h_full,
                "confidence": conf,
            }

            crop, (cx1, cy1) = self._crop_with_padding(frame_bgr, x1n, y1n, x2n, y2n)
            if crop is None or crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]
            crop_ts = timestamp_ms + det_idx
            if do_landmarks:
                pose_lm, face_lm = self._run_landmarks(crop, crop_ts)
            else:
                pose_lm, face_lm = [], []

            centroid = ((x1n + x2n) / (2.0 * w_full), (y1n + y2n) / (2.0 * h_full))

            if cls_id == 0:
                people.append({
                    "bbox": bbox,
                    "centroid": centroid,
                    "landmarks": pose_lm,
                    "face_landmarks": face_lm,
                    "crop_offset": (cx1, cy1),
                    "crop_size": (crop_w, crop_h),
                })
            elif cls_id == 67:
                phones.append({
                    "bbox": bbox,
                    "centroid": centroid,
                    "crop_offset": (cx1, cy1),
                    "crop_size": (crop_w, crop_h),
                })

        return {"people": people, "phones": phones}

    def close(self):
        self.pose_detector.close()
        self.face_detector.close()
