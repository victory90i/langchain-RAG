import os
import sys
import time

import cv2

try:
    from .alarm import AlarmController, IncidentLogger, apply_spotlight, draw_dashboard, draw_flag_overlay
    from .detection import ClassroomDetector
    from .stream import VideoStream, choose_stream_source
    from .tracking import CentroidTracker, StudentSuspicionTracker, build_student_records
except ImportError:
    from alarm import AlarmController, IncidentLogger, apply_spotlight, draw_dashboard, draw_flag_overlay
    from detection import ClassroomDetector
    from stream import VideoStream, choose_stream_source
    from tracking import CentroidTracker, StudentSuspicionTracker, build_student_records


def main(source=0):
    stream = VideoStream(source)
    print(f"[INFO] Starting classroom monitoring with source: {source}")
    detector = ClassroomDetector()
    tracker = CentroidTracker(max_distance=0.15, max_missed_frames=20)
    suspicion = StudentSuspicionTracker(threshold=0.6, required_frames=6)
    alarm = AlarmController(sound_path="alarm.mp3", cooldown_sec=2.0)
    logger = IncidentLogger(csv_path="classroom_alerts.csv")

    total_frames = 0
    start = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                break

            total_frames += 1
            timestamp_ms = int(total_frames * (1000 / max(stream.fps, 1.0)))
            detections = detector.process_frame(frame, timestamp_ms)
            records = build_student_records(detections, tracker, frame.shape[1], frame.shape[0])

            flagged_bboxes = []
            alarm_active = False

            for rec in records:
                student_id = rec["student_id"]
                score, should_alert = suspicion.update(
                    student_id=student_id,
                    head_yaw_deg=rec["head_yaw_deg"],
                    torso_lean_deg=rec["torso_lean_deg"],
                    object_near_hand=rec["object_near_hand"],
                )

                bbox = rec["bbox"]
                x1 = int(bbox["x_min"] * frame.shape[1])
                y1 = int(bbox["y_min"] * frame.shape[0])
                x2 = int(bbox["x_max"] * frame.shape[1])
                y2 = int(bbox["y_max"] * frame.shape[0])
                pixel_bbox = [x1, y1, x2, y2]

                if should_alert:
                    alarm_active = True
                    flagged_bboxes.append(pixel_bbox)
                    alarm.trigger()
                    logger.log_incident(student_id, score, rec["head_yaw_deg"], rec["torso_lean_deg"], rec["object_near_hand"])

                draw_flag_overlay(frame, pixel_bbox, student_id, score, is_flagged=should_alert)

            if flagged_bboxes:
                frame = apply_spotlight(frame, flagged_bboxes, dim_factor=0.35)

            elapsed = time.time() - start
            fps = total_frames / max(elapsed, 1e-6)
            draw_dashboard(frame, fps, len(tracker.tracked), alarm_active)

            cv2.imshow("Classroom Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.release()
        detector.close()
        cv2.destroyAllWindows()
        print(f"Processed {total_frames} frames in {time.time() - start:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = choose_stream_source()

    if isinstance(src, str) and os.path.exists(src):
        main(src)
    else:
        main(src)
