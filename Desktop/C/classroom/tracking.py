import math


class CentroidTracker:
    """Stable student ID assignment by centroid matching."""

    def __init__(self, max_distance=0.15, max_missed_frames=20, min_confirmation_frames=3):
        self.next_id = 0
        self.tracked = {}
        self.missed_frames = {}
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.min_confirmation_frames = min_confirmation_frames
        self.pending = {}
        self.pending_next_key = 0

    def _promote_pending(self, pending_key):
        new_id = self.next_id
        self.next_id += 1
        self.tracked[new_id] = self.pending[pending_key]["centroid"]
        self.missed_frames[new_id] = 0
        del self.pending[pending_key]
        return new_id

    def update(self, detected_centroids):
        assignments = {}
        used_ids = set()

        unmatched = []
        for i, (dx, dy) in enumerate(detected_centroids):
            best_id, best_dist = None, self.max_distance
            for pid, (px, py) in self.tracked.items():
                if pid in used_ids:
                    continue
                dist = math.hypot(dx - px, dy - py)
                if dist < best_dist:
                    best_id, best_dist = pid, dist

            if best_id is not None:
                assignments[i] = best_id
                self.tracked[best_id] = (dx, dy)
                self.missed_frames[best_id] = 0
                used_ids.add(best_id)
            else:
                unmatched.append(i)

        still_unmatched = []
        for i in unmatched:
            dx, dy = detected_centroids[i]
            best_key, best_dist = None, self.max_distance
            for key, info in self.pending.items():
                px, py = info["centroid"]
                dist = math.hypot(dx - px, dy - py)
                if dist < best_dist:
                    best_key, best_dist = key, dist

            if best_key is not None:
                self.pending[best_key]["centroid"] = (dx, dy)
                self.pending[best_key]["count"] += 1
                if self.pending[best_key]["count"] >= self.min_confirmation_frames:
                    new_id = self._promote_pending(best_key)
                    assignments[i] = new_id
                    used_ids.add(new_id)
            else:
                still_unmatched.append(i)

        for i in still_unmatched:
            key = self.pending_next_key
            self.pending_next_key += 1
            self.pending[key] = {"centroid": detected_centroids[i], "count": 1}

        matched_pending_keys = set()
        for i in unmatched:
            dx, dy = detected_centroids[i]
            for key, info in self.pending.items():
                px, py = info["centroid"]
                if math.hypot(dx - px, dy - py) < self.max_distance:
                    matched_pending_keys.add(key)
                    break

        for key in list(self.pending.keys()):
            if key not in matched_pending_keys and self.pending[key]["count"] < self.min_confirmation_frames:
                del self.pending[key]

        for pid in list(self.tracked.keys()):
            if pid not in used_ids:
                self.missed_frames[pid] = self.missed_frames.get(pid, 0) + 1
                if self.missed_frames[pid] > self.max_missed_frames:
                    del self.tracked[pid]
                    del self.missed_frames[pid]

        return assignments


class StudentSuspicionTracker:
    """Suspicion score with phone-in-hand as the dominant signal."""

    def __init__(self, threshold=0.6, required_frames=6):
        self.threshold = threshold
        self.required_frames = required_frames
        self.debounce_counters = {}

    def compute_score(self, head_yaw_deg, torso_lean_deg, object_near_hand):
        score = 0.0
        if head_yaw_deg is not None and abs(head_yaw_deg) > 30:
            score += 0.25
        if torso_lean_deg is not None and abs(torso_lean_deg) > 20:
            score += 0.25
        if object_near_hand:
            score += 0.7
        return min(score, 1.0)

    def update(self, student_id, head_yaw_deg, torso_lean_deg, object_near_hand):
        score = self.compute_score(head_yaw_deg, torso_lean_deg, object_near_hand)
        is_over_threshold = score >= self.threshold

        count = self.debounce_counters.get(student_id, 0)
        count = count + 1 if is_over_threshold else 0
        self.debounce_counters[student_id] = count
        should_alert = count >= self.required_frames
        return score, should_alert


def _global_landmarks(person, full_w, full_h):
    out = []
    crop_offset = person.get("crop_offset", (0, 0))
    crop_size = person.get("crop_size", (1, 1))
    cx, cy = crop_offset
    cw, ch = crop_size
    for lm in person.get("landmarks", []):
        px = cx + (lm["x"] * cw)
        py = cy + (lm["y"] * ch)
        out.append({
            "x": px / full_w,
            "y": py / full_h,
            "z": lm.get("z", 0.0),
            "visibility": lm.get("visibility", 1.0),
        })
    return out


def _head_yaw_deg_from_face(face_landmarks):
    if not face_landmarks or len(face_landmarks) < 455:
        return None
    nose = face_landmarks[1]
    left_cheek = face_landmarks[234]
    right_cheek = face_landmarks[454]
    face_mid_x = (left_cheek["x"] + right_cheek["x"]) / 2.0
    face_width = abs(right_cheek["x"] - left_cheek["x"])
    if face_width < 1e-4:
        return None
    offset = (nose["x"] - face_mid_x) / (face_width / 2.0)
    offset = max(-1.0, min(1.0, offset))
    return offset * 70.0


def _torso_lean_deg(landmarks):
    if len(landmarks) < 25:
        return None
    ls = landmarks[11]
    rs = landmarks[12]
    lh = landmarks[23]
    rh = landmarks[24]
    vis_min = min(ls.get("visibility", 1.0), rs.get("visibility", 1.0), lh.get("visibility", 1.0), rh.get("visibility", 1.0))
    if vis_min < 0.3:
        return None
    shoulder_x = (ls["x"] + rs["x"]) / 2.0
    shoulder_y = (ls["y"] + rs["y"]) / 2.0
    hip_x = (lh["x"] + rh["x"]) / 2.0
    hip_y = (lh["y"] + rh["y"]) / 2.0
    return math.degrees(math.atan2(shoulder_x - hip_x, -(shoulder_y - hip_y)))


def _phone_near_hand(person, phones, full_w, full_h):
    if not person.get("landmarks"):
        return False
    global_landmarks = _global_landmarks(person, full_w, full_h)
    if len(global_landmarks) < 25:
        return False

    hand_points = []
    for idx in (15, 16, 19, 20):
        if idx < len(global_landmarks):
            hand_points.append((global_landmarks[idx]["x"], global_landmarks[idx]["y"]))

    if not hand_points:
        return False

    for phone in phones:
        box = phone["bbox"]
        px1 = box["x_min"] * full_w
        py1 = box["y_min"] * full_h
        px2 = box["x_max"] * full_w
        py2 = box["y_max"] * full_h
        pad = 0.12 * max(full_w, full_h)
        x_min = max(0.0, px1 - pad)
        y_min = max(0.0, py1 - pad)
        x_max = px2 + pad
        y_max = py2 + pad
        for hx, hy in hand_points:
            hx_px = hx * full_w
            hy_px = hy * full_h
            if x_min <= hx_px <= x_max and y_min <= hy_px <= y_max:
                return True
            phone_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
            if math.hypot(hx_px - phone_center[0], hy_px - phone_center[1]) < 0.18 * max(full_w, full_h):
                return True
    return False


def build_student_records(detections, tracker, full_w, full_h):
    people = detections.get("people", [])
    phones = detections.get("phones", [])
    centroids = [person["centroid"] for person in people]
    assignments = tracker.update(centroids)

    records = []
    for det_idx, student_id in assignments.items():
        person = people[det_idx]
        face_landmarks = []
        if person.get("face_landmarks"):
            for lm in person["face_landmarks"]:
                face_landmarks.append({"x": lm["x"], "y": lm["y"], "z": lm.get("z", 0.0)})

        pose_landmarks = _global_landmarks(person, full_w, full_h)
        head_yaw = _head_yaw_deg_from_face(face_landmarks) if face_landmarks else None
        torso_lean = _torso_lean_deg(pose_landmarks) if pose_landmarks else None
        object_near_hand = _phone_near_hand(person, phones, full_w, full_h)

        records.append({
            "student_id": student_id,
            "bbox": person["bbox"],
            "centroid": person["centroid"],
            "head_yaw_deg": head_yaw,
            "torso_lean_deg": torso_lean,
            "object_near_hand": object_near_hand,
        })

    return records
