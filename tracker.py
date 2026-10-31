import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(boxA, boxB):
    """Computes IoU between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    return inter / float(areaA + areaB - inter + 1e-6)


class KalmanBoxTracker:
    """
    Kalman filter tracking bounding box state:
    [x, y, s, r, vx, vy, vs]
    where:
      x, y: center coordinates
      s: scale (area)
      r: aspect ratio (w / h)
      vx, vy, vs: velocity components
    """

    count = 0

    def __init__(self, bbox, class_name, conf, frame_idx):
        # Convert [x1, y1, x2, y2] to [x, y, s, r]
        w = max(1e-2, bbox[2] - bbox[0])
        h = max(1e-2, bbox[3] - bbox[1])
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)

        # State vector
        self.x = np.array([x, y, s, r, 0.0, 0.0, 0.0], dtype=float)

        # Covariance matrix
        self.P = np.diag([10.0, 10.0, 50.0, 1.0, 100.0, 100.0, 10.0])

        # Process noise covariance
        self.Q = np.diag([1.0, 1.0, 10.0, 0.01, 5.0, 5.0, 1.0])

        # Measurement noise covariance
        self.R = np.diag([5.0, 5.0, 20.0, 0.1])

        self.class_name = class_name
        self.conf = conf
        self.peak_conf = conf

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.hits = 1
        self.time_since_update = 0
        self.counted = False
        self.history = [bbox]

    def predict(self):
        """Predicts the next bounding box position based on road motion dynamics."""
        # Forward camera motion prior: objects predictably move downwards (vy > 0) and expand (vs > 0)
        F = np.eye(7)
        F[0, 4] = 1.0  # x += vx
        F[1, 5] = 1.0  # y += vy
        F[2, 6] = 1.0  # s += vs

        self.x = np.dot(F, self.x)
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

        self.time_since_update += 1
        return self.get_bbox()

    def update(self, bbox, conf, frame_idx):
        """Updates the state with a newly detected bounding box."""
        w = max(1e-2, bbox[2] - bbox[0])
        h = max(1e-2, bbox[3] - bbox[1])
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)

        z = np.array([x, y, s, r], dtype=float)

        # Measurement matrix H
        H = np.zeros((4, 7))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0

        y_res = z - np.dot(H, self.x)
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y_res)
        I = np.eye(7)
        self.P = np.dot(I - np.dot(K, H), self.P)

        self.conf = conf
        self.peak_conf = max(self.peak_conf, conf)
        self.last_frame = frame_idx
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)

    def get_bbox(self):
        """Converts [x, y, s, r] state back to [x1, y1, x2, y2]."""
        x, y, s, r = self.x[0], self.x[1], max(1.0, self.x[2]), max(1e-2, self.x[3])
        w = np.sqrt(s * r)
        h = s / w
        return [x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0]


class RoadHazardKalmanTracker:
    """
    Velocity-Predictive Multi-Object Tracker for Road Damage Detection.
    Features:
      - Kalman-filtered motion prediction to handle vehicle speed & road vibration.
      - Hungarian algorithm (scipy linear_sum_assignment) for global bipartite matching.
      - Strict Anti-Double-Count protocol: each physical road defect is assigned
        a unique ID and counted exactly once across the video.
      - Extended occlusion tolerance (coasting) to prevent re-counting after vehicles pass.
    """

    def __init__(self, min_confirmation_hits=2, max_age=30, iou_threshold=0.20):
        self.min_confirmation_hits = min_confirmation_hits
        self.max_age = max_age
        self.iou_threshold = iou_threshold

        self.trackers = []
        self.cumulative_counts = {"pothole": 0, "manhole": 0, "crack": 0}
        self.hazard_registry = {}  # id -> summary dict

    def update(self, detections, frame_idx):
        """
        Args:
          detections: list of dicts [{'bbox': [x1, y1, x2, y2], 'class_name': str, 'conf': float}]
          frame_idx: current video frame integer
        Returns:
          active_tracked_hazards, cumulative_counts
        """
        # Step 1: Predict new locations of all existing trackers
        predicted_boxes = []
        for trk in self.trackers:
            pred_box = trk.predict()
            predicted_boxes.append(pred_box)

        # Step 2: Global association using Hungarian algorithm
        matched_indices, unmatched_dets, unmatched_trks = self._associate_detections(
            detections, predicted_boxes
        )

        # Step 3: Update matched trackers
        for d_idx, t_idx in matched_indices:
            det = detections[d_idx]
            trk = self.trackers[t_idx]
            trk.update(det["bbox"], det["conf"], frame_idx)

            # Confirm and count strictly ONCE
            if trk.hits >= self.min_confirmation_hits and not trk.counted:
                trk.counted = True
                cls = trk.class_name
                self.cumulative_counts[cls] = self.cumulative_counts.get(cls, 0) + 1
                self.hazard_registry[trk.id] = {
                    "id": trk.id,
                    "class_name": cls,
                    "first_frame": trk.first_frame,
                    "last_frame": frame_idx,
                    "peak_conf": trk.peak_conf,
                }
            elif trk.counted:
                self.hazard_registry[trk.id]["last_frame"] = frame_idx
                self.hazard_registry[trk.id]["peak_conf"] = max(
                    self.hazard_registry[trk.id]["peak_conf"], trk.peak_conf
                )

        # Step 4: Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_trk = KalmanBoxTracker(
                bbox=det["bbox"],
                class_name=det["class_name"],
                conf=det["conf"],
                frame_idx=frame_idx,
            )
            if self.min_confirmation_hits <= 1:
                new_trk.counted = True
                cls = new_trk.class_name
                self.cumulative_counts[cls] = self.cumulative_counts.get(cls, 0) + 1
                self.hazard_registry[new_trk.id] = {
                    "id": new_trk.id,
                    "class_name": cls,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "peak_conf": det["conf"],
                }
            self.trackers.append(new_trk)

        # Step 5: Remove expired trackers (lost for > max_age frames)
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # Active hazards in the current frame
        active_in_frame = [
            t for t in self.trackers if t.time_since_update == 0
        ]
        return active_in_frame, self.cumulative_counts

    def _associate_detections(self, detections, predicted_boxes):
        """Bipartite matching between detections and predicted tracker positions."""
        if len(self.trackers) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.trackers)))

        cost_matrix = np.zeros((len(detections), len(self.trackers)), dtype=float)

        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                if det["class_name"] != trk.class_name:
                    cost_matrix[d, t] = 1e5  # Incompatible classes
                    continue

                iou = compute_iou(det["bbox"], predicted_boxes[t])
                
                # Centroid distance normalized by box size
                cx_d = (det["bbox"][0] + det["bbox"][2]) / 2.0
                cy_d = (det["bbox"][1] + det["bbox"][3]) / 2.0
                cx_t = (predicted_boxes[t][0] + predicted_boxes[t][2]) / 2.0
                cy_t = (predicted_boxes[t][1] + predicted_boxes[t][3]) / 2.0
                dist = np.hypot(cx_d - cx_t, cy_d - cy_t)
                scale = np.sqrt(max(1.0, (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])))
                norm_dist = dist / scale

                # Cost metric: 1.0 - IoU + distance penalty
                cost = (1.0 - iou) + 0.3 * min(2.0, norm_dist)
                cost_matrix[d, t] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_indices = []
        unmatched_detections = set(range(len(detections)))
        unmatched_trackers = set(range(len(self.trackers)))

        for r, c in zip(row_ind, col_ind):
            # Verify match threshold
            iou = compute_iou(detections[r]["bbox"], predicted_boxes[c])
            cost = cost_matrix[r, c]

            if cost < 1.5 or iou >= self.iou_threshold:
                matched_indices.append((r, c))
                unmatched_detections.discard(r)
                unmatched_trackers.discard(c)

        return matched_indices, sorted(list(unmatched_detections)), sorted(list(unmatched_trackers))

