import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    return inter / float(areaA + areaB - inter + 1e-6)


class TrackedHazard:
    def __init__(self, hazard_id, class_name, bbox, conf, frame_idx):
        self.id = hazard_id
        self.class_name = class_name
        self.bbox = bbox
        self.conf = conf
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.hits = 1
        self.counted = False

    @property
    def centroid(self):
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )


class RoadHazardTracker:
    def __init__(self, min_confirmation_frames=2, max_disappeared=20):
        self.min_confirmation_frames = min_confirmation_frames
        self.max_disappeared = max_disappeared
        self.next_hazard_id = 1
        self.active_tracks = {}
        self.cumulative_counts = {"pothole": 0, "manhole": 0, "crack": 0}

    def update(self, detections, frame_idx):
        matched_track_ids = set()
        unmatched = []

        for det in detections:
            d_bbox = det["bbox"]
            d_cls = det["class_name"]
            d_conf = det["conf"]
            d_cx = (d_bbox[0] + d_bbox[2]) / 2.0
            d_cy = (d_bbox[1] + d_bbox[3]) / 2.0

            best_tid = None
            best_iou = 0.0
            best_dist = float("inf")

            for tid, track in self.active_tracks.items():
                if tid in matched_track_ids or track.class_name != d_cls:
                    continue

                iou = compute_iou(track.bbox, d_bbox)
                tcx, tcy = track.centroid
                dist = np.hypot(d_cx - tcx, d_cy - tcy)
                box_diag = np.hypot(track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1])

                if iou >= 0.20 and iou > best_iou:
                    best_iou = iou
                    best_tid = tid
                elif best_iou < 0.20 and dist < max(60.0, box_diag * 1.5) and dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                t = self.active_tracks[best_tid]
                t.bbox = d_bbox
                t.conf = d_conf
                t.last_frame = frame_idx
                t.hits += 1
                matched_track_ids.add(best_tid)

                if t.hits >= self.min_confirmation_frames and not t.counted:
                    t.counted = True
                    self.cumulative_counts[t.class_name] = (
                        self.cumulative_counts.get(t.class_name, 0) + 1
                    )
            else:
                unmatched.append(det)

        for det in unmatched:
            new_track = TrackedHazard(
                hazard_id=self.next_hazard_id,
                class_name=det["class_name"],
                bbox=det["bbox"],
                conf=det["conf"],
                frame_idx=frame_idx,
            )
            if self.min_confirmation_frames <= 1:
                new_track.counted = True
                self.cumulative_counts[new_track.class_name] = (
                    self.cumulative_counts.get(new_track.class_name, 0) + 1
                )
            self.active_tracks[self.next_hazard_id] = new_track
            self.next_hazard_id += 1

        expired = [
            tid
            for tid, track in self.active_tracks.items()
            if frame_idx - track.last_frame > self.max_disappeared
        ]
        for tid in expired:
            del self.active_tracks[tid]

        visible = [t for t in self.active_tracks.values() if t.last_frame == frame_idx]
        return visible, self.cumulative_counts


def find_default_model():
    candidates = [
        PROJECT_DIR / "best.pt",
        PROJECT_DIR / "runs" / "road_damage_baseline" / "weights" / "best_640.pt",
        PROJECT_DIR / "runs" / "road_damage_2class_640" / "weights" / "best.pt",
        PROJECT_DIR / "yolo11n.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return PROJECT_DIR / "best.pt"


def normalize_hazard_class(raw_name: str) -> str:
    name = str(raw_name).lower().strip()
    if any(k in name for k in ["pothole", "d40", "rutting", "bump"]):
        return "pothole"
    elif any(k in name for k in ["manhole", "d70", "drain", "utility"]):
        return "manhole"
    elif any(k in name for k in ["crack", "d00", "d10", "d20", "fissure", "longitudinal", "transverse", "alligator"]):
        return "crack"
    elif "other" in name:
        return "pothole"
    return name


def find_test_videos():
    tests_dir = PROJECT_DIR / "tests"
    if not tests_dir.is_dir():
        return []
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return [p for p in sorted(tests_dir.iterdir()) if p.suffix.lower() in exts]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Road Damage Detection Player with Cumulative Tracking"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(find_default_model()),
        help="Path to YOLO model weights (.pt). Default: best.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Video file path. Default: plays videos from tests/",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Initial confidence threshold (default: 0.20). Adjust with +/- keys.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference frame size (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device to run on (e.g., '0' for GPU, 'cpu').",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated video output to runs/player_recordings/",
    )
    return parser.parse_args()


def play_video(video_path, model, args, current_conf):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {video_path}")
        return current_conf, False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay = max(1, int(1000 / orig_fps))

    video_name = Path(video_path).name
    window_name = "Road Damage Detection - Unique Hazard Tracker"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 600)

    writer = None
    if args.save:
        out_dir = PROJECT_DIR / "runs" / "player_recordings"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"tracked_{Path(video_path).stem}.mp4"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, orig_fps, (w, h))

    tracker = RoadHazardTracker(min_confirmation_frames=2, max_disappeared=int(orig_fps * 1.5))

    frame_idx = 0
    paused = False
    prev_time = time.time()
    fps = orig_fps

    color_map = {
        "pothole": (40, 50, 230),   # Red / Coral
        "manhole": (230, 150, 20),  # Blue / Cyan
        "crack": (30, 215, 255),    # Yellow
    }

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = model.predict(
                source=frame,
                conf=current_conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                raw_name = model.names.get(cls_id, "")
                c_name = normalize_hazard_class(raw_name)

                detections.append(
                    {
                        "bbox": box.xyxy[0].tolist(),
                        "class_name": c_name,
                        "conf": float(box.conf[0].item()),
                    }
                )

            visible_hazards, cumulative = tracker.update(detections, frame_idx)

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Draw boxes with persistent IDs
            annotated_frame = frame.copy()
            for hazard in visible_hazards:
                x1, y1, x2, y2 = [int(v) for v in hazard.bbox]
                col = color_map.get(hazard.class_name, (0, 255, 0))

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), col, 2)
                badge_text = f"#{hazard.id} {hazard.class_name.capitalize()} {hazard.conf:.0%}"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(
                    annotated_frame,
                    (x1, max(0, y1 - th - 8)),
                    (x1 + tw + 6, y1),
                    col,
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    badge_text,
                    (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Draw top HUD
            h_img, w_img = annotated_frame.shape[:2]
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w_img, 65), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.78, annotated_frame, 0.22, 0, annotated_frame)

            total_u = sum(cumulative.values())
            hud_title = f"{video_name} | Frame {frame_idx}/{total_frames} | {fps:.1f} FPS | Conf: {current_conf:.2f}"
            cv2.putText(
                annotated_frame,
                hud_title,
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            hud_counts = (
                f"UNIQUE CUMULATIVE -> Potholes: {cumulative.get('pothole', 0)} | "
                f"Manholes: {cumulative.get('manhole', 0)} | "
                f"Cracks: {cumulative.get('crack', 0)} | "
                f"Total Defects: {total_u}"
            )
            cv2.putText(
                annotated_frame,
                hud_counts,
                (15, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 200),
                2,
                cv2.LINE_AA,
            )

            if writer:
                writer.write(annotated_frame)

            display_frame = annotated_frame
        else:
            display_frame = annotated_frame.copy()
            cv2.putText(
                display_frame,
                "PAUSED - Press SPACE to Resume",
                (display_frame.shape[1] // 2 - 200, display_frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, display_frame)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(delay if not paused else 50) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            return current_conf, True
        elif key == 32:
            paused = not paused
        elif key in (ord("+"), ord("=")):
            current_conf = min(0.95, round(current_conf + 0.05, 2))
        elif key in (ord("-"), ord("_")):
            current_conf = max(0.05, round(current_conf - 0.05, 2))
        elif key in (ord("r"), ord("R")):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            tracker = RoadHazardTracker(min_confirmation_frames=2, max_disappeared=int(orig_fps * 1.5))
            paused = False
        elif key in (ord("n"), ord("N")):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    return current_conf, False


def main():
    args = parse_args()
    model_path = Path(args.weights)
    if not model_path.is_file():
        print(f"Error: Weights file not found: {model_path}")
        sys.exit(1)

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(str(model_path))

    playlist = [args.source] if args.source else find_test_videos()
    if not playlist:
        print("No test videos found in tests/ folder.")
        sys.exit(1)

    current_conf = args.conf
    for video_path in playlist:
        current_conf, should_exit = play_video(video_path, model, args, current_conf)
        if should_exit:
            break


if __name__ == "__main__":
    main()
