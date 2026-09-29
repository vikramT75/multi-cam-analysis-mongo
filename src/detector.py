"""
detector.py
───────────
Main execution loop for the Retail Store Intelligence edge node.

Run one instance per camera:
    python src/detector.py --config config.yaml
    python src/detector.py --config config_cam2.yaml
"""

import cv2
import time
import yaml
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from video_streamer import VideoStreamer
from spatial_analytics import ZoneAnalyzer
from journey_tracker import JourneyTracker
from telemetry import TelemetrySender
from reid_extractor import ReIDExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Multi Cam Analysis Edge Node")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to camera configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cam_name      = config["camera"]["name"]
    zone_configs  = config["analytics"]["zones"]
    classes_track = config["analytics"].get("classes_to_track", [0])
    conf_thresh   = config["model"]["confidence_threshold"]
    iou_thresh    = config["model"]["iou_threshold"]

    logger.info("Initialising Multi Cam Analysis node: %s", cam_name)
    logger.info("Zones: %s", [z["name"] for z in zone_configs])

    model     = YOLO(config["model"]["weights"])
    streamer  = VideoStreamer(config["camera"]["source"], cam_name).start()
    telemetry = TelemetrySender()
    analyzer  = ZoneAnalyzer(zone_configs)
    journey   = JourneyTracker([z["name"] for z in zone_configs])
    reid_extractor = ReIDExtractor()

    window_title = f"Multi Cam Analysis - {cam_name}"
    target_frame_time = 1.0 / streamer.fps
    logger.info("Inference started. Press 'q' to quit, 'i' to toggle HUD.")
    first_frame = True
    show_hud = True
    frame_count = 0
    active_signatures = {}

    try:
        while True:
            t0 = time.time()
            frame_count += 1

            frame = streamer.read()
            if frame is None:
                logger.info("Stream ended for %s.", cam_name)
                break

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                classes=classes_track,
                conf=conf_thresh,
                iou=iou_thresh,
            )

            # ReID extraction periodically
            signatures_to_send = {}
            if frame_count % 15 == 0 and results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                embeddings = reid_extractor.extract(frame, boxes)
                for tid, emb in zip(track_ids, embeddings):
                    active_signatures[tid] = emb
                    signatures_to_send[tid] = emb

            annotated = results[0].plot()
            track_zone_map = analyzer.process_tracks(annotated, results)
            journey.update(track_zone_map)
            annotated = analyzer.draw_zones(annotated, show_hud=show_hud)

            cv2.imshow(window_title, annotated)

            if first_frame:
                cv2.setWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
                first_frame = False

            telemetry.send({
                "camera":      cam_name,
                "timestamp":   time.time(),
                "zones":       analyzer.get_zone_states(),
                "track_zones": track_zone_map,
                "funnel":      journey.get_funnel(),
                "transitions": journey.get_transitions(),
                "sankey":      journey.get_sankey_data(),
                "alerts":      analyzer.get_alerts(),
                "signatures":  signatures_to_send,
            })

            elapsed    = time.time() - t0
            sleep_time = target_frame_time - elapsed
            wait_ms    = max(1, int(sleep_time * 1000) if sleep_time > 0 else 1)

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("i"):
                show_hud = not show_hud

    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down %s.", cam_name)
        streamer.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
