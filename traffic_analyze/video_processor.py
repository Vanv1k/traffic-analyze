import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
import psycopg2
from datetime import datetime, timedelta
from typing import Optional, Dict
from collections import deque

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#B97D1B", "#3C76D1"])

class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: Optional[str] = None,
        confidence_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        drone_id: int = 1,
        min_track_length: int = 5,
        db_config: Dict[str, str] = None
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path 
        self.drone_id = drone_id
        self.min_track_length = min_track_length
        self.px_per_meter = 15  # Примерное кол-во пикселей на 1 метр
        self.stopped_speed_threshold = 2.0  # Порог скорости для определения остановки (м/с)
        self.speed_smoothing_window = 5  # Окно для сглаживания скорости
        self.speed_change_threshold = 5.0  # Порог изменения скорости для обновления (км/ч)

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.class_names = self.model.names
        print("Class ID to Name mapping:", self.class_names)

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(color=COLORS, text_color=sv.Color.BLACK)
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )

        self.db_config = db_config
        self.conn = psycopg2.connect(**self.db_config)
        self.cursor = self.conn.cursor()

        self.tracks: Dict[int, dict] = {}
        self.mission_id = self._save_mission_info()

    def _save_mission_info(self) -> int:
        query = """
            INSERT INTO missions (drone_id, video_path, start_time, fps)
            VALUES (%s, %s, %s, %s) RETURNING mission_id;
        """
        self.cursor.execute(query, (self.drone_id, self.target_video_path, datetime.now(), self.video_info.fps))
        self.conn.commit()
        return self.cursor.fetchone()[0]

    def _update_tracks(self, detections: sv.Detections, frame_number: int):
        if len(detections) == 0:
            return

        start_time = datetime.now()
        fps = self.video_info.fps
        
        for tracker_id, class_id, bbox in zip(detections.tracker_id, detections.class_id, detections.xyxy):
            tracker_id = int(tracker_id)
            x, y, x2, y2 = map(int, bbox)
            center_x, center_y = (x + x2) // 2, (y + y2) // 2

            if tracker_id not in self.tracks:
                self.tracks[tracker_id] = {
                    "class_name": self.class_names[int(class_id)],
                    "first_frame": frame_number,
                    "first_bbox": (x, y, x2 - x, y2 - y),
                    "first_time": start_time + timedelta(seconds=frame_number / fps),
                    "frame_count": 1,
                    "last_center": (center_x, center_y),
                    "speed": 0.0,
                    "speed_history": deque(maxlen=self.speed_smoothing_window),
                    "smoothed_speed": 0.0,
                    "last_speed_update": 0.0,
                    "is_stopped": False
                }
            else:
                last_x, last_y = self.tracks[tracker_id]["last_center"]
                pixel_distance = np.sqrt((center_x - last_x) ** 2 + (center_y - last_y) ** 2)
                meter_distance = pixel_distance / self.px_per_meter
                speed_mps = meter_distance * fps 
                speed_kmh = speed_mps * 3.6
                
                # Добавляем текущую скорость в историю
                self.tracks[tracker_id]["speed_history"].append(speed_kmh)
                
                # Сглаживаем скорость по скользящему среднему
                smoothed_speed = np.mean(self.tracks[tracker_id]["speed_history"]) if self.tracks[tracker_id]["speed_history"] else 0
                
                # Обновляем скорость только если изменение значительное
                if abs(smoothed_speed - self.tracks[tracker_id]["last_speed_update"]) > self.speed_change_threshold:
                    self.tracks[tracker_id]["last_speed_update"] = smoothed_speed
                
                self.tracks[tracker_id]["speed"] = speed_mps
                self.tracks[tracker_id]["smoothed_speed"] = smoothed_speed
                self.tracks[tracker_id]["last_center"] = (center_x, center_y)
                self.tracks[tracker_id]["frame_count"] += 1
                
                # Определяем, остановился ли автомобиль
                self.tracks[tracker_id]["is_stopped"] = smoothed_speed < (self.stopped_speed_threshold * 3.6)

    def _save_tracks_to_db(self):
        query = """
            INSERT INTO tracked_objects (
                mission_id, tracker_id, class_name, first_frame_number,
                bbox_x, bbox_y, bbox_width, bbox_height, first_seen_timestamp, avg_speed_kmh
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (mission_id, tracker_id) DO UPDATE 
            SET avg_speed_kmh = EXCLUDED.avg_speed_kmh;
        """
        for tracker_id, track in self.tracks.items():
            if track["frame_count"] >= self.min_track_length:
                x, y, width, height = track["first_bbox"]
                avg_speed = track["last_speed_update"] if track["speed_history"] else 0
                self.cursor.execute(
                    query,
                    (self.mission_id, tracker_id, track["class_name"], track["first_frame"],
                     x, y, width, height, track["first_time"], int(avg_speed)))
        self.conn.commit()

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = frame.copy()
        
        # Создаем метки с ID, классом и скоростью для каждого объекта
        labels = []
        for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
            tracker_id = int(tracker_id)
            speed_text = ""
            if tracker_id in self.tracks:
                speed_text = f" {self.tracks[tracker_id]['last_speed_update']:.1f}km/h"
            labels.append(f"#{tracker_id} {self.class_names[class_id]}{speed_text}")
        
        # Сначала рисуем все боксы
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        
        # Затем рисуем траектории
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        
        # Затем рисуем метки
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        
        # Рисуем остановившиеся автомобили с красным контуром
        for tracker_id, class_id, bbox in zip(detections.tracker_id, detections.class_id, detections.xyxy):
            tracker_id = int(tracker_id)
            if tracker_id in self.tracks and self.tracks[tracker_id]["is_stopped"]:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Статистика по кадру
        vehicle_count = len(detections)
        stopped_vehicles = sum(1 for tid in detections.tracker_id 
                              if int(tid) in self.tracks and self.tracks[int(tid)]["is_stopped"])
        active_speeds = [self.tracks[int(tid)]["last_speed_update"] 
                         for tid in detections.tracker_id if int(tid) in self.tracks]
        
        avg_speed_kmh = np.mean(active_speeds) if active_speeds else 0.0
        
        # Определение уровня загруженности
        if vehicle_count < 5:
            congestion = "Low"
            bg_color = sv.Color.GREEN
        elif vehicle_count <= 15:
            congestion = "Medium"
            bg_color = sv.Color.BLUE
        else:
            congestion = "High"
            bg_color = sv.Color.RED

        # Определение пробки
        traffic_jam = "Jam" if vehicle_count > 10 and avg_speed_kmh < 10 else "No Jam"
        jam_color = sv.Color.RED if traffic_jam == "Jam" else sv.Color.GREEN

        # Смещаем информационный блок правее (x=300 вместо 50)
        info_x = 300
        
        # Рисуем информационные блоки
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Vehicles: {vehicle_count} (Stopped: {stopped_vehicles})",
            text_anchor=sv.Point(x=info_x, y=50),
            background_color=sv.Color.BLACK,
            text_color=sv.Color.WHITE
        )
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Congestion: {congestion}",
            text_anchor=sv.Point(x=info_x, y=80),
            background_color=bg_color,
            text_color=sv.Color.WHITE
        )
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"Traffic: {traffic_jam} (Avg: {avg_speed_kmh:.1f} km/h)",
            text_anchor=sv.Point(x=info_x, y=110),
            background_color=jam_color,
            text_color=sv.Color.WHITE
        )
        
        return annotated_frame

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)
        frame_number = 0

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    frame_number += 1
                    annotated_frame = self.process_frame(frame, frame_number)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                frame_number += 1
                annotated_frame = self.process_frame(frame, frame_number)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

        self._save_tracks_to_db()
        self.conn.close()
        return self.mission_id

    def process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        results = self.model(frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        self._update_tracks(detections, frame_number)
        return self.annotate_frame(frame, detections)