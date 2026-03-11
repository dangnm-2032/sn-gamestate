from tracklab.pipeline.detectionlevel_module import DetectionLevelModule
from tracklab.pipeline.videolevel_module import VideoLevelModule
import torch
from typing import Any
import pandas as pd

class PlayerBallAssigner(DetectionLevelModule):
    input_columns = ['bbox_ltwh', 'category_id']
    output_columns = ['ball_control']

    def __init__(
        self,
        cfg,
        batch_size,
        **kwargs
    ):
        super().__init__(batch_size)
        self.cfg = cfg

    def get_center_of_bbox(self, bbox):
        x1, y1, w, h = bbox
        return int(x1 + w / 2), int(y1 + h / 2)

    def measure_distance(self, p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        return {}

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        # Get ball bbox
        ball_bbox = None
        for idx, detection in detections.iterrows():
            if detection['category_id'] == 0:  # Ball category ID
                ball_bbox = detection['bbox_ltwh']
                break
                
        detections['ball_control'] = False
        if ball_bbox is None:
            return detections
        
        miniumum_distance = 99999
        assigned_player = -1

        ball_position = self.get_center_of_bbox(ball_bbox)

        for player_id, detection in detections.iterrows():
            if detection['category_id'] == 0:
                continue

            x, y, l, h = detection['bbox_ltwh']

            distance_left = self.measure_distance((x, y + h), ball_position)
            distance_right = self.measure_distance((x + l, y + h), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.cfg.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        if assigned_player != -1:
            detections.loc[assigned_player, 'ball_control'] = True

        return detections


class TeamBallControl(VideoLevelModule):

    input_columns = ["team", "ball_control"]
    output_columns = ["frame_count", "possession_left", "possession_right"]

    def __init__(self, **kwargs):
        super().__init__()

    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):

        detections["frame_count"] = 0
        detections["possession_left"] = 0
        detections["possession_right"] = 0

        left_frames = 0
        right_frames = 0
        total_frames = 0

        for image_id, frame_det in detections.groupby("image_id"):

            ball = frame_det[frame_det["ball_control"] == True]
            
            if not ball.empty:
                team = ball.iloc[0]["team"]

                total_frames += 1
                if team == "left":
                    left_frames += 1
                elif team == "right":
                    right_frames += 1

            # assign values to all detections of this frame
            detections.loc[frame_det.index, "frame_count"] = total_frames
            detections.loc[frame_det.index, "possession_left"] = left_frames
            detections.loc[frame_det.index, "possession_right"] = right_frames

        # detections.to_csv('/home/yuuhanase/workspace/soccer_analysis/components/sn-gamestate/outputs/data.csv')
        # exit()

        return detections