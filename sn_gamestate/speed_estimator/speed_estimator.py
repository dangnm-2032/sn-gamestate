from tracklab.pipeline.videolevel_module import VideoLevelModule
import torch
from typing import Any
import pandas as pd

class SpeedEstimator(VideoLevelModule):
    input_columns = ['bbox_pitch', 'track_id']
    output_columns = ['speed']

    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__()
        self.cfg = cfg

    def measure_distance(self, p1, p2, width_scale, height_scale):
        return (abs(p1[0]-p2[0])**2 * width_scale**2 + abs(p1[1]-p2[1])**2 * height_scale**2)**0.5

    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        prev_image_id = None
        detections['speed'] = None
        for image_id, frame_det in detections.groupby("image_id"):
            if not prev_image_id:
                prev_image_id = image_id
                continue

            # TODO: calculate speed for each player
            # TODO: use prev_frame to calculate speed
            for idx, row in frame_det.iterrows():
                player_id = row['track_id']
                player_pos = (row['bbox_pitch']['x_bottom_middle'], row['bbox_pitch']['y_bottom_middle'])
                prev_frame_det = detections[detections['image_id'] == prev_image_id]
                if player_id not in prev_frame_det['track_id'].values:
                    continue
                prev_player_pos = prev_frame_det[prev_frame_det['track_id'] == player_id]['bbox_pitch'].iloc[0]
                if not prev_player_pos:
                    continue
                prev_player_pos = (prev_player_pos['x_bottom_middle'], prev_player_pos['y_bottom_middle'])
                distance = self.measure_distance(player_pos, prev_player_pos, self.cfg.width_scale, self.cfg.height_scale)
                # TODO: calculate speed
                speed = distance * self.cfg.fps 
                detections.loc[idx, 'speed'] = speed

            prev_image_id = image_id

        return detections
