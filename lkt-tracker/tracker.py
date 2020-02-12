import os

import click
import cv2
import numpy as np

DEFAULT_DRAW_COLOR = (0, 255, 0)
DEFAULT_FEATURES_PARAMS = dict(
    maxCorners=20, useHarrisDetector=True, qualityLevel=0.01, minDistance=10,
)

DEFAULT_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


@click.command()
@click.argument("input_video")
@click.option("--output-dir", "-o", type=str, help="Output folder")
def lkt_tracker(input_video: str, output_dir: str) -> None:
    video = cv2.VideoCapture(input_video)

    success, frame = video.read()
    previous_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    previous_points = cv2.goodFeaturesToTrack(
        previous_frame_gray, **DEFAULT_FEATURES_PARAMS
    )

    while 1:
        success, frame = video.read()
        if not success:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_points, st, _ = cv2.calcOpticalFlowPyrLK(
            previous_frame_gray, frame_gray, previous_points, None, **DEFAULT_LK_PARAMS
        )

        good_new = new_points[st == 1]
        good_old = previous_points[st == 1]

        previous_frame_gray = frame_gray.copy()
        previous_points = good_new.reshape(-1, 1, 2)

    video.release()


if __name__ == "__main__":
    lkt_tracker()
