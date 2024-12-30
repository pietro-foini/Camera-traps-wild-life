from typing import Union, Optional
import pickle
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from camera_traps.motion_detection.geometry_utils import compose_polygon, get_bbox_without_intersection, expand_bbox
from camera_traps.motion_detection.tracking_objects import tracking
from camera_traps.model.model import efficientnet_b0

cmap = plt.get_cmap("Set1")


def get_video_properties(video_path: str) -> tuple[cv2.VideoCapture, int, int, int]:
    """
    Open video file using OpenCV and get some properties like number of frames and the corresponding width and height.

    :param video_path: the path to video file (e.g. .mp4, .avi, etc.)
    :return: the opened OpenCV video, the number of frames, the width and the height of the frames
    """
    # Open video.
    video = cv2.VideoCapture(video_path)
    # Get video properties.
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return video, fps, width, height


def get_background(video_path: str, n_frames: int = 50) -> np.array:
    """
    Retrieves the background frame from a video by calculating the median of randomly selected frames.

    :param video_path: path to the video file
    :param n_frames: number of frames to randomly select for calculating the median
    :return: the computed background frame
    """
    # Open video.
    video = cv2.VideoCapture(video_path)
    # We will randomly select some frames for the calculating the median.
    frame_indices = video.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=n_frames)
    # We will store the frames in array.
    frames = []
    for idx in frame_indices:
        # Set the frame id to read that particular frame.
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if frame is not None:
            frames.append(frame)
    # Calculate the median.
    median_frame = np.median(frames, axis=0).astype(np.uint8)

    return median_frame


def get_pixel_difference(frame1: np.ndarray, frame2: np.ndarray) -> list[np.ndarray]:
    """
    Calculate pixel difference between two input images.

    :param frame1: the first input image
    :param frame2: the second input image
    :return: list of contours representing the pixel differences between the two images
    """
    frame = cv2.absdiff(frame1, frame2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    _, thresh = cv2.threshold(frame, 20, 255, cv2.THRESH_BINARY)
    # Dilate the threshold image to fill in holes, then find contours on threshold image.
    thresh = cv2.dilate(thresh, None, iterations=2)
    # Get all the found contours.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def get_color_by_label_or_index(label: Union[int, str]) -> tuple[int, ...]:
    """
    Return a color based on some predefined label's name or index conditions.

    :param label: the provided label or index
    :return: the color associated to the provided label or index
    """
    if isinstance(label, int):
        return tuple([int(c * 255) for c in cmap.colors[label % 9]])
    elif isinstance(label, str):
        if label == "human":
            return 0, 255, 0
        elif label == "vehicle":
            return 0, 0, 255
        elif label in ["cat", "dog"]:
            return 255, 165, 0
        else:
            return 255, 0, 0


def get_primary_label(labels: pd.Series, min_count: int = 3, min_occurrence: float = 25):
    """

    :param labels: a pandas Series containing labels string, ideally for a unique tracked object
    :param min_count: the minimum number of labels count of the tracked object to be considered valid
    :param min_occurrence:
    :return:
    """
    occurrences = labels.value_counts(normalize=True, dropna=False).mul(100)
    if len(labels) >= min_count and occurrences.max() > min_occurrence:
        return occurrences.idxmax()
    else:
        return "None_of_the_above"


def detect_motion_on_fixed_video(input_video_path: str, input_background_path: Optional[str] = None,
                                 area_filer_out: int = 3000, weights_path: Optional[str] = None,
                                 score_filter_out: float = 95, tracked_prediction: bool = True,
                                 output_video_path: Optional[str] = "output.mp4") -> pd.DataFrame:
    """
    Detect motion searching difference between current frame and a provided background or an average frame along
    all video. The bounding boxes that identify a motion are given as input to the prediction model in order to
    classify them.

    # TODO: Change model: improve time consuming performances.
    # TODO: Change approach for detecting motion and corresponding bounding boxes.

    :param input_video_path: the path to input video file (e.g. .mp4, .avi, etc.)
    :param input_background_path: the path to the input background image; if not provided a background is
        automatically computed averaging frames along the provided video
    :param area_filer_out: the bounding boxes' areas that will not be considered for motion detection if smaller
    :param weights_path: the path to the weights of the model for predicting the detected bounding boxes; it must
        contain at least two files: 'weights.h5' and 'labels'
    :param score_filter_out: the model scores that will not be considered for output predictions if smaller
    :param tracked_prediction: if activated an algorithm tracks the detected objected over time along the video
    :param output_video_path: the path to output file (.mp4)
    :return:
    """
    # Open video.
    video, fps, width, height = get_video_properties(video_path=input_video_path)

    # Create background.
    if input_background_path:
        background = cv2.imread(input_background_path)
    else:
        background = get_background(video_path=input_video_path)

    t1 = time.time()

    num_frames = 0
    list_frames = list()
    ids, bboxes, coordinates = list(), list(), list()
    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        # Get difference between current frame and background image.
        contours = get_pixel_difference(background, frame)
        # Filter out based on area limit and get polygons.
        polygons = [compose_polygon(*cv2.boundingRect(c)) for c in contours if cv2.contourArea(c) > area_filer_out]
        # Get polygons without intersection.
        bbox_coordinates = get_bbox_without_intersection(polygons)

        # Loop over the box coordinates.
        for x, y, w, h in bbox_coordinates:
            new_x, new_y, new_w, new_h = expand_bbox(x, y, w, h, percentage=0)
            # Get bounding box image.
            box = frame[new_y:new_y + new_h, new_x:new_x + new_w]
            box = cv2.cvtColor(box, cv2.COLOR_BGR2RGB)
            box = cv2.resize(box, (128, 128))
            # Store bounding boxes data.
            ids.append(num_frames)
            bboxes.append(box)
            coordinates.append((new_x, new_y, new_w, new_h))

        num_frames += 1

        # Add frame.
        list_frames.append(frame)

    # Store results.
    box_detection = pd.DataFrame({"id_frame": ids,
                                  "box": coordinates,
                                  "score": None,
                                  "label": None,
                                  "color_label": None,
                                  "centroid": None,
                                  "track_index": None,
                                  "color_track": None})

    if box_detection.empty:
        return box_detection

    if weights_path:
        # Load pretrained model.
        with open(f"{weights_path}/labels", "rb") as fp:
            labels = pickle.load(fp)
        trained_model = efficientnet_b0(num_classes=len(labels), input_shape=(128, 128, 3))
        trained_model.load_weights(f"{weights_path}/weights.h5")

        # Get predictions.
        predictions = trained_model.predict(np.stack(bboxes, axis=0), batch_size=32, verbose=1)
        # Get the best predictions.
        box_detection["score"] = np.max(predictions, axis=1)
        box_detection["label"] = np.argmax(predictions, axis=1)
        # Postprocessing.
        labels = {i: label for i, label in enumerate(labels)}
        box_detection["score"] = box_detection["score"].round(2) * 100
        box_detection["label"] = box_detection["label"].map(labels)
        # Filter on prediction score.
        box_detection.loc[box_detection["score"] < score_filter_out, "label"] = "None_of_the_above"

    # Tracking objects based on centroid movement distance.
    if tracked_prediction:
        box_detection["centroid"] = box_detection["box"].apply(lambda coord: compose_polygon(*coord).centroid)
        box_detection["track_index"] = tracking(box_detection, distance_limit=30)
        box_detection["color_track"] = box_detection["track_index"].apply(get_color_by_label_or_index)
        box_detection["label"] = box_detection.groupby("track_index")["label"].transform(get_primary_label)

    # Set color based on label.
    box_detection["color_label"] = box_detection["label"].apply(get_color_by_label_or_index)

    t2 = time.time()

    logging.info(f"Total elapsed time: {t2 - t1} s")

    if not output_video_path:
        return box_detection

    # Create output video containing motion detection and related predictions.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(width, height))

    for i, frame in enumerate(list_frames):
        frame_objects_detection = box_detection.query("id_frame == @i & label != 'None_of_the_above'")
        for index, row in frame_objects_detection.iterrows():
            x, y, w, h = row["box"]
            # Draw bounding box rectangle.
            cv2.rectangle(frame, (x, y), (x + w, y + h), row["color_label"] or row["color_track"], 2)
            # Define the text to write over the bounding boxes.
            text = f"{row['label']}: {'{:.2f}'.format(row['score'])}%" if weights_path else "n/a"
            # Write label and percentage prediction score for the current bounding box.
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            frame = cv2.rectangle(frame, (x, y - 20), (x + w, y), row["color_label"] or row["color_track"], -1)
            frame = cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Write modified frame.
        output_video.write(frame)

    cv2.destroyAllWindows()
    output_video.release()

    logging.info("Bye...")

    return box_detection
