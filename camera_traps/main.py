import logging

from camera_traps.motion_detection import gui
from camera_traps.motion_detection.capture_motion import detect_motion_on_fixed_video

app = gui.MenuGUI()
app.mainloop()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    box_detection = detect_motion_on_fixed_video(input_video_path=app.videoLoc,
                                                 input_background_path=app.backgroundLoc,
                                                 area_filer_out=3000,
                                                 weights_path=app.modelFolderLoc,
                                                 tracked_prediction=app.trackingLoc,
                                                 score_filter_out=app.thresholdLoc)

    # Drop unuseful information for analysis.
    box_detection.drop(["color_label", "color_track", "centroid", "box"], axis=1, inplace=True)
    print(box_detection)
