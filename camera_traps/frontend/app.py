import os
import tempfile

import av
import pandas as pd
import requests
import streamlit as st
import supervision as sv
from PIL import Image

from camera_traps.settings import S

st.set_page_config(page_title="Camera Traps Wildlife", layout="wide")
st.title("🐾 Camera Traps Wildlife Detector")

# Sidebar mode selector
option = st.sidebar.selectbox("Select Mode", ["Image Classification", "Video Detection"])

if option == "Image Classification":
    st.header("Image Classification")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        col1, col2 = st.columns(2)

        # Display original image.
        image = Image.open(uploaded_image)
        col1.image(image, caption="Uploaded Image", width="stretch")

        if col1.button("Analyze Image"):
            files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}

            with st.spinner("Analyzing image (this may take a few minutes)..."):
                try:
                    response = requests.post(f"{S.API_URL}/image/predict-image", files=files)

                    if response.status_code == 200:
                        data = response.json()
                        col2.success("Processing complete!")

                        # Display prediction results
                        col2.subheader("Results:")
                        for pred in data.get("predictions", []):
                            col2.write(f"**Class:** {pred['class_name']} — **Confidence:** {pred['confidence']:.2%}")
                    else:
                        col2.error(f"API Error: {response.status_code} - {response.text}")
                except Exception as e:
                    col2.error(f"Failed to connect to FastAPI server: {e}")

elif option == "Video Detection":
    st.header("Video Detection")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        col1, col2 = st.columns(2)

        # Display the original output video in Streamlit.
        with col1:
            st.subheader("Original Video")
            st.video(uploaded_video)

        if st.button("Analyze Video"):
            files = {"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)}

            with st.spinner("1/2: Fetching predictions..."):
                try:
                    response = requests.post(f"{S.API_URL}/video/predict-video", files=files)
                except Exception as e:
                    st.error(f"Failed to connect to FastAPI server: {e}")
                    st.stop()

            if response.status_code != 200:
                st.error(f"API Error {response.status_code}: {response.text}")
                st.stop()

            data = response.json()
            predictions = data.get("predictions", [])

            if not predictions:
                st.info("No wildlife detected in the video.")
                st.stop()

            with st.spinner("2/2: Annotating video frames..."):
                flattened_predictions = []
                for item in predictions:
                    flattened_predictions.append(
                        {
                            "class_name": item["class_name"],
                            "tracking_id": item["tracking_id"],
                            "confidence": item["confidence"],
                            "xmin": item["box"]["xmin"],
                            "ymin": item["box"]["ymin"],
                            "xmax": item["box"]["xmax"],
                            "ymax": item["box"]["ymax"],
                            "frame_id": item["frame_id"],
                        }
                    )
                df = pd.DataFrame(flattened_predictions)

                # Save uploaded file to a temporary location.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                    tmp_in.write(uploaded_video.getvalue())
                    tmp_in_path = tmp_in.name

                output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

                try:
                    # Open video with PyAV.
                    in_container = av.open(tmp_in_path)
                    in_stream = in_container.streams.video[0]

                    # Setup output video writer.
                    out_container = av.open(output_path, mode="w")
                    out_stream = out_container.add_stream("h264", rate=in_stream.average_rate or 30)
                    out_stream.width = in_stream.codec_context.width
                    out_stream.height = in_stream.codec_context.height
                    out_stream.pix_fmt = "yuv420p"

                    # Setup annotators.
                    box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
                    label_annotator = sv.LabelAnnotator(
                        text_scale=0.6, text_thickness=1, color_lookup=sv.ColorLookup.TRACK
                    )

                    # Iterate over frames and apply annotations.
                    for frame_id, frame in enumerate(in_container.decode(video=0)):
                        frame_bgr = frame.to_ndarray(format="bgr24")
                        df_frame = df[df["frame_id"] == frame_id]

                        if not df_frame.empty:
                            labels = [
                                f"{lbl} #{tid}" for lbl, tid in zip(df_frame["class_name"], df_frame["tracking_id"])
                            ]
                            detections = sv.Detections(
                                xyxy=df_frame[["xmin", "ymin", "xmax", "ymax"]].to_numpy(),
                                tracker_id=df_frame["tracking_id"].to_numpy(),
                            )
                            frame_bgr = box_annotator.annotate(scene=frame_bgr, detections=detections)
                            frame_bgr = label_annotator.annotate(scene=frame_bgr, detections=detections, labels=labels)

                        out_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
                        for packet in out_stream.encode(out_frame):
                            out_container.mux(packet)

                    for packet in out_stream.encode():
                        out_container.mux(packet)

                    in_container.close()
                    out_container.close()

                    # Display the annotated output video in Streamlit.
                    with col2:
                        st.subheader("Annotated Video")
                        with open(output_path, "rb") as video_file:
                            st.video(video_file.read())

                    st.divider()
                    st.subheader("Summary")
                    summary_df = (
                        df[["tracking_id", "class_name"]]
                        .drop_duplicates()
                        .rename(columns={"tracking_id": "Track ID", "class_name": "Label"})
                        .sort_values(by="Track ID")
                    )
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                finally:
                    # Cleanup temporary files.
                    if os.path.exists(tmp_in_path):
                        os.remove(tmp_in_path)
                    if os.path.exists(output_path):
                        os.remove(output_path)
