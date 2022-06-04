import time

import cv2
import numpy as np

import face_morphing.face_keypoints_detection_pipeline as kp_pipeline


def main() -> None:
    OPENCV_WINDOW_NAME = "Face Keypoints Detection FPS Measurement"
    cv2.namedWindow(OPENCV_WINDOW_NAME, cv2.WINDOW_NORMAL)

    web_camera = cv2.VideoCapture(0)

    prev_frame_time = 0
    next_frame_time = 0

    kps_pipeline = kp_pipeline.FaceKeypointsDetectionPipeline()

    while web_camera.isOpened():
        ret, frame = web_camera.read()
        if not ret:
            break

        try:
            pred_keypoints = kps_pipeline.predict(frame)
        except Exception:
            continue

        next_frame_time = time.time()

        fps = round(1 / (next_frame_time - prev_frame_time), 4)
        prev_frame_time = next_frame_time

        img_with_keypoints = frame.copy()
        for keypoint in pred_keypoints:
            x, y = round(keypoint[0]), round(keypoint[1])
            img_with_keypoints = cv2.circle(img_with_keypoints, (x, y), radius=5, color=(0,255,0), thickness=-1)

        display_frame = np.hstack((frame, img_with_keypoints))
        display_frame = cv2.putText(
            img=display_frame,
            text=f"FPS = {str(fps)}",
            org=(15, 55),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 255, 0),
            thickness=5,
        )

        cv2.imshow(OPENCV_WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    web_camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()