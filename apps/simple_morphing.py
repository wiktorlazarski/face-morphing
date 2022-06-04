import numpy as np
import argparse

import cv2

import face_morphing.face_keypoints_detection_pipeline as kp_pipeline
import face_morphing.morphing.keypoints_alignment as kp_morph

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Homography morphing")

    # fmt: off
    parser.add_argument("--from_image", type=str, required=True)
    parser.add_argument("--to_image", type=str, required=True)
    # fmt: on

    args = parser.parse_args()

    morphing_alg = kp_morph.KeypointsAlignmentMorphing(num_morphs=300)
    kps_detection = kp_pipeline.FaceKeypointsDetectionPipeline()

    from_img = cv2.imread(args.from_image)
    from_img = cv2.cvtColor(from_img, cv2.COLOR_BGR2RGB)
    from_kps = kps_detection(from_img)
    from_img = cv2.cvtColor(from_img, cv2.COLOR_RGB2BGR)

    to_img = cv2.imread(args.to_image)
    to_img = cv2.cvtColor(to_img, cv2.COLOR_BGR2RGB)
    to_kps = kps_detection(to_img)
    to_img = cv2.cvtColor(to_img, cv2.COLOR_RGB2BGR)

    from_with_keypoints = from_img.copy()
    for keypoint in from_kps:
        x, y = round(keypoint[0]), round(keypoint[1])
        from_with_keypoints = cv2.circle(from_with_keypoints, (x, y), radius=5, color=(0,255,0), thickness=-1)

    to_with_keypoints = to_img.copy()
    for keypoint in to_kps:
        x, y = round(keypoint[0]), round(keypoint[1])
        to_with_keypoints = cv2.circle(to_with_keypoints, (x, y), radius=5, color=(0,255,0), thickness=-1)

    morphin_seq = morphing_alg.looped_morphing(
        from_img, to_img, from_kps, to_kps, combined_warped=False
    )

    cv2.imshow("From image predicted keypoints", from_with_keypoints)
    cv2.imshow("To image predicted keypoints", to_with_keypoints)
    while True:
        for morph in morphin_seq:
            cv2.imshow("Homography morphing", morph)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                exit(0)
