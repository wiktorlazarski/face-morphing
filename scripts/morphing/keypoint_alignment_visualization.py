import argparse

import cv2

import face_morphing.morphing.keypoints_alignment as morph_alg


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser("Visualizes a result of the face morphing cross-dissolve algorithm.")

    parser.add_argument("-fi", "--first_image", type=str, required=True, help="A file path to a first image.")
    parser.add_argument("-si", "--second_image", type=str, required=True, help="A file path to a second image.")
    parser.add_argument("-n", "--num_morphs", type=int, default=300, help="A number of morph generated in a sequence.")
    parser.add_argument("-l", "--looped", action="store_true", help="Visualize morphing in a loop.")
    parser.add_argument("-cw", "--combined_warped", action="store_true", help="Visualize morphing in a loop.")

    # fmt: on
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from_img = cv2.imread(args.first_image)
    to_img = cv2.imread(args.second_image)

    morphing_alg = morph_alg.KeypointsAlignmentMorphing(args.num_morphs)

    # Compute by NN
    from_kps = None
    to_kps = None

    morph_seq = (
        morphing_alg.looped_morphing(
            from_img, to_img, from_kps, to_kps, combined_warped=args.combined_warped
        )
        if args.looped
        else morphing_alg.generate_morph_sequence(
            from_img, to_img, from_kps, to_kps, combined_warped=args.combined_warped
        )
    )

    window_name = "homography morphing"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        for morph in morph_seq:
            cv2.imshow(window_name, morph)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                exit(0)


if __name__ == "__main__":
    main()
