import argparse

import cv2

import face_morphing.morphing.cross_dissolve as morph_alg


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser("Visualizes a result of the face morphing cross-dissolve algorithm.")

    parser.add_argument("-fi", "--first_image", type=str, required=True, help="A file path to a first image.")
    parser.add_argument("-si", "--second_image", type=str, required=True, help="A file path to a second image.")
    parser.add_argument("-n", "--num_morphs", type=int, default=300, help="A number of morph generated in a sequence.")
    parser.add_argument("-l", "--looped", action="store_true", help="Visualize morphing in a loop.")

    # fmt: on
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from_img = cv2.imread(args.first_image)
    to_img = cv2.imread(args.second_image)

    f_h, f_w = from_img.shape[:2]
    resized = cv2.resize(to_img, (f_w, f_h))

    cd_morphing = morph_alg.CrossDissolveMorphing(args.num_morphs)

    morph_seq = (
        cd_morphing.looped_morphing(from_img, resized)
        if args.looped
        else cd_morphing.generate_morph_sequence(from_img, resized)
    )

    while True:
        for morph in morph_seq:
            cv2.imshow("cross-dissolve morphing", morph)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                exit(0)


if __name__ == "__main__":
    main()
