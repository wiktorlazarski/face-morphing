import argparse

import cv2

import face_morphing.morphing.cross_dissolve as cd_morph

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cross-dissolve morphing")

    # fmt: off
    parser.add_argument("--from_image", type=str, required=True)
    parser.add_argument("--to_image", type=str, required=True)
    # fmt: on

    args = parser.parse_args()

    morphing_alg = cd_morph.CrossDissolveMorphing(num_morphs=300)

    from_img = cv2.imread(args.from_image)
    to_img = cv2.imread(args.to_image)

    morphin_seq = morphing_alg.looped_morphing(from_img, to_img)

    while True:
        for morph in morphin_seq:
            cv2.imshow("Cross-dissolve morphing", morph)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                exit(0)
