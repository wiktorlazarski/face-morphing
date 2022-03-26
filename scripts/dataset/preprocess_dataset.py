import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess dataset")

    # fmt: off
    parser.add_argument("--raw_dataset_dir", "-r", type=Path, required=True, help="Raw Helen dataset directory")
    parser.add_argument("--output_dataset_dir", "-o", type=Path, help="Preprocessed Helen dataset directory")
    # fmt: on

    return parser.parse_args()


def create_dir_structure(root_path: Path) -> Path:
    logging.info(f"Creating folders structure in {root_path}...")

    images_path = root_path / "images"
    images_path.mkdir(parents=True, exist_ok=False)

    return images_path


def copy_images(src_path: Path, dst_path: Path) -> None:
    logging.info(f"Copying images from {src_path} to {dst_path}...")
    for image in src_path.glob("*.jpg"):
        shutil.copyfile(image, dst_path / image.name)


def create_csv(src_path: Path, dst_path: Path) -> None:
    logging.info(f"Creating metadata.csv in {dst_path}...")

    filenames, key_points, attr_df, headers = [], [], [], []

    for txt_file in src_path.glob("*.txt"):

        with open(txt_file, "r") as file:
            lines = file.readlines()
        filenames.append(lines[0].strip())

        key_points = [(key_point.strip()).split(" , ") for key_point in lines[1:]]
        key_points = [item for key_point in key_points for item in key_point]
        attr_df.append(key_points)

    for iter in range(1, int(len(attr_df[0]) / 2) + 1):
        headers.extend(["kp_" + str(iter) + "_x", "kp_" + str(iter) + "_y"])

    attr_df = pd.DataFrame(attr_df, columns=headers)
    attr_df.insert(0, "Filenames", filenames)

    metadata = dst_path / "metadata.csv"
    attr_df.to_csv(str(metadata), index=False)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    logging.info("Creating preprocessed dataset...")
    args = parse_args()

    destination_path = create_dir_structure(args.output_dataset_dir)

    joined_source_path = args.raw_dataset_dir / "helen/helen"
    for dir in joined_source_path.glob("helen_*"):
        copy_images(joined_source_path / dir.name, destination_path)

    create_csv(args.raw_dataset_dir / "helen/helen/annotation", args.output_dataset_dir)

    logging.info("Finished creating preprocessed dataset.")


if __name__ == "__main__":
    main()
