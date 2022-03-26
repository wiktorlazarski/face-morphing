import argparse
import logging
import random
import shutil
import typing as t
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess dataset")

    parser.add_argument(
        "--preprocessed_dataset_dir",
        "-p",
        type=Path,
        required=True,
        help="Preprocessed Helen dataset directory",
    )
    parser.add_argument(
        "--output_dataset_dir",
        "-o",
        type=Path,
        help="Output dataset directory",
    )

    return parser.parse_args()


def create_dataset_structure(root_path: Path) -> None:
    logging.info(f"Creating folders structure in {root_path}...")

    train_path = root_path / "train"
    val_path = root_path / "val"
    test_path = root_path / "test"

    for path in [train_path, val_path, test_path]:
        images_path = path / "images"
        images_path.mkdir(parents=True, exist_ok=False)


def split_dataset(
    path: Path, train_frac: float = 0.8, val_frac: float = 0.1
) -> t.Tuple[list, list, list]:
    logging.info("Splitting dataset...")

    images = list((path / "images").glob("*.jpg"))
    random.shuffle(images)
    images_num = len(images)

    train_index = int(images_num * train_frac)
    val_index = train_index + int(images_num * val_frac)
    train_dataset = images[:train_index]
    val_dataset = images[train_index:val_index]
    test_dataset = images[val_index:]

    return train_dataset, val_dataset, test_dataset


def copy_images(
    output_dset_root: Path, train_dataset: list, val_dataset: list, test_dataset: list
) -> None:
    logging.info("Copying images...")

    for name_dataset, dataset in {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }.items():
        for image in dataset:
            shutil.copy(image, output_dset_root / name_dataset / "images" / image.name)


def create_dataset_csv(
    src_path: Path,
    dst_path: Path,
    train_dataset: list,
    val_dataset: list,
    test_dataset: list,
) -> None:
    logging.info(f"Creating metadata.csv in {dst_path}...")

    rows_csv = pd.read_csv(src_path / "metadata.csv")

    for name, dataset in {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }.items():
        filenames, attr_df = [], []
        for image in dataset:
            filenames.append(image.name.rsplit(".", 1)[0])
        attr_df.append(rows_csv.loc[rows_csv["Filenames"].isin(filenames)])
        attr_df = pd.DataFrame(attr_df[0], columns=rows_csv.columns)
        metadata = dst_path / name / "metadata.csv"
        attr_df.to_csv(str(metadata), index=False)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    logging.info("Building dataset...")
    args = parse_args()

    create_dataset_structure(args.output_dataset_dir)
    train_dataset, val_dataset, test_dataset = split_dataset(
        args.preprocessed_dataset_dir
    )
    copy_images(args.output_dataset_dir, train_dataset, val_dataset, test_dataset)
    create_dataset_csv(
        args.preprocessed_dataset_dir,
        args.output_dataset_dir,
        train_dataset,
        val_dataset,
        test_dataset,
    )

    logging.info("Finished building dataset.")


if __name__ == "__main__":
    main()
