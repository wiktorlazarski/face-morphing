import os
import typing as t

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics

import face_morphing.image_processing as ip
import face_morphing.model as mdl
import scripts.training.augmentations as aug
import scripts.training.data_loading as dl

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        batch_size: int,
        num_workers: int,
        dataset_root: str,
        nn_input_image_resolution: int,
        use_all_augmentations: bool,
        resize_augmentation_keys: t.Optional[t.List[str]] = None,
        augmentation_keys: t.Optional[t.List[str]] = None,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_root = dataset_root
        self.nn_input_image_resolution = nn_input_image_resolution
        self.use_all_augmentations = use_all_augmentations
        self.resize_augmentation_keys = resize_augmentation_keys
        self.augmentation_keys = augmentation_keys

    def setup(self, stage: t.Optional[str] = None):
        preprocessing_pipeline = ip.PreprocessingPipeline(
            nn_image_input_resolution=self.nn_input_image_resolution,
        )
        augmentation_pipeline = aug.AugmentationPipeline(
            use_all_augmentations=self.use_all_augmentations,
            resize_augmentation_keys=self.resize_augmentation_keys,
            augmentation_keys=self.augmentation_keys,
        )
        self.train_dataset = dl.HelenFaceMorphingDataset(
            dataset_root=os.path.join(ROOT_DIR, self.dataset_root, "train"),
            preprocess_pipeline=preprocessing_pipeline,
            augmentation_pipeline=augmentation_pipeline,
        )

        self.validation_dataset = dl.HelenFaceMorphingDataset(
            dataset_root=os.path.join(ROOT_DIR, self.dataset_root, "val"),
            preprocess_pipeline=preprocessing_pipeline,
            augmentation_pipeline=augmentation_pipeline,
        )

        self.test_dataset = dl.HelenFaceMorphingDataset(
            dataset_root=os.path.join(ROOT_DIR, self.dataset_root, "test"),
            preprocess_pipeline=preprocessing_pipeline,
            augmentation_pipeline=augmentation_pipeline,
        )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        *,
        lr: float,
        base_model: str = 'resnet50',
        number_of_cordinates: int = 388
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = mdl.KeyPointsModel(
            number_of_cordinates=number_of_cordinates,
            base_model=base_model
        )

        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

        self.lr = lr

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ):
        return self._step(batch, mse_metric=self.train_mse)

    def validation_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ):
        return self._step(batch, mse_metric=self.val_mse)

    def test_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ):
        return self._step(batch, mse_metric=self.test_mse)

    def training_epoch_end(self, outputs) -> None:
        self._summarize_epoch(
            log_prefix="train", outputs=outputs, mse_metric=self.train_mse
        )

    def validation_epoch_end(self, outputs) -> None:
        self._summarize_epoch(
            log_prefix="val", outputs=outputs, mse_metric=self.val_mse
        )

    def test_epoch_end(self, outputs) -> None:
        self._summarize_epoch(
            log_prefix="test", outputs=outputs, mse_metric=self.test_mse
        )

    def _step(
        self,
        batch: t.Tuple[torch.Tensor, torch.Tensor],
        mse_metric: torchmetrics.MeanSquaredError,
    ):
        images, points = batch
        predicted_points = self.model(images)
        loss = F.mse_loss(predicted_points, points)

        mse_metric(predicted_points, points)

        return { 'loss': loss }

    def _summarize_epoch(
        self,
        log_prefix: str,
        outputs: pl.utilities.types.EPOCH_OUTPUT,
        mse_metric: torchmetrics.MeanSquaredError,
    ) -> None:
        mean_loss = torch.mean(torch.tensor([out['loss'] for out in outputs]))
        self.log(f"{log_prefix}_loss", mean_loss, on_epoch=True)

        mse = mse_metric.compute()
        self.log(f"{log_prefix}_mse", mse, on_epoch=True)

        mse_metric.reset()

