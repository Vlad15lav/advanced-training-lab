import click
import torch

from utils.check import get_env_clearml
from data.digits_mnist import DigitsMnistModule
from models.digits_network import DigitsGeneratorNetwork

from clearml import Task

from lightning import seed_everything
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar
)


CONFIG = {
    # Датасет/аугментация
    "normalize_mean": 0.5,
    "normalize_std": 0.5,
    "valid_size": 0,
    "random_state_split": 42,
    "batch_size": 64,

    # Параметры модели
    "noise_dim": 100,

    # Training параметры
    "epochs": 10,
    "lr": 2e-4,
    "weight_decay": 0.0,
    "b1": 0.5,
    "b2": 0.999,
    "log_every_n_steps": 1,
    "debug_samples_epoch": 1,
    "debug_samples": 32,
    "best_model": True,
    "early_stopping": False,
    "early_stopping_patience": 5,
    "best_model_name": "digits-generate-model",

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,
    "random_seed": 2025
}


def train_model(model, dataset, config: dict):
    """Обучает модель с помощью PyTorch Lightning

    Args:
        model (nn.Module): модель, которую нужно обучить
        dataset (LightningDataModule): датасет, используемый для
            обучения
        config (dict): словарь с параметрами обучения
    """
    # Настраиваем EarlyStopping и автосохранение лучшей модели
    callback_list = [RichProgressBar()]
    if config["early_stopping"]:
        callback_list.append(
            EarlyStopping(
                monitor="Gen/loss",
                mode="min",
                patience=config["early_stopping_patience"]
            )
        )

    if config["best_model"]:
        callback_list.append(
            ModelCheckpoint(
                save_top_k=1,
                save_last=True,
                monitor="Gen/loss",
                mode="min",
                dirpath="../models/",
                filename=config["best_model_name"],
                save_weights_only=True,
                enable_version_counter=False
            )
        )

    # Тренируем модель
    trainer = Trainer(
        max_epochs=config["epoch"],
        log_every_n_steps=config["log_every_n_steps"],
        default_root_dir='../',
        callbacks=callback_list
    )

    print("Тренировка модели...")
    trainer.fit(model, datamodule=dataset)
    print("Модель обучена! Веса сохранены в advanced-training-lab/models")


@click.command()
@click.option(
    "--epoch",
    type=int,
    default=10,
    help="Epochs on training"
)
@click.option(
    "--debug-samples-epoch",
    type=int,
    default=1,
    help="Frequency debug samples"
)
def main(epoch: int, debug_samples_epoch: int):
    CONFIG["epoch"] = epoch
    CONFIG["debug_samples_epoch"] = debug_samples_epoch

    get_env_clearml()
    seed_everything(CONFIG["random_seed"])

    # Настройка логирования ClearML
    task = Task.init(
        project_name="ClearML Course",
        task_name="MNIST Digits Generator"
    )
    task.add_tags(["baseline", "model"])
    task.connect(
        CONFIG,
        name="GAN Config"
    )

    dataset = DigitsMnistModule(config=CONFIG)
    print("Датасет подготовлен!")
    model = DigitsGeneratorNetwork(config=CONFIG, task=task)
    print("Модель создана!")

    train_model(model, dataset, CONFIG)
    task.close()


if __name__ == "__main__":
    main()
