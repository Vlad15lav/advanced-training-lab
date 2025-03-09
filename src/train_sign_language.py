import click

from src.data.sign_language import SignLanguageModule
from src.models.sign_network import SignConvNetwork

from lightning import seed_everything
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


CONFIG = {
    # Датасет/аугментация
    "random_horizontal_flip": 0.1,
    "random_rotation": 0.2,
    "valid_size": 0.2,
    "random_state_split": 42,
    "batch_size": 128,

    # Параметры модели
    "stride": 1,
    "dilation": 1,
    "n_classes": 25,

    # Training параметры
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "log_every_n_steps": 1,
    "T_max": 20,
    "best_model": True,
    "early_stopping": True,
    "early_stopping_patience": 5,
    "best_model_name": "sign-language-model",

    "device": "cpu",  # ""cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,
    "random_seed": 2025
}


def train_model(model, dataset, fast_dev_run: bool, config: dict):
    """Обучает модель с помощью PyTorch Lightning

    Args:
        model (nn.Module): модель, которую нужно обучить
        dataset (LightningDataModule): датасет, используемый для
            обучения
        fast_dev_run (bool): если True, то
            производится быстрый тестовый прогон
        config (dict): словарь с параметрами обучения
    """
    # Быстрое тестирование с одним прогоном
    if fast_dev_run:
        try:
            trainer = Trainer(fast_dev_run=True, enable_progress_bar=False)
            trainer.fit(model, datamodule=dataset)

            print("Тестовый прогон успешно пройден!")
        except Exception:
            print("Тестовый прогон завершился с ошибкой!")
            return None

    # Настраиваем EarlyStopping и автосохранение лучшей модели
    callback_list = []
    if config["early_stopping"]:
        callback_list.append(
            EarlyStopping(
                monitor="valid/F1-Score",
                mode="max",
                patience=config["early_stopping_patience"]
            )
        )

    if config["best_model"]:
        callback_list.append(
            ModelCheckpoint(
                save_top_k=1,
                monitor="valid/F1-Score",
                mode="max",
                dirpath="./models/",
                filename=config["best_model_name"],
                save_weights_only=True
            )
        )

    # Тренируем модель
    trainer = Trainer(
        max_epochs=config["epochs"],
        log_every_n_steps=config["log_every_n_steps"],
        callbacks=callback_list
    )

    print("Тренировка модели...")
    trainer.fit(model, datamodule=dataset)
    print("Модель обучена! Веса сохранены в ./models")


@click.command()
@click.option("--fast-dev-run", is_flag=True, help="Running in fast dev mode")
def main(fast_dev_run: bool):
    seed_everything(CONFIG["random_seed"])
    print(f"Зафиксирован random seed: {CONFIG['random_seed']}")

    dataset = SignLanguageModule(config=CONFIG)
    print("Датасет подготовлен!")
    model = SignConvNetwork(config=CONFIG)
    print("Модель создана!")

    train_model(model, dataset, fast_dev_run, CONFIG)


if __name__ == "__main__":
    main()
