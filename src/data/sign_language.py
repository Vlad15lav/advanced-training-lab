import os
import pandas as pd
import torch

from utils.downloader import download_file, unzip_file
from lightning import LightningDataModule

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomApply,
    RandomRotation
)


TRAIN_DATASET_LINK = (
    "https://github.com/a-milenkin/ml_instruments"
    "/raw/refs/heads/main/data/sign_mnist_train.csv.zip"
)
TEST_DATASET_LINK = (
    "https://github.com/a-milenkin/ml_instruments"
    "/raw/refs/heads/main/data/sign_mnist_test.csv.zip"
)
DATA_DIR = os.path.normpath(os.path.join(os.getcwd(), "../data/raw"))
ZIP_PATH = os.path.join(DATA_DIR, "dataset.zip")


class SignLanguage(Dataset):
    def __init__(self, dataframe: str, transform=None):
        """
        Инициализирует объект SignLanguage

        Args:
            dataframe (str): путь к csv-файлу с данными
            transform (optional): преобразование, которое
                будет применяться к изображениям
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        """Возвращает размер датасета"""
        return len(self.dataframe)

    def __getitem__(self, index):
        """Возвращает пример из датасета

        Args:
            index (int): индекс примера

        Returns:
            image (torch.tensor): изображение
            label (torch.tensor): метка
        """
        image = self.dataframe.iloc[index, 1:].values
        label = self.dataframe.values[index, 0]

        image = torch.tensor(image, dtype=torch.uint8) / 255.
        image = image.view(-1, 28, 28)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label


class SignLanguageModule(LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def prepare_data(self):
        """Подготовка данных

        Создает папку для данных, если ее не существует, и скачивает
        датасет, если он не существует.
        """
        os.makedirs(DATA_DIR, exist_ok=True)

        if not os.path.exists(os.path.join(DATA_DIR, "sign_mnist_train.csv")):
            download_file(TRAIN_DATASET_LINK, ZIP_PATH)
            unzip_file(ZIP_PATH, DATA_DIR)
            os.remove(ZIP_PATH)

        if not os.path.exists(os.path.join(DATA_DIR, "sign_mnist_test.csv")):
            download_file(TEST_DATASET_LINK, ZIP_PATH)
            unzip_file(ZIP_PATH, DATA_DIR)
            os.remove(ZIP_PATH)

    def setup(self, stage: str):
        """Подготовка данных для обучения, валидации и тестирования

        Args:
            stage (str): стадия, для которой готовятся данные
        """
        if stage == "fit":
            transforms_train = Compose([
                RandomHorizontalFlip(
                    p=self.config["random_horizontal_flip"]
                ),
                RandomApply(
                    [
                        RandomRotation(degrees=(-180, 180))
                    ],
                    p=self.config["random_rotation"]
                )
            ])

            train = pd.read_csv(os.path.join(DATA_DIR, "sign_mnist_train.csv"))
            train, val = train_test_split(
                train,
                test_size=self.config["valid_size"],
                random_state=self.config["random_state_split"]
            )

            self.train = SignLanguage(train, transform=transforms_train)
            self.val = SignLanguage(val)
        else:
            test = pd.read_csv(os.path.join(DATA_DIR, "sign_mnist_test.csv"))
            self.test = SignLanguage(test)

    def _make_dataloader(self, dataset):
        """
        Создает DataLoader из датасета

        Args:
            dataset (SignLanguage): датасет

        Returns:
            DataLoader: DataLoader из датасета
        """
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=True,
            persistent_workers=self.config["num_workers"] > 0
        )

    def train_dataloader(self):
        """Создает DataLoader для обучения"""
        return self._make_dataloader(self.train)

    def val_dataloader(self):
        """Создает DataLoader для валидации"""
        return self._make_dataloader(self.val)

    def test_dataloader(self):
        """Создает DataLoader для тестирования"""
        return self._make_dataloader(self.test)

    def teardown(self, stage: str):
        """Очищает память от датасетов и их загрузчиков"""
        if stage == "fit":
            del self.train, self.val
        else:
            del self.test


if __name__ == "__main__":
    dataset = SignLanguageModule({
        "random_horizontal_flip": 0,
        "random_rotation": 0,
        "valid_size": 0.2,
        "random_state_split": 42,
        "batch_size": 32,
        "num_workers": 0
    })

    dataset.prepare_data()
    dataset.setup("fit")

    for X, y in dataset.train_dataloader():
        print(X.shape, y.shape)
        break
