import os

from lightning import LightningDataModule

from sklearn.model_selection import train_test_split

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize
)

DATA_DIR = os.path.normpath(os.path.join(os.getcwd(), "../data/raw"))


class DigitsMnistModule(LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def prepare_data(self):
        """Подготовка данных

        Создает папку для данных, если ее не существует, и скачивает
        датасет, если он не существует.
        """
        os.makedirs(DATA_DIR, exist_ok=True)

        if not os.path.exists(os.path.join(DATA_DIR, "training.pt")):
            MNIST(DATA_DIR, train=True, download=True)

        if not os.path.exists(os.path.join(DATA_DIR, "test.pt")):
            MNIST(DATA_DIR, train=False, download=True)

    def setup(self, stage: str):
        """Подготовка данных для обучения, валидации и тестирования

        Args:
            stage (str): стадия, для которой готовятся данные
        """
        transforms = Compose([
            ToTensor(),
            Normalize(
                (self.config["normalize_mean"],),
                (self.config["normalize_std"],),
            )
        ])

        if stage == "fit":
            dataset = MNIST(
                root=DATA_DIR,
                train=True,
                download=False,
                transform=transforms
            )

            if self.config["valid_size"] > 0:
                self.train, self.val = train_test_split(
                    dataset,
                    test_size=self.config["valid_size"],
                    random_state=self.config["random_state_split"]
                )
            else:
                self.train = dataset
                _, self.val = train_test_split(
                    dataset,
                    test_size=1
                )
        else:
            self.test = MNIST(
                root=DATA_DIR,
                train=False,
                download=False,
                transform=transforms
            )

    def _make_dataloader(self, dataset, shuffle=False):
        """Создает DataLoader из датасета

        Args:
            dataset (Digits MNIST): датасет

        Returns:
            DataLoader: DataLoader из датасета
        """
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=self.config["num_workers"] > 0
        )

    def train_dataloader(self):
        """Создает DataLoader для обучения"""
        return self._make_dataloader(self.train, shuffle=True)

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
    dataset = DigitsMnistModule({
        "normalize_mean": 0.5,
        "normalize_std": 0.5,
        "valid_size": 0.2,
        "random_state_split": 42,
        "batch_size": 64,
        "num_workers": 0
    })

    dataset.prepare_data()
    dataset.setup("fit")

    for X, y in dataset.train_dataloader():
        print(X.shape, y.shape)
        print(X)
        print(y)
        break
