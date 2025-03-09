import torch

from lightning import LightningModule

from torch import nn
from torch.optim import AdamW, lr_scheduler
from sklearn.metrics import f1_score


class SignConvNetwork(LightningModule):
    def __init__(self, config: dict):
        """Инициализирует сверточную сеть для классификации знаков

        Args:
            config (dict): словарь с параметрами
                stride (int): шаг свертки
                dilation (int): dilation rate for convolution
                n_classes (int): количество классов
        """
        super(SignConvNetwork, self).__init__()

        self.config = config
        self.stride = config["stride"]
        self.dilation = config["dilation"]
        self.n_classes = config["n_classes"]

        self.block1 = nn.Sequential(
            # (batch, 1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1,
                stride=self.stride,
                dilation=self.dilation
            ),
            nn.BatchNorm2d(8),
            # (batch, 8, 28, 28)
            nn.AvgPool2d(2),
            # (batch, 8, 14, 14)
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            # (batch, 8, 14, 14)
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1,
                stride=self.stride,
                dilation=self.dilation
            ),
            nn.BatchNorm2d(16),
            # (batch, 16, 14, 14)
            nn.AvgPool2d(2),
            # (batch, 16, 7, 7)
            nn.ReLU()
        )

        self.lin1 = nn.Linear(in_features=16 * 7 * 7, out_features=100)
        # (batch, 100)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.lin2 = nn.Linear(100, self.n_classes)
        # (batch, 25)

        self.ce_loss = nn.CrossEntropyLoss()
        self.all_step_preds = []
        self.all_step_labels = []

    def forward(self, x):
        """Выполняет forward-pass для заданного входного тензора x.

        Args:
            x (torch.tensor): входной тензор, (batch, 1, 28, 28)

        Returns:
            torch.tensor: output tensor, (batch, 25)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = x.view((x.shape[0], -1))
        x = self.lin1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.lin2(x)

        return x

    def basic_step(self, batch, batch_idx, step: str):
        """Выполняет шаг обучения/валидации/тестирования

        Args:
            batch (tuple): images (torch.tensor) и labels (torch.tensor)
            batch_idx (int): индекс текущего шага
            step (str): тип шага ('train', 'valid', 'test')

        Returns:
            dict: словарь loss_dict, содержащий loss для шага
        """
        images, labels = batch
        logits = self(images)

        loss = self.ce_loss(logits, labels)
        loss_dict = {
            f"{step}/loss": loss
        }

        self.log_dict(loss_dict, prog_bar=True)
        self.all_step_preds.append(logits.argmax(dim=1))
        self.all_step_labels.append(labels)
        return loss_dict

    def training_step(self, batch, batch_idx):
        """Выполняет шаг обучения"""
        loss_dict = self.basic_step(batch, batch_idx, "train")
        return loss_dict["train/loss"]

    def validation_step(self, batch, batch_idx):
        """Выполняет шаг валидации"""
        loss_dict = self.basic_step(batch, batch_idx, "valid")
        return loss_dict["valid/loss"]

    def test_step(self, batch, batch_idx):
        """Выполняет шаг тестирования"""
        loss_dict = self.basic_step(batch, batch_idx, "test")
        return loss_dict["test/loss"]

    def epoche_metric(self, stage: str):
        """Вычисляет метрику F1-Score для эпохи на
        тренировочной/валидационной/тестовой выборке

        Args:
            stage (str): тип множества ('train', 'valid', 'test')
        """
        preds = torch.cat(self.all_step_preds).cpu().numpy()
        labels = torch.cat(self.all_step_labels).cpu().numpy()

        self.log(
            name=f"{stage}/F1-Score",
            value=f1_score(labels, preds, average="macro"),
            prog_bar=True
        )
        self.all_step_preds.clear()
        self.all_step_labels.clear()

    def on_training_epoch_end(self):
        """Вызывается в конце каждой эпохи обучения.
        Вычисляет метрику F1-Score для обучающей выборки и логирует её.
        """
        self.epoche_metric("train")

    def on_validation_epoch_end(self):
        """Вызывается в конце каждой эпохи валидации.
        Вычисляет метрику F1-Score для валидационной выборки и логирует её.
        """
        self.epoche_metric("valid")

    def configure_optimizers(self):
        """Конфигурирует оптимизатор и планировщик обучения.

        Возвращает словарь, содержащий оптимизатор AdamW и планировщик CosineAnnealingLR,
        а также параметры для планировщика (мониторинг метрики, интервал и частоту)
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"]
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.config["T_max"]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                "interval": "epoch",
                "frequency": 1
            }
        }


if __name__ == "__main__":
    config = {
        "stride": 1,
        "dilation": 1,
        "n_classes": 25,
        "batch_size": 1,
        "num_workers": 0,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "T_max": 1
    }

    model = SignConvNetwork(config=config)
    print(model)
