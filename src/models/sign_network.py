import torchmetrics as tm

from lightning import LightningModule

from torch import nn
from torch.optim import AdamW, lr_scheduler
from test.metrics import FalseDiscoveryRate


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
        self.stride = config.get("stride", 1)
        self.dilation = config.get("dilation", 1)
        self.n_classes = config["n_classes"]
        self.f_beta = config.get("f_beta", 1.0)

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

        self.softmax = nn.Softmax(dim=1)
        self.ce_loss = nn.CrossEntropyLoss()

        # F-beta Score
        self.train_f_beta = tm.FBetaScore(
            task="multiclass",
            num_classes=self.n_classes,
            beta=self.f_beta
        )
        self.valid_f_beta = tm.FBetaScore(
            task="multiclass",
            num_classes=self.n_classes,
            beta=self.f_beta
        )
        # ROC-AUC
        self.train_roc_auc = tm.AUROC(
            task="multiclass",
            num_classes=self.n_classes
        )
        self.valid_roc_auc = tm.AUROC(
            task="multiclass",
            num_classes=self.n_classes
        )
        # Custom False Discovery Rate
        self.train_fdr = FalseDiscoveryRate(num_classes=self.n_classes)
        self.valid_fdr = FalseDiscoveryRate(num_classes=self.n_classes)

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
        """Выполняет шаг обучения/валидации

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

        if step == "train":
            f_beta_score = self.train_f_beta(logits, labels)
            roc_auc_score = self.train_roc_auc(logits, labels)
            fdr_score = self.train_fdr(logits, labels)
        elif step == "valid":
            f_beta_score = self.valid_f_beta(logits, labels)
            roc_auc_score = self.valid_roc_auc(logits, labels)
            fdr_score = self.valid_fdr(logits, labels)

        metrics = {
            f"{step}/loss": loss,
            f"{step}/f_beta": f_beta_score,
            f"{step}/roc_auc": roc_auc_score,
            f"{step}/fdr": fdr_score
        }
        self.log_dict(
            metrics,
            prog_bar=True,
            on_epoch=True,
            on_step=(step == "train")
        )

        return metrics

    def training_step(self, batch, batch_idx):
        """Выполняет шаг обучения"""
        metrics = self.basic_step(batch, batch_idx, "train")
        return metrics["train/loss"]

    def validation_step(self, batch, batch_idx):
        """Выполняет шаг валидации"""
        metrics = self.basic_step(batch, batch_idx, "valid")
        return metrics["valid/loss"]

    def test_step(self, batch, batch_idx):
        """Выполняет шаг тестирования"""
        metrics = self.basic_step(batch, batch_idx, "test")
        return metrics["test/loss"]

    def configure_optimizers(self):
        """Конфигурирует оптимизатор и планировщик обучения.

        Возвращает словарь, содержащий оптимизатор AdamW
        и планировщик CosineAnnealingLR, а также параметры
        для планировщика (мониторинг метрики, интервал и частоту)
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
        "T_max": 1,
        "f_beta": 1
    }

    model = SignConvNetwork(config=config)
    print(model)
