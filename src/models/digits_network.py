import torch

from lightning import LightningModule

from torch import nn
from torch.optim import AdamW
from torchvision.transforms import ToPILImage


# Определение генератора
class Generator(nn.Module):
    def __init__(self, noise_dim):
        """Инициализирует генератор

        Args:
            noise_dim (int): размерность вектора шума
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Вход: вектор шума размера noise_dim
            nn.Linear(noise_dim, 256 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 7, 7)),
            # Состояние: (256, 7, 7)
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4, stride=2, padding=1, bias=False
            ),  # -> (128, 14, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, 1,
                kernel_size=4, stride=2, padding=1, bias=False
            ),  # -> (1, 28, 28)
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Определение дискриминатора
class Discriminator(nn.Module):
    def __init__(self):
        """Инициализирует дискриминатор
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Вход: изображение (1, 28, 28)
            nn.Conv2d(
                1, 64,
                kernel_size=4, stride=2, padding=1, bias=False
            ),  # -> (64, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                64, 128,
                kernel_size=4, stride=2, padding=1, bias=False
            ),  # -> (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)


class DigitsGeneratorNetwork(LightningModule):
    def __init__(self, config: dict, task):
        """Инициализирует сверточную сеть для генерации цифр 0-9

        Args:
            config (dict): словарь с параметрами
        """
        super(DigitsGeneratorNetwork, self).__init__()

        self.config = config
        self.noise_dim = config["noise_dim"]
        self.automatic_optimization = False

        self.gen_model = Generator(noise_dim=self.noise_dim)
        self.disc_model = Discriminator()

        self.bce_loss = nn.BCELoss()

        # ClearML logger
        self.clearml_logger = None
        if task is not None:
            self.clearml_logger = task.get_logger()

    def forward(self, x):
        """Выполняет forward-pass для заданного входного тензора x.

        Args:
            x (torch.tensor): входной тензор, (batch, noise_dim)

        Returns:
            torch.tensor: output tensor, (batch, 1, 28, 28)
        """
        return self.gen_model(x)

    def training_step(self, batch, batch_idx):
        """Выполняет шаг обучения/валидации

        Args:
            batch (tuple): images (torch.tensor) и labels (torch.tensor)
            batch_idx (int): индекс текущего шага
        """
        real_images, _ = batch
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, device=self.config["device"])
        fake_labels = torch.zeros(batch_size, device=self.config["device"])

        optimizer_g, optimizer_d = self.optimizers()

        # Обновляем дискриминатор
        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad()
        output = self.disc_model(real_images)

        ## Обучение на реальных изображениях
        errD_real = self.bce_loss(output, real_labels)
        self.manual_backward(errD_real)
        D_x = output.mean().item()

        ## Обучение на фейковых изображениях
        noise = torch.randn(
            batch_size,
            self.noise_dim,
            device=self.config["device"]
        )
        fake_images = self.gen_model(noise)
        output = self.disc_model(fake_images.detach())

        errD_fake = self.bce_loss(output, fake_labels)
        self.manual_backward(errD_fake)

        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        # Обновляем генератор
        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()

        output = self.disc_model(self.gen_model(noise))
        errG = self.bce_loss(output, real_labels)
        self.manual_backward(errG)
        D_G_z2 = output.mean().item()

        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        # Логируем метрики в ClearML
        metrics = {
            "Dis/loss": errD.item(),
            "Gen/loss": errG.item(),
            "Dis/real": D_x,
            "Dis-before/fake": D_G_z1,
            "Dis-after/fake": D_G_z2
        }

        self.clearml_logger.report_scalar(
            "Generated Loss", "Gen/loss", errG.item(), self.global_step
        )
        self.clearml_logger.report_scalar(
            "Discriminator Loss", "Dis/loss", errD.item(), self.global_step
        )
        self.clearml_logger.report_scalar(
            "Discriminator Probability", "Dis/real", D_x, self.global_step
        )
        self.clearml_logger.report_scalar(
            "Discriminator Probability",
            "Dis-before/fake",
            D_G_z1,
            self.global_step
        )
        self.clearml_logger.report_scalar(
            "Discriminator Probability",
            "Dis-after/fake",
            D_G_z2,
            self.global_step
        )

        self.log_dict(
            metrics,
            prog_bar=True,
            on_epoch=True,
            on_step=True
        )

    def validation_step(self, batch, batch_idx):
        """Шаг валидации: генерирует debug-изображения

        Args:
            batch (tuple): images (torch.tensor) и labels (torch.tensor)
            batch_idx (int): индекс текущего шага
        """
        if self.clearml_logger is None or batch_idx != 0:
            return
        if (self.current_epoch + 1) % self.config["debug_samples_epoch"] != 0:
            return

        transforms = ToPILImage()
        with torch.no_grad():
            fixed_noise = torch.randn(
                self.config["debug_samples"],
                self.noise_dim,
                device=self.config["device"]
            )
            fake_images = self(fixed_noise).detach().cpu()

        for i in range(self.config["debug_samples"]):
            img = transforms(fake_images[i])
            self.clearml_logger.report_image(
                title="Validation Images",
                series=f"Generated Digits Sample[{i+1}]",
                iteration=self.global_step,
                image=img
            )

    def configure_optimizers(self):
        """Конфигурирует оптимизатор и планировщик обучения
        """
        optimizer_g = AdamW(
            self.gen_model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
            betas=(self.config["b1"], self.config["b2"])
        )
        optimizer_d = AdamW(
            self.disc_model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
            betas=(self.config["b1"], self.config["b2"])
        )

        return [optimizer_g, optimizer_d], []


if __name__ == "__main__":
    config = {
        "noise_dim": 100,
        "batch_size": 1,
        "num_workers": 0,
        "lr": 2e-4,
        "weight_decay": 0,
        "b1": 0.5,
        "b2": 0.999,
        "debug_samples": 64
    }

    model = DigitsGeneratorNetwork(config=config, task=None)
    print(model)
