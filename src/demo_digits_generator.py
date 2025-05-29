import os
import math
import click
import torch

from models.digits_network import DigitsGeneratorNetwork
from torchvision.utils import make_grid, save_image


CONFIG = {
    # Параметры модели
    "noise_dim": 100,
    "best_model_name": "digits-generate-model",

    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


@click.command()
@click.option(
    "--samples",
    type=int,
    default=16,
    help="Count of samples"
)
@click.option(
    "--output_path",
    type=str,
    default="generated_grid.png",
    help="Path to save the generated image grid"
)
def main(samples: int, output_path: str):
    model = DigitsGeneratorNetwork.load_from_checkpoint(
        os.path.join("../weights/", CONFIG["best_model_name"] + ".ckpt"),
        config=CONFIG,
        task=None
    ).to(CONFIG["device"])
    print("Веса загружены!")

    print("Генерация изображений...")
    with torch.no_grad():
        fixed_noise = torch.randn(
            samples,
            CONFIG["noise_dim"],
            device=CONFIG["device"]
        )

        fake_images = model(fixed_noise).detach().cpu()

    # Создание сетки изображений
    grid = make_grid(
        fake_images,
        nrow=math.ceil(math.sqrt(samples)),
        normalize=True,
        padding=2,
        pad_value=255
    )

    # Сохранение в файл
    save_image(grid, output_path)
    print(f"Изображения сохранены в {output_path}")


if __name__ == "__main__":
    main()
