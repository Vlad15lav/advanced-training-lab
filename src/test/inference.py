import random
import torch
import numpy as np
import matplotlib.pyplot as plt


CLASS_TO_LETTER = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I",
    9: "K", 10: "L", 11: "M", 12: "N", 13: "O",
    14: "P", 15: "Q", 16: "R", 17: "S", 18: "T",
    19: "U", 20: "V", 21: "W", 22: "X", 23: "Y"
}


def plot_sample(image, label=None):
    """Отображает изображение жеста и, если задано,
    добавляет заголовок с меткой

    Args:
        image (numpy.ndarray): Изображение жеста для отображения
        label (str, optional): Метка жеста, отображаемая в заголовке
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray_r")
    plt.axis("off")

    if label:
        plt.title(
            f"Sign: {label}", fontsize=14, fontweight="bold", pad=10
        )

    plt.show()


def test_sample(model, test_loader, idx: int = None):
    """Тестирует модель на случайном примере из тестовой выборки

    Args:
        model: тестируемая модель
        test_loader: DataLoader тестовой выборки
        idx: индекс примера из тестовой выборки
    """
    imgs, labels = next(iter(test_loader))

    if idx is None or idx >= len(imgs) or idx < 0:
        idx = random.randint(0, len(imgs) - 1)

    img = imgs[idx].cpu()

    model = model.cpu()
    model.eval()
    with torch.no_grad():
        pred = model(img[:, None])[0]

    pred = pred.argmax().item()
    label = labels[idx].item()

    print(f"Предсказанная метка: {CLASS_TO_LETTER[pred]}")
    print(f"Истинная метка: {CLASS_TO_LETTER[label]}")

    img = np.uint8(img.numpy() * 255).reshape(28, 28)
    plot_sample(
        img,
        label=(
            f"Predicted: {CLASS_TO_LETTER[pred]}"
            f" - True: {CLASS_TO_LETTER[label]}"
        )
    )
