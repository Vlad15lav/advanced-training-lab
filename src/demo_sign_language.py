import os

from data.sign_language import SignLanguageModule
from models.sign_network import SignConvNetwork

from test.inference import test_sample


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
    "best_model_name": "sign-language-model",
    "device": "cpu",
    "num_workers": 2
}


def main():
    dataset = SignLanguageModule(config=CONFIG)
    print("Датасет подготовлен!")

    dataset.setup("test")
    model = SignConvNetwork.load_from_checkpoint(
        os.path.join("../weights/", CONFIG["best_model_name"] + ".ckpt"),
        config=CONFIG
    )
    print("Веса загружены!")

    print("Инференс тестового образца...")
    test_sample(model, dataset.test_dataloader())


if __name__ == "__main__":
    main()
