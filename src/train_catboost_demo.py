#!/usr/bin/env python
import os
import warnings
import click

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union, List, Tuple
from getpass import getpass
from clearml import Task, Logger

from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score


REQUIRED_ENV = (
    "CLEARML_WEB_HOST",
    "CLEARML_API_HOST",
    "CLEARML_FILES_HOST",
    "CLEARML_API_ACCESS_KEY",
    "CLEARML_API_SECRET_KEY"
)

RANDOM_SEED = 2025
CAT_FEATURES = ("model", "car_type", "fuel_type")
DROP_FEATURES = ("car_id", "target_reg")
TARGET_COLUMN = ("target_class")
VALID_SIZE = 0.25

CATBOOST_PARAMS = {
    "random_seed": RANDOM_SEED,
    "depth": 4,
    "learning_rate": 0.05,
    "early_stopping_rounds": 25,
    "eval_metric": "Accuracy",
    "loss_function": "MultiClass",
    "use_best_model": True,

    "bootstrap_type": "Bernoulli",
    "subsample": 0.80,
    "colsample_bylevel": 0.098,
    "l2_leaf_reg": 9,
    "min_data_in_leaf": 240,
    "max_bin": 180,
    "random_strength": 1,

    "task_type": "CPU",
    "thread_count": -1
}


def get_env_clearml():
    """Функция проверки переменных окружения для подключения к ClearML.
    Запрашивает пользователя ввести отсутствующие значения
    """
    missing_env = []
    for env in REQUIRED_ENV:
        if os.getenv(env) is None:
            missing_env.append(env)

    if missing_env:
        warnings.warn((
            "The following environment variables are missing:\n"
            f"{', '.join(missing_env)}\nPlease create them on "
            "https://app.clear.ml/settings/workspace-configuration"
        ))

    for env in missing_env:
        os.environ[env] = getpass(prompt=f"Entry {env}: ")


def set_seed(seed: int = 2025):
    """Функция фиксирования рандомизации для воспроизводимости

    Args:
        seed (int): Начальное значение генератора чисел
    """
    np.random.seed(seed)
    pd.core.common.random_state(seed)


def plot_bar_distribution(
    data: pd.DataFrame,
    columns: Union[str, List[str], Tuple[str]],
    series: str,
    logger: Logger
):
    """Функция отрисовка bar графика для категориальных столбцов

    Args:
        data (Pool): Анализируемая выборка
        column (str, tuple, list): Название анализируемых столбцов
        series (str): Название графика
        logger (Logger): Объект ClearML для логирования
    """
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        # Вычисление значений для графика
        cat_counts = data[col].value_counts()
        cat_percentages = cat_counts / len(data[col]) * 100

        # Отрисовка графика
        plt.figure(figsize=(8, 5))
        ax = cat_counts.plot(kind='bar', color='skyblue', alpha=0.7)

        for i, count in enumerate(cat_counts):
            ax.text(
                i,
                count + 0.05,
                f'{cat_percentages.iloc[i]:.2f}%',
                ha='center',
                fontsize=10
            )

        plt.xlabel('Значения')
        plt.ylabel('Частота')
        plt.title(f'Распределение {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()

        logger.report_matplotlib_figure(
            title='Dataset Analysis',
            series=f"{series} ({col})",
            figure=plt.gcf()
        )


def plot_data_analysis(data: Pool, logger: Logger):
    """Функция для EDA датасета

    Args:
        data (Pool): Анализируемая выборка
        logger (Logger): Объект ClearML для логирования
    """
    # График распределение признаков
    data.hist(bins=20, figsize=(25, 6), layout=(-1, 6))
    logger.report_matplotlib_figure(
        title='Dataset Analysis',
        series='Features Distribution',
        figure=plt.gcf()
    )

    # Графики распределения категориальных признаков
    plot_bar_distribution(
        data=data,
        columns=CAT_FEATURES,
        series="Categorial Features",
        logger=logger
    )

    # График распределение таргета
    plot_bar_distribution(
        data=data,
        columns=TARGET_COLUMN,
        series="Target Class Imbalance",
        logger=logger
    )


def get_dataset(config: dict, logger: Logger) -> Union[Pool, Pool]:
    """Функция чтения датасета и подготовка для модели

    Args:
        config (dict): Конфигурация скрипта
        logger (Logger): Объект ClearML для логирования
    Returns:
        train_pool (Pool): Обучающая выборка
        valid_pool (Pool): Валидационная выборка
    """
    dataset = pd.read_csv(config["path_data"])
    dataset.drop(columns=list(DROP_FEATURES), axis=1, inplace=True)

    # Разбиение данных
    train, valid = train_test_split(
        dataset,
        test_size=VALID_SIZE,
        random_state=42
    )

    # Логирование данных
    data_info = pd.DataFrame({
        "Column": dataset.columns,
        "Non-Null Count": dataset.count().values,
        "Dtype": dataset.dtypes.values.astype(str)
    })

    logger.report_table(
        title="Dataset",
        series="Full Dataset Info",
        table_plot=data_info
    )
    logger.report_table(
        title="Dataset",
        series="Valid set",
        table_plot=valid
    )

    # EDA
    if config["plot_data"]:
        plot_data_analysis(train, logger)

    # Разбиение X и y для CatBoost
    X_train, y_train = train.drop(columns=TARGET_COLUMN), train[TARGET_COLUMN]
    X_valid, y_valid = valid.drop(columns=TARGET_COLUMN), valid[TARGET_COLUMN]

    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    valid_pool = Pool(data=X_valid, label=y_valid, cat_features=CAT_FEATURES)

    return train_pool, valid_pool


def train_model(train_pool: Pool, valid_pool: Pool, config: dict):
    """Обучение модели CatBoostClassifier

    Args:
        train_pool (Pool): Обучающая выборка
        valid_pool (Pool): Валидационная выборка
        config (dict): Конфигурация скрипта
    Returns:
        model: Обученная модель CatBoostClassifier
    """
    CATBOOST_PARAMS["iterations"] = config["iterations"]

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        verbose=config["verbose"]
    )

    return model


def eval_model(model, data_pool: Pool, logger: Logger, prefix_name: str = ""):
    """Оценка качества модели классификации

    Args:
        model: Обученная модель классификации
        data_pool (Pool): Оцениваемая выборка
        logger (Logger): Объект ClearML для логирования
        prefix_name (str): Префикс для названия графиков
    """
    y_pred = model.predict(data_pool)
    y_label = data_pool.get_label()

    accuracy = accuracy_score(y_label, y_pred)
    cls_report = classification_report(
        y_label,
        y_pred,
        target_names=np.unique(data_pool.get_label()),
        output_dict=True,
        zero_division=0
    )
    cls_report = pd.DataFrame(cls_report).T

    logger.report_single_value(
        name=(f"{prefix_name} " if prefix_name else "") + "Accuracy",
        value=accuracy
    )
    logger.report_table(
        title="Metrics",
        series=f"{prefix_name} Classification Report",
        table_plot=cls_report
    )


@click.command()
@click.option(
    "--path-data",
    default=("https://github.com/a-milenkin/ml_instruments"
             "/raw/refs/heads/main/data/quickstart_train.csv"),
    help="Path of dataset",
    type=str
)
@click.option("--iterations", default=500, help="Number of iterations")
@click.option("--verbose", default=False, help="Verbose output", type=int)
@click.option("--plot-data", is_flag=True, help="Plot EDA for data")
def main(**kwargs):
    # Инициализация конфига, seed, переменных ClearML
    config = dict(kwargs)
    get_env_clearml()
    set_seed(seed=RANDOM_SEED)

    # Настройка логирования ClearML
    task = Task.init(
        project_name="ClearML Course",
        task_name="CatBoost Model baseline"
    )
    task.add_tags(["baseline", "model"])
    logger = Logger.current_logger()
    task.connect(
        CATBOOST_PARAMS,
        name="Baseline Config"
    )

    # Загрузка датасета, обучение CatBoost, оценка качества
    train_pool, valid_pool = get_dataset(config, logger)
    model = train_model(train_pool, valid_pool, config)
    eval_model(model, valid_pool, logger, prefix_name="Valid")

    # Сохраняем модель и завершаем логирование
    model.save_model("catboost_model.cbm")
    task.close()


if __name__ == "__main__":
    main()
