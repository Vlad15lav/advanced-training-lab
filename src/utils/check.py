import os
import warnings

from getpass import getpass


REQUIRED_ENV = (
    "CLEARML_WEB_HOST",
    "CLEARML_API_HOST",
    "CLEARML_FILES_HOST",
    "CLEARML_API_ACCESS_KEY",
    "CLEARML_API_SECRET_KEY"
)


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
