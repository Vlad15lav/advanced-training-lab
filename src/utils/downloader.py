import requests
import zipfile


def download_file(url, save_path):
    """Скачивает файл из интернета

    Args:
        url (str): URL-адрес файла
        save_path (str): путь к файлу, куда будет сохранен скачанный файл
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Файл скачан: {save_path}")


def unzip_file(zip_path, extract_to):
    """Распаковывает zip-архив

    Args:
        zip_path (str): путь к zip-архиву
        extract_to (str): путь к папке, куда будет распакован архив
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Файл распакован в: {extract_to}")
