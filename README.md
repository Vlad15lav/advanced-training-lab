# 🚀 Advanced Training Lab  

⚡ **PyTorch Lightning — революция или просто удобный синтаксический сахар?**  
Этот репозиторий поможет вам разобраться на практике! 

<center><img src="images/repo-preview.png"></center>
<center>DALLE 3 о противостоянии PyTorch и Lightning</center>

## 📌 Описание  

**Advanced Training Lab** — это проект, созданный для демонстрации возможностей **PyTorch Lightning** ⚡ и **ClearML** 🛠️ при обучении моделей машинного обучения и глубокого обучения.  

---

## 📂 Структура проекта  

📁 **`src/`** — основной код проекта (модель, датасет, тренировка, тестирование).    
📁 **`src/data/`** — скрипты для загрузки и инициализации дата модулей.  
📁 **`src/models/`** — скрипты для создания моделей.  
📁 **`src/test/`** — скрипты для тестирования моделей.  
📁 **`src/utils/`** — вспомогательные функции.  
🐍 **`src/train_sign_language.py`** — скрипт для обучения модели на датасете жестового языка.  
🐍 **`src/demo_sign_language.py`** — скрипт для инференса модели на одном примере датасета жестового языка.  
📄 **`requirements.txt`** — список зависимостей.  
📄 **`README.md`** — это файл, который Вы сейчас читаете.  

---

## 🚀 Как запустить проект  

1️⃣ Клонируем проект из удаленного репозитория 🛠️:  
```bash
git clone https://github.com/Vlad15lav/advanced-training-lab.git
cd ./advanced-training-lab
```

2️⃣ Установите зависимости 📦:  
```bash
pip install -r requirements.txt
```

3️⃣ Перейдите в директорию с кодом 🗂️:  
```bash
cd ./src
```

4️⃣ Запустите обучение модели 🎓:  
4️⃣.1️⃣ Запуск обучение модели перевода с языка жестов на английский.  
Параметр **fast-dev-run** для проверки корректности пайплайна обучения.  
```bash
python train_sign_language.py --fast-dev-run
```
4️⃣.2️⃣ Запуск обучение модели генерации рукописных цифр.    
```bash
python train_digits_generator.py --epoch 20 --debug-samples-epoch 2
```

5️⃣ Запустите инференс на одном тестовом примере 🧪:  
5️⃣.1️⃣ Для модели перевода с языка жестов на английский.  
```bash
python demo_sign_language.py
```
5️⃣.2️⃣ Для модели генерации рукописных цифр.
```bash
python demo_digits_generator.py --samples 9 --output_path ./generated_digits_grid.png
```

## 🔗 Полезные ссылки

- [Welcome to ⚡ PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [The Infrastructure Platform For AI Builders](https://clear.ml/)
- [🤖 Машинное обучение с помощью ClearML и Pytorch Lightning ⚡](https://stepik.org/course/214389)
