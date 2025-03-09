# 🚀 Advanced Training Lab  

![PyTorch Versus](images/pytorch_versus.webp)

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

1️⃣ Установите зависимости 📦:  
```bash
pip install -r requirements.txt
```

2️⃣ Запустите обучение модели 🎓:
```bash
python src/train_sign_language.py
```

3️⃣ Запустите инференс на одном тестовом примере 🧪:
```bash
python src/demo_sign_language.py
```

## 🔗 Полезные ссылки

- [Welcome to ⚡ PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [The Infrastructure Platform For AI Builders](https://clear.ml/)
- [🤖 Машинное обучение с помощью ClearML и Pytorch Lightning ⚡](https://stepik.org/course/214389)
