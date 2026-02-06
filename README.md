# 🏛️ MonuVision AI - Monument Classification

AI-powered Indian Monument Recognition using EfficientNetV2 with a stunning Gen Z aesthetic frontend.

![MonuVision AI](https://img.shields.io/badge/AI-Monument%20Recognition-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)

## ✨ Features

- 🎯 **95%+ Accuracy** - EfficientNetV2 with advanced augmentation
- 🖼️ **24 Indian Monuments** - Taj Mahal, Red Fort, Qutub Minar, and more
- 🚀 **TTA Support** - Test-Time Augmentation for higher accuracy
- 🎨 **Gen Z Aesthetic** - Three.js particle globe, glassmorphism UI
- 📱 **Responsive Design** - Works on desktop and mobile

## 🛠️ Tech Stack

- **Backend**: Python, TensorFlow/Keras, Flask
- **Frontend**: HTML5, CSS3, JavaScript, Three.js
- **Model**: EfficientNetV2-S with transfer learning
- **Dataset**: [Indian Monuments Dataset](https://www.kaggle.com/datasets/danushkumarv/indian-monuments-image-dataset)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
```bash
python train.py
```

### 3. Start the API Server
```bash
python api/server.py
```

### 4. Open the Frontend
Open `frontend/index.html` in your browser, or serve it:
```bash
cd frontend && python -m http.server 8080
```

## 📁 Project Structure

```
monument-classification/
├── api/
│   └── server.py          # Flask REST API
├── frontend/
│   ├── index.html         # Main page
│   ├── styles.css         # Gen Z aesthetic styles
│   └── app.js             # Three.js + API integration
├── models/                # Trained models (gitignored)
├── config.py              # Configuration settings
├── model.py               # EfficientNetV2 architecture
├── train.py               # Training pipeline
├── predict.py             # Prediction utilities
├── data_loader.py         # Dataset loading + augmentation
└── requirements.txt       # Python dependencies
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/classes` | List monument classes |
| POST | `/predict` | Predict monument (add `?tta=true` for TTA) |

## 🎯 Supported Monuments

Ajanta Caves, Charminar, Gateway of India, Golden Temple, Hampi, Hawa Mahal, India Gate, Khajuraho, Konark Sun Temple, Lotus Temple, Meenakshi Temple, Mysore Palace, Qutub Minar, Red Fort, Sanchi Stupa, Statue of Unity, Taj Mahal, Thanjavur Temple, Victoria Memorial, and more...

## 📝 License

MIT License - feel free to use this project!

---

Built with ❤️ by [Nikhil Mana](https://github.com/NikhilMana)
