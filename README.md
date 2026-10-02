# Transition-Aware Human Activity Recognition Using Deep Learning and Gradient Boosting

> A multi-model framework for sensor-based Human Activity Recognition (HAR) combining CNN, DeepConvLSTM, SDAE, MLP, and LightGBM — with a focus on transition-aware classification of human activities from smartphone inertial sensor data.

---

## 📌 Project Overview

This project implements and compares multiple deep learning and machine learning approaches for classifying human activities from raw smartphone sensor signals (accelerometer + gyroscope). Unlike standard HAR systems, this work emphasizes **transition-awareness** — the ability to correctly classify ambiguous in-between activity states that sit at the boundary of static and dynamic movement classes.

### Activities Recognized

| Type | Activities |
|------|-----------|
| Static Postures | STANDING, SITTING, LYING |
| Dynamic Activities | WALKING, WALKING UPSTAIRS, WALKING DOWNSTAIRS |

---

## 📊 Results

**Table 1.** Performance comparison of all implemented models on the HAPT smartphone sensor dataset.

| Method | Accuracy | Precision | Recall | F1-score |
|--------|:--------:|:---------:|:------:|:--------:|
| LightGBM | **96.33** | **96.58** | **96.37** | **96.43** |
| DeepConvLSTM | 95.66 | 95.71 | 95.84 | 95.72 |
| CNN | 95.29 | 95.46 | 95.50 | 95.47 |
| MLP | 93.81 | 93.97 | 94.04 | 93.85 |
| SDAE | 78.28 | 78.83 | 78.47 | 78.25 |

---

## 🗂️ Project Structure

```
.
├── configs/
│   └── default.json          # Hyperparameters for all models
├── data/
│   └── hapt_data_set/        # Raw sensor dataset (place dataset here)
├── logs/                     # Training logs
├── models/
│   ├── cnn.py                # CNN architecture
│   ├── deep_conv_lstm.py     # DeepConvLSTM architecture
│   ├── lgbm.py               # LightGBM model
│   ├── mlp.py                # MLP architecture
│   ├── sdae.py               # Stacked Denoising AutoEncoder
│   └── tunings/              # Hyperparameter tuning scripts
├── notebooks/
│   └── plot_raw_data.ipynb   # Exploratory data analysis
├── src/
│   ├── data_prep/
│   │   ├── create_features.py      # Feature engineering (621 features)
│   │   ├── load.py                 # Data loading utilities
│   │   ├── preprocess_raw_data.py  # Raw signal preprocessing
│   │   └── preprocessing.py       # Windowing & normalization
│   ├── keras_callback.py           # Custom Keras callbacks
│   └── utils.py                    # Shared utilities
├── run_cnn.py                # Train & evaluate CNN
├── run_deep_conv_lstm.py     # Train & evaluate DeepConvLSTM
├── run_generate_features.py  # Generate engineered features
├── run_lgbm.py               # Train & evaluate LightGBM
├── run_mlp.py                # Train & evaluate MLP
├── run_sdae.py               # Train & evaluate SDAE
├── Dockerfile                # CPU Docker environment
├── Dockerfile_gpu            # GPU Docker environment
├── docker-compose.yml        # Docker Compose config
├── pyproject.toml            # Python dependencies (Poetry)
└── Makefile                  # Convenience commands
```

---

## ⚙️ Setup

### Option 1: Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/KOD-GIT/Transition-Aware-Human-Activity-Recognition-Using-Deep-Learning-and-Gradient-Boosting-.git
cd Transition-Aware-Human-Activity-Recognition-Using-Deep-Learning-and-Gradient-Boosting-

# Start GPU container
make start-gpu

# Install dependencies inside container
poetry install
```

### Option 2: Local with Poetry

```bash
# Install Poetry if not already installed
pip install poetry

# Install project dependencies
poetry install
```

### Requirements

- Python 3.8+
- TensorFlow / Keras 2.3
- LightGBM 3.0+
- pandas, scikit-learn, matplotlib, seaborn, shap, statsmodels

See [`pyproject.toml`](./pyproject.toml) for the full list.

---

## 📁 Dataset

This project uses the **HAPT (Human Activities and Postural Transitions)** dataset from the UCI Machine Learning Repository.

- Place the dataset under: `data/hapt_data_set/`
- The dataset includes 6 activity classes recorded via smartphone accelerometer and gyroscope at 50 Hz
- Download: [UCI HAPT Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

---

## 🚀 Running the Models

```bash
# Generate engineered features (run first)
poetry run python run_generate_features.py

# Train and evaluate LightGBM
poetry run python run_lgbm.py

# Train and evaluate CNN
poetry run python run_cnn.py

# Train and evaluate DeepConvLSTM
poetry run python run_deep_conv_lstm.py

# Train and evaluate SDAE
poetry run python run_sdae.py

# Train and evaluate MLP
poetry run python run_mlp.py
```

---

## 🧠 Methods

### Feature Engineering + LightGBM
Raw sensor signals are transformed into **621 handcrafted features** using statistical measures, FFT, and signal processing techniques. A LightGBM classifier is trained via 5-fold cross-validation.

### Convolutional Neural Network (CNN)
A sliding-window CNN processes raw sensor windows directly. Serves as a deep learning baseline.

### Deep Convolutional LSTM (DeepConvLSTM)
Extends the CNN by replacing fully-connected layers with LSTM layers to capture temporal dependencies in the sensor signal.

### Stacked Denoising AutoEncoder (SDAE)
Unsupervised pre-training via denoising autoencoders, followed by supervised fine-tuning with a softmax classifier.

### Multi-Layer Perceptron (MLP)
A simple two-hidden-layer feedforward network trained on raw sensor windows.

---

## 🙋 Author

**KOD** — [GitHub Profile](https://github.com/KOD-GIT)
