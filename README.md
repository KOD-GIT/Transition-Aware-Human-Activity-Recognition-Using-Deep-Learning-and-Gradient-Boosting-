# Transition-Aware Human Activity Recognition Using Deep Learning and Gradient Boosting

> A multi-model framework for **12-class** Human Activity Recognition (HAR) using the HAPT dataset — classifying both basic activities and postural transitions from smartphone inertial sensor data.

---

## 📌 Project Overview

This project implements and compares four approaches for classifying human activities and **postural transitions** from raw smartphone sensor signals (accelerometer + gyroscope). The key distinction of this project is **transition-awareness**: correctly identifying the six transitory states between static postures that are typically ignored in standard HAR benchmarks.

### Dataset: HAPT (Human Activities and Postural Transitions)

**Source:** UCI Machine Learning Repository  
**Signals:** Raw 3-axial acceleration and 3-axial angular velocity sampled at 50 Hz across 30 subjects

| Type | Classes |
|------|---------|
| Basic Activities (6) | Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Lying |
| Postural Transitions (6) | Stand-to-Sit, Sit-to-Stand, Sit-to-Lie, Lie-to-Sit, Stand-to-Lie, Lie-to-Stand |
| **Total** | **12 classes** |

---

## 👥 Team & Contribution Plan

| Member | Branch | Responsibility |
|--------|--------|----------------|
| Shreemoy Das | `feature/lightgbm` | Transition-aware feature generation, class balancing, LightGBM pipeline |
| Vinayak Shankar | `feature/mlp` | Engineered-feature vs raw-window MLP — code and report alignment |
| Sharath P | `feature/1d-cnn` | 12-class 1D-CNN and transition-class evaluation |
| Suraj Shuban | `feature/deepconvlstm` | 12-class DeepConvLSTM, temporal windowing and transition analysis |
| Shreemoy Das *(lead)* | `docs/project-alignment` | README, attribution, repository structure and integration reviews |

---

## 🗂️ Project Structure

```
.
├── configs/
│   └── default.json              # Hyperparameters for all models (12-class output)
├── data/
│   └── hapt_data_set/            # Raw HAPT sensor dataset (place dataset here)
├── logs/                         # Training logs
├── models/
│   ├── cnn.py                    # 1D-CNN architecture (12-class)
│   ├── deep_conv_lstm.py         # DeepConvLSTM architecture (12-class)
│   ├── lgbm.py                   # LightGBM model
│   ├── mlp.py                    # MLP architecture
│   └── tunings/                  # Hyperparameter tuning scripts
├── notebooks/                    # Exploratory data analysis
├── src/
│   ├── data_prep/
│   │   ├── create_features.py    # Feature engineering (transition-aware)
│   │   ├── load.py               # Data loading (all 12 classes)
│   │   ├── preprocess_raw_data.py
│   │   └── preprocessing.py     # Windowing & normalization
│   ├── keras_callback.py         # Custom Keras callbacks
│   └── utils.py                  # Shared utilities
├── run_cnn.py                    # Train & evaluate 1D-CNN
├── run_deep_conv_lstm.py         # Train & evaluate DeepConvLSTM
├── run_generate_features.py      # Generate engineered features
├── run_lgbm.py                   # Train & evaluate LightGBM
├── run_mlp.py                    # Train & evaluate MLP
├── pyproject.toml                # Python dependencies (Poetry)
└── Makefile                      # Convenience commands
```

---

## ⚙️ Setup

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/KOD-GIT/Transition-Aware-Human-Activity-Recognition-Using-Deep-Learning-and-Gradient-Boosting-.git
cd Transition-Aware-Human-Activity-Recognition-Using-Deep-Learning-and-Gradient-Boosting-

make start-gpu
poetry install
```

### Option 2: Local with Poetry

```bash
pip install poetry
poetry install
```

---

## 📁 Dataset

Place the HAPT dataset under: `data/hapt_data_set/`

Download: [UCI HAPT Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

> ⚠️ The dataset contains **12 activity classes** — 6 basic activities and 6 postural transitions. All pipeline components must handle all 12 classes.

---

## 🚀 Running the Models

```bash
# Step 1: Generate engineered features (required before LightGBM/MLP)
poetry run python run_generate_features.py

# Train and evaluate each model
poetry run python run_lgbm.py
poetry run python run_cnn.py
poetry run python run_deep_conv_lstm.py
poetry run python run_mlp.py
```

---

## 🧠 Methods

### Feature Engineering + LightGBM
Transition-aware handcrafted features (statistical, FFT, signal processing) fed into a LightGBM classifier with class-balancing for the underrepresented transition classes.

### 1D Convolutional Neural Network (CNN)
Sliding-window CNN processing raw sensor signals across all 12 activity classes including transitions.

### Deep Convolutional LSTM (DeepConvLSTM)
Extends the CNN with LSTM layers to capture temporal dependencies — especially useful for detecting short-duration transition events.

### Multi-Layer Perceptron (MLP)
Feedforward network — input strategy (engineered features vs raw windows) to be determined and aligned with the report.

---

## ⚠️ Known Issues & TODO

- [ ] `configs/default.json` — update `num_class` from 6 → 12
- [ ] `src/data_prep/load.py` — include all 6 postural transition classes
- [ ] All runner scripts — update prediction arrays and confusion matrices for 12 classes
- [ ] `logs/logger_lgbm.py` — currently missing, must be created before LightGBM runs
- [ ] MLP strategy — decide engineered features vs raw windows and align with report
- [ ] Add transition-class precision, recall, and F1 to evaluation metrics
- [ ] Add `notebooks/` EDA notebook for raw signal visualization

---

## 📝 First Commits Planned

```
docs: align README with twelve-class HAPT project
fix(data): include six postural-transition classes in loader
fix(config): change model output to twelve classes
fix(lgbm): restore missing logging dependency
feat(metrics): add transition-class precision, recall and F1
```

---

## 🙋 Authors

**Team KOD** — [GitHub](https://github.com/KOD-GIT)

> This project extends and substantially modifies a six-class HAR baseline to support twelve-class transition-aware classification as part of ICT 4442.
