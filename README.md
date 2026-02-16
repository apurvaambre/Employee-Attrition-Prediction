# 📊 Employee Attrition Prediction

Employee Attrition Prediction is a machine learning project designed to predict whether an employee is likely to leave an organization based on historical workforce data.
The project demonstrates the complete ML workflow — from data preprocessing and model training to deployment through a web-based application.

---

## 📖 Project Title & Overview

Employee attrition is a major challenge for organizations. This project explores how predictive analytics can help identify potential turnover early.

The system includes:

* Data preprocessing and feature handling
* Training multiple machine learning models
* Saving trained models for deployment
* A web application interface for prediction

The repository demonstrates an end-to-end data science workflow including model development and application integration.

---

## ✨ Features

* 🤖 Machine Learning–based attrition prediction
* 📊 Data preprocessing and model training pipeline
* 🌐 Web interface for real-time predictions
* 💾 Saved models for reuse and deployment
* 🧩 Modular project structure

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Scikit-learn, Pandas, NumPy
* **Web Framework:** Flask
* **Modeling:** Machine Learning Classification
* **Environment:** Virtualenv / pip

---

## ⚙️ Installation Steps

### 1️⃣ Clone Repository

```bash
git clone https://github.com/apurvaambre/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage Instructions

### 🔹 Train Models

```bash
python train_models.py
```

This generates trained models inside the `models/` folder.

### 🔹 Run Web Application

```bash
python app.py
```

Open browser:

```
http://localhost:5000
```

Enter employee details to receive attrition predictions.

---

## 📂 Folder Structure

```
Employee-Attrition-Prediction/
│
├── data/               # Dataset files
├── models/             # Saved ML models
├── app.py              # Flask application
├── train_models.py     # Model training script
├── requirements.txt    # Dependencies
└── README.md
```

---

⭐ If you like this project, consider giving it a star!
