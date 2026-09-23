# Machine Learning Based Intrusion Detection System (ML-IDS)

A Machine Learning Based Intrusion Detection System (ML-IDS) developed as a college mini project to identify potentially malicious network traffic using supervised and unsupervised machine learning techniques.

The system uses the **NSL-KDD dataset**, Random Forest, and Isolation Forest to classify network traffic as normal or suspicious. A Flask-based web interface allows users to analyze manually entered network-flow features and upload CSV datasets for prediction and performance evaluation.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [Features](#features)
* [Technologies Used](#technologies-used)
* [Dataset](#dataset)
* [System Architecture](#system-architecture)
* [Machine Learning Algorithms](#machine-learning-algorithms)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [How to Run](#how-to-run)
* [Using the Web Application](#using-the-web-application)
* [CSV Dataset Evaluation](#csv-dataset-evaluation)
* [Performance Evaluation](#performance-evaluation)
* [Limitations](#limitations)
* [Future Enhancements](#future-enhancements)
* [Disclaimer](#disclaimer)

---

## Project Overview

Network security is an important part of protecting computer systems from unauthorized access and malicious activities. An Intrusion Detection System (IDS) monitors network-related information to identify potentially suspicious behavior.

This project implements a machine learning-based IDS prototype that analyzes network connection features and predicts whether traffic is normal or represents a potential attack.

The system combines two machine learning approaches:

1. **Random Forest:** A supervised learning algorithm trained using labelled network traffic.
2. **Isolation Forest:** An anomaly detection algorithm used to identify unusual traffic patterns.

The application also includes a Flask web interface that supports manual network-flow analysis and CSV-based prediction.

The project is intended for educational purposes and demonstrates how machine learning can be applied to network intrusion detection.

---

## Objectives

The main objectives of this project are:

* To understand the application of machine learning in network intrusion detection.
* To preprocess and analyze network traffic data using the NSL-KDD dataset.
* To develop a supervised classification model using Random Forest.
* To explore anomaly detection using Isolation Forest.
* To combine model predictions into a single detection verdict.
* To provide a web-based interface for traffic analysis.
* To evaluate model performance using standard classification metrics.
* To demonstrate how machine learning can support cybersecurity analysis.

---

## Features

### 1. Random Forest Classification

Uses a supervised machine learning model trained on labelled network traffic to classify connections as normal or attack traffic.

### 2. Isolation Forest Anomaly Detection

Uses an unsupervised anomaly detection approach to identify traffic patterns that differ from the normal traffic used during training.

### 3. Combined Detection

Combines the Random Forest and Isolation Forest outputs.

The combined detector reports an attack when either model identifies the traffic as an attack or anomaly.

### 4. Manual Network-Flow Analysis

Allows users to enter selected network-flow features through the web interface and receive model predictions.

### 5. CSV Dataset Upload

Allows users to upload a CSV file containing network-flow features for batch prediction.

### 6. Performance Metrics

For labelled CSV data, the application can calculate classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

### 7. Downloadable Predictions

Provides prediction results as a downloadable CSV file when the upload endpoint completes successfully.

### 8. Preset Traffic Examples

Includes predefined examples for demonstrating normal traffic and selected attack scenarios through the manual analyzer.

---

## Technologies Used

| Technology   | Purpose                                    |
| ------------ | ------------------------------------------ |
| Python       | Main programming language                  |
| Flask        | Web application and prediction endpoints   |
| Pandas       | Dataset loading and manipulation           |
| NumPy        | Numerical operations                       |
| Scikit-learn | Machine learning algorithms and evaluation |
| Joblib       | Saving and loading trained models          |
| HTML         | Web interface structure                    |
| CSS          | Web interface styling                      |
| JavaScript   | Client-side interactions and API requests  |
| NSL-KDD      | Network intrusion detection dataset        |

---

## Dataset

This project uses the **NSL-KDD dataset**, a widely used dataset for research and educational experiments involving network intrusion detection.

The dataset contains network connection records with features describing traffic characteristics and labels indicating normal or attack activity.

### Dataset features

The NSL-KDD records contain 41 input features, including:

* Basic connection information
* Content-related features
* Traffic statistics
* Host-based traffic statistics
* Categorical features such as protocol, service, and connection flag

The original records also contain an attack label and a difficulty-level field.

### Data preprocessing

The preprocessing stage includes:

1. Loading the dataset.
2. Separating the input features and target label.
3. Removing the difficulty-level field from the model input.
4. Converting the original labels into binary classes: Normal (0) and Attack (1).
5. Encoding categorical features.
6. Scaling numerical features where required by the implementation.
7. Preparing the processed data for model training and prediction.

**Important:** The training and test datasets must remain separate. Performance reported as test performance should be calculated using data that was not used to fit the models.

---

## System Architecture

```text
             NSL-KDD Dataset
                    |
                    v
             Data Preprocessing
                    |
                    v
             Feature Preparation
                    |
          +---------+---------+
          |                   |
          v                   v
    Random Forest       Isolation Forest
    Classification     Anomaly Detection
          |                   |
          v                   v
      RF Verdict          IF Verdict
          |                   |
          +---------+---------+
                    |
                    v
             Combined Detection
                    |
                    v
             Normal / Attack
                    |
                    v
              Flask Web App
                    |
                    v
          Prediction Results
```

The web application accepts manual feature inputs or CSV data, preprocesses the supplied features using the saved preprocessing objects, and obtains predictions from the trained models.

---

## Machine Learning Algorithms

### 1. Random Forest

Random Forest is a supervised ensemble learning algorithm that combines predictions from multiple decision trees.

In this project, it is trained using labelled network traffic to distinguish normal connections from attack connections.

**Purpose:** Binary classification of network traffic.

### 2. Isolation Forest

Isolation Forest is an anomaly detection algorithm that identifies observations that are easier to isolate from the rest of the data.

In this project, it is trained using normal traffic samples to identify potential anomalies.

**Purpose:** Detect unusual traffic patterns that may indicate attacks.

### 3. Combined Detection

The combined detector uses an OR-based decision rule:

* If Random Forest predicts Attack, the combined verdict is Attack.
* If Isolation Forest flags an anomaly, the combined verdict is Attack.
* If neither model flags the traffic, the combined verdict is Normal.

This approach can increase attack detection, but it may also increase false positives. Its performance should therefore be evaluated separately from the individual models.

---

## Project Structure

```text
ML-Based-Intrusion-Detection-System/
│
├── app.py
├── train.py
├── data_utils.py
├── predict.py
├── web_demo.html
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── Trained model and preprocessing files
│
├── dataset/
│   └── NSL-KDD dataset files (if included)
│
└── screenshots/
    └── Application screenshots
```

The `models/`, `dataset/`, and `screenshots/` directories are illustrative. Keep the actual filenames and directory structure consistent with your implementation.

---

## Installation

### Prerequisites

Make sure the following are installed:

* Python 3
* pip
* Git (optional, for cloning the repository)

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ML-Based-Intrusion-Detection-System.git
```

Move into the project directory:

```bash
cd ML-Based-Intrusion-Detection-System
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

Ensure `requirements.txt` contains the dependencies actually used by the project.

---

## How to Run

### Step 1: Prepare the trained models

If the trained model and preprocessing files are included in the repository, ensure they are in the locations expected by `app.py`.

Otherwise, train the models using the project's training script and verify that the generated artifacts are saved to the expected paths.

```bash
python train.py
```

### Step 2: Start the Flask application

```bash
python app.py
```

### Step 3: Open the web interface

Open the following address in your browser:

```text
http://localhost:5000
```

The application should display the ML-IDS web interface.

---

## Using the Web Application

### Manual traffic analysis

1. Open the web application.
2. Enter the available network-flow feature values.
3. Submit the form for prediction.
4. Review the Random Forest, Isolation Forest, and combined verdicts displayed by the application.

The manual analyzer uses the features provided by the form. Any features not supplied by the user may be filled using application defaults, so the result may not represent a complete network connection record.

### Preset examples

Use the available presets to demonstrate selected normal and attack traffic examples.

The presets are intended for demonstration and do not replace evaluation on an independent test dataset.

---

## CSV Dataset Evaluation

The CSV upload functionality supports batch prediction on network-flow records.

### CSV requirements

The CSV should contain:

* The 41 expected NSL-KDD input feature columns.
* An optional label column, such as `attack_type` or `label`, if evaluation metrics are required.

The column names and feature order must be compatible with the preprocessing pipeline used by the application.

### Steps

1. Open the web interface.
2. Locate the CSV Dataset Evaluation section.
3. Select a correctly formatted CSV file.
4. Upload the file.
5. Review the predictions and any evaluation metrics returned.
6. Download the generated prediction CSV using the provided download link.

If the uploaded CSV contains labels, the application can compare the predictions against those labels and calculate evaluation metrics. If the CSV has no labels, classification metrics cannot be calculated.

Use the actual NSL-KDD test set for test evaluation rather than uploading the training set and presenting its results as unseen-data performance.

---

## Performance Evaluation

The models should be evaluated using a separate test dataset.

The following metrics can be used to assess classification performance:

| Metric           | Description                                             |
| ---------------- | ------------------------------------------------------- |
| Accuracy         | Proportion of all predictions that are correct          |
| Precision        | Proportion of predicted attacks that are actual attacks |
| Recall           | Proportion of actual attacks correctly detected         |
| F1-score         | Harmonic mean of precision and recall                   |
| Confusion Matrix | Shows correct and incorrect predictions by class        |

### Results

The following table presents the model performance results obtained during evaluation on the NSL-KDD test dataset.

| Model              |           Accuracy |          Precision |             Recall |           F1-score |
| ------------------ | -----------------: | -----------------: | -----------------: | -----------------: |
| Random Forest      | 77.27% | 96.67% | 62.21% | 75.70% |
| Isolation Forest   | 79.69% | 97.14% | 66.27% | 78.79% |
| Combined Detection | 85.09% | 96.63% | 76.47% | 85.38% |

### Combined Model Confusion Matrix

| Actual / Predicted | Normal (0) | Attack (1) |
|--------------------|-----------|------------|
| Normal (0)         | 9,369     | 342        |
| Attack (1)         | 3,019     | 9,814      |

Rows represent actual classes, and columns represent predicted classes.
Class 0 = Normal and Class 1 = Attack.

**Evaluation note:** Clearly state which dataset split was used, how labels were mapped, and whether the values are from binary classification or another evaluation task. Do not mix training metrics with test metrics.

---

## Limitations

* The system uses the NSL-KDD dataset, which does not represent every type of modern network traffic or attack.
* The application is a dataset-based prototype and does not capture live network packets.
* Manual analysis uses a limited set of user-entered features and may fill missing features with defaults.
* Model predictions depend on the quality and compatibility of the input features and preprocessing.
* The combined detection rule may increase false positives.
* Results obtained on NSL-KDD should not be interpreted as guaranteed detection performance in a production network.

---

## Future Enhancements

Possible future improvements include:

* Multiclass attack classification, such as DoS, Probe, R2L, and U2R.
* A prediction-history feature.
* A visual performance dashboard.
* Improved input validation and error handling.
* Additional testing on suitable datasets.
* Live traffic collection and analysis in a controlled lab environment.
* Improved reporting and export options.

These are potential extensions and are not claimed as implemented unless they are added and tested.

---

## Disclaimer

This project is developed for educational, academic, and research purposes.

It is intended to demonstrate machine learning techniques for network intrusion detection. It should not be considered a production-ready security monitoring solution or a replacement for professional security tools.

Only test systems and network traffic that you own or have explicit authorization to analyze.

---

