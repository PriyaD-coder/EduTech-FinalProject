# Cryptocurrency Price Predictor

## Overview

The Cryptocurrency Price Predictor is a Deep Learning-based application that predicts the future price of Bitcoin using historical market data. The project uses a Long Short-Term Memory (LSTM) neural network, which is well-suited for time-series forecasting. Historical Bitcoin prices are collected using the Yahoo Finance API, preprocessed, and used to train the model. The trained model predicts the next day's Bitcoin closing price and displays the results through a Streamlit web application.

---

## Features

- Download real-time Bitcoin historical data using Yahoo Finance API.
- Data preprocessing and normalization using MinMaxScaler.
- Time-series forecasting using LSTM (Long Short-Term Memory).
- Predict future Bitcoin prices.
- Interactive Streamlit dashboard.
- Actual vs Predicted Price visualization.
- Model evaluation using RMSE.
- Save trained model for future predictions.

---

## Technology Stack

| Technology        | Purpose                   |
|-------------------|---------------------------|
| Python            | Programming Language      |
| TensorFlow        | Deep Learning Framework   |
| Keras             | LSTM Model Development    |
| Pandas            | Data Processing           |
| NumPy             | Numerical Computing       |
| Matplotlib        | Data Visualization        |
| Scikit-learn      | Data Scaling & Evaluation |
| Streamlit         | Web Application           |
| Yahoo Finance API | Historical Bitcoin Data   |

---

## Project Structure

```
Cryptocurrency-Price-Predictor/
│
├── dataset/
│   └── bitcoin.csv
│
├── graphs/
│   └── prediction.png
│
├── models/
│   └── lstm_model.keras
│
├── fetch_data.py
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
└── Internship_Report.pdf
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Cryptocurrency-Price-Predictor.git
```

Navigate to the project folder:

```bash
cd Cryptocurrency-Price-Predictor
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Project Workflow

### Step 1: Download Dataset

```bash
python fetch_data.py
```

This downloads historical Bitcoin data and saves it as:

```
dataset/bitcoin.csv
```

---

### Step 2: Train the LSTM Model

```bash
python train.py
```

This will:

- Load Bitcoin historical data
- Normalize the dataset
- Train the LSTM model
- Save the trained model
- Generate prediction graph

Output:

```
models/lstm_model.keras
graphs/prediction.png
```

---

### Step 3: Predict Future Price

```bash
python predict.py
```

Example Output:

```
Predicted Bitcoin Price

Tomorrow's Predicted Price: 
$118,250.35
```

---

### Step 4: Launch Web Application

```bash
python -m streamlit run app.py
```

The application will open automatically in your browser.

Default URL:

```
http://localhost:8501
```

---

## Dataset

The dataset is collected using the Yahoo Finance API.

Dataset includes:

- Date
- Open Price
- High Price
- Low Price
- Close Price
- Volume

---

## LSTM Model Architecture

- Input Layer
- LSTM Layer (64 Units)
- Dropout Layer (20%)
- LSTM Layer (64 Units)
- Dropout Layer
- Dense Layer
- Output Layer

Optimizer:

```
Adam
```

Loss Function:

```
Mean Squared Error (MSE)
```

---

## Evaluation Metric

The model performance is evaluated using:

- Root Mean Squared Error (RMSE)

Lower RMSE indicates better prediction accuracy.

---

## Screenshots

### Home Page

- Streamlit Dashboard

### Model Training

- Epoch-wise Training

### Prediction Graph

- Actual vs Predicted Bitcoin Price

### Prediction Result

- Tomorrow's Bitcoin Price

---

## Future Enhancements

- Multi-Cryptocurrency Prediction (Ethereum, Solana, Dogecoin)
- 7-Day and 30-Day Forecasting
- News Sentiment Analysis
- Real-Time Market Dashboard
- Cloud Deployment
- Mobile Application
- Trading Signal Generation

---

## Learning Outcomes

Through this project, the following skills were developed:

- Deep Learning
- Time-Series Forecasting
- LSTM Neural Networks
- Data Preprocessing
- API Integration
- Model Evaluation
- Streamlit Web Development
- Financial Data Analysis

---

## Author

**Priya Dattatray Deshmukh**

Data Science Internship Project

EduTech Solutions

---

## Acknowledgements

I sincerely thank EduTech Solutions and my mentor for their continuous guidance and support throughout the internship. I also acknowledge the developers of TensorFlow, Keras, Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib, and Yahoo Finance API for providing excellent open-source tools that made this project possible.

---

## License

This project is developed for educational and internship purposes only.
