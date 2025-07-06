# TransGuard-AI-Real-Time-Anomaly-Detection-for-Financial-Transactions-Using-Interactive-Dashboards

🚀 TransGuard AI
Real-Time Fraud Detection Dashboard for Financial Transactions

## 🌟 Overview
TransGuard AI is a machine learning project designed to detect fraudulent financial transactions in real-time. It uses machine learning models, including Random Forest and Isolation Forest, to analyze transaction data and identify potential fraud. The solution includes an interactive Streamlit dashboard for monitoring and exploring model results.

This project was developed as part of an academic study comparing supervised and unsupervised learning methods for fraud detection.

## 🔎 Features
✅ Real-time fraud detection via interactive dashboard
✅ Random Forest supervised classification model
✅ Isolation Forest unsupervised anomaly detection
✅ Visual analytics for fraud investigation
✅ Ability to simulate new transactions via sliders
✅ Display of confusion matrices, ROC curves, and classification metrics
✅ Easily extensible for other fraud datasets

## 📊 Dataset
This project uses the Credit Card Fraud Detection dataset publicly available from Kaggle. The dataset contains transactions made by European cardholders in September 2013.

~284,807 transactions

Only ~0.17% are fraudulent

Features: PCA components (V1 to V28), Amount, and Time

## Note: 
Due to confidentiality, the dataset is not included in this repository. Please download it directly from Kaggle.


## 🛠️ Installation
Clone the repo:

git clone https://github.com/DataMindMedic/TransGuard-AI-Real-Time-Anomaly-Detection-for-Financial-Transactions-Using-Interactive-Dashboards

## Install requirements:

pip install -r requirements.txt
Or manually install key libraries:
pip install streamlit scikit-learn pandas numpy matplotlib seaborn

## 🚀 How to Run the Dashboard
Start the Streamlit app:
streamlit run dashboard/app.py
Then open the provided local URL in your web browser (e.g. http://localhost:8501).

## ⚙️ Training and Evaluation
### Random Forest
The Random Forest model can be trained and evaluated via the script:
python models/random_forest_model.py

## This script:

Splits the dataset
Trains the Random Forest classifier
Prints classification metrics
Saves the model to disk (e.g. random_forest_model.pkl)

### Isolation Forest
To run experiments with Isolation Forest:

python models/isolation_forest_model.py

##This script:

Fits Isolation Forest on the entire dataset
Calculates metrics and confusion matrix
Saves results for comparison

## 🧪 Model Comparison
The dashboard and notebooks in this project allow you to:

Compare supervised vs. unsupervised approaches
Visualize confusion matrices

## Examine model metrics:

Accuracy
Precision
Recall
F1-score
ROC AUC

##📷 Screenshots


## 🤝 Contributing
Pull requests are welcome! Feel free to submit issues for:
Feature suggestions
Bug reports
Documentation improvements

## 📄 License
This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgements

1. Kaggle Credit Card Fraud Detection Dataset

2. Streamlit team

3. scikit-learn contributors

4. Academic supervisors and reviewers

✨ Author
Micheal Omotosho
Email: datamindmedic@gmail.com
GitHub: https://github.com/DataMindMedic


##⭐️ Citation (for Academic Report)
If citing this project in your academic work:

Omotosho, M. (2025). TransGuard AI: Real-Time Fraud Detection using Machine Learning Models. 
