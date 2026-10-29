# 🏥 SmartCare-UP: Public Healthcare Demand Analysis & OPD Volume Prediction

## 📌 Project Overview

**SmartCare-UP** is a healthcare data analytics and predictive modelling project that analyzes public healthcare service patterns across districts in Uttar Pradesh, India. It uses government Health Management Information System (HMIS) data to explore healthcare indicators and estimate reported Allopathic Outpatient Department (OPD) volume using machine learning algorithms.

Developed using Python and Streamlit, the project provides an interactive dashboard for exploring district-level healthcare data, visualizing service patterns, and comparing the performance of multiple regression models.

## 🎯 Key Features

* **Data Overview:** Displays dataset statistics, including records, districts, and available healthcare indicators.
* **Healthcare Indicator Exploration:** Enables users to select indicators and compare recorded values across districts.
* **District-Level Analysis:** Identifies variations in reported healthcare service indicators.
* **Predictive Modelling:** Estimates reported Allopathic OPD volume using five regression algorithms.
* **Model Comparison:** Evaluates models using MAE, RMSE, and R² scores.
* **Interactive Dashboard:** Presents visualizations, analytical findings, and model limitations.

## 🤖 Machine Learning Algorithms

The project compares five regression algorithms:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Extra Trees Regressor
5. Gradient Boosting Regressor

Gradient Boosting achieves the best RMSE in the current evaluation, with an MAE of approximately 232,332, an RMSE of approximately 340,031, and an R² score of 0.69.

## 📊 Dataset

The cleaned HMIS master dataset contains 900 records, with 525 records usable for modelling, 75 districts represented in the model data, and 80 healthcare indicators available for exploration.

## 🛠️ Technology Stack

* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Dashboard:** Streamlit
* **Data Format:** CSV

## 🚀 Installation and Execution

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
pip install -r requirements.txt
streamlit run app.py
```

Replace the repository placeholders with your actual GitHub repository details.

## 🔮 Future Scope

Future improvements include integrating regularly updated HMIS data, adding population and healthcare facility indicators, developing time-series forecasting with appropriate temporal validation, and enhancing district-level geographical visualizations.

## 💡 Project Impact

SmartCare-UP demonstrates how government healthcare data and machine learning can support the exploration of district-level service patterns and evidence-informed healthcare planning. Since the available records are sparse reporting snapshots and may be cumulative, the current model estimates recorded OPD volume rather than definitively forecasting future monthly demand.

**Author:** Krishnakant Yadav
**Domain:** Data Science | Healthcare Analytics | Machine Learning
