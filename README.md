---

# Heart Failure Prediction Using Ensemble Learning Methods

This project focuses on predicting heart failure using ensemble learning techniques and comparing their performance. The implemented algorithms include **Random Forest**, **AdaBoost**, **XGBoost**, and **LightGBM**, alongside a simple neural network built with Keras. The dataset utilized is the [Heart Disease Dataset from Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data), featuring 13 clinical attributes and a binary target variable indicating the presence or absence of heart disease.

## Project Overview

Ensemble learning leverages multiple models to enhance prediction accuracy. This project explores **Bagging** (e.g., Random Forest) and **Boosting** (e.g., AdaBoost, XGBoost, LightGBM) techniques, supplemented by a neural network experiment. Model performance is assessed using metrics like Accuracy, Recall, Precision, F1 Score, Sensitivity, Specificity, and ROC AUC.

## Dataset Information

- **Source**: [Heart Disease Dataset from Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **Features**:
  - `age`: Patient's age
  - `sex`: Gender (1 = male, 0 = female)
  - `cp`: Chest pain type
  - `trestbps`: Resting blood pressure
  - `chol`: Serum cholesterol
  - `fbs`: Fasting blood sugar (>120 mg/dl, 1 = true, 0 = false)
  - `restecg`: Resting electrocardiographic results
  - `thalach`: Maximum heart rate achieved
  - `exang`: Exercise-induced angina (1 = yes, 0 = no)
  - `oldpeak`: ST depression induced by exercise
  - `slope`: Slope of the peak exercise ST segment
  - `ca`: Number of major vessels colored by fluoroscopy
  - `thal`: Thalassemia type
- **Target Variable**: Binary (0 = no heart disease, 1 = heart disease present)

## Algorithms Used

- **Random Forest**: A Bagging method using multiple decision trees.
- **AdaBoost**: A Boosting technique that iteratively corrects errors.
- **XGBoost**: An efficient, high-performance Boosting algorithm.
- **LightGBM**: A Boosting method optimized for large datasets.
- **Neural Network (Keras)**: A basic neural network for binary classification.

## Performance Metrics

The following metrics were used to evaluate the models:
- **Accuracy**: Overall correctness of predictions
- **Recall (Sensitivity)**: True positive rate
- **Precision**: Positive predictive value
- **F1 Score**: Harmonic mean of Precision and Recall
- **Specificity**: True negative rate
- **Cohen's Kappa**: Agreement beyond chance
- **ROC AUC**: Area under the Receiver Operating Characteristic curve

## Results

The **XGBoost** model outperformed others with the following metrics:
- **Accuracy**: 98.05%
- **Recall (Sensitivity)**: 96.05%
- **Precision**: 100.00%
- **F1 Score**: 0.98
- **Specificity**: 100.00%
- **Cohen's Kappa**: 0.96
- **ROC AUC**: 0.99

These results highlight XGBoost's superior accuracy and reliability, with no false positives. Performance details for other models are also available in the project for comparison.

## Conclusion

This project underscores the effectiveness of ensemble learning, particularly **XGBoost**, in heart failure prediction. The comparison with Random Forest, AdaBoost, LightGBM, and a neural network offers valuable insights into their performance for medical diagnosis, with XGBoost excelling in accuracy and robustness.

## References

- Pal, M., & Parija, S. (2020). *Prediction of Heart Diseases using Random Forest*.
- Charan, K. S., et al. (2022). *Comparison of Data Mining Algorithms for Heart Disease Prediction*.
- Yang, J., & Guan, G. (2022). *XGBoost for Heart Failure Prediction*.
- Islam, M., et al. (2022). *Ensemble Learning for Heart Disease Prediction*.
- Al-Dawood, I. (2019). *LightGBM for Large-Scale Heart Disease Prediction*.

