# Healthcare Premium Prediction

## Project Overview
This project aims to **predict annual healthcare insurance premiums** for customers using **machine learning models** based on demographic, lifestyle, and medical history features. The project focuses on **accurate premium estimation**, **reducing extreme prediction errors**, and providing actionable insights for insurance companies.

---

## Application / Live Demo
You can try the **Healthcare Premium Prediction app** here:  
[🔗 Open App](https://ml-project--healthcare-premium-prediction.streamlit.app/)  

*The app allows users to input customer details and predicts annual insurance premiums using the trained models.*

---

## Dataset
- Source: Synthetic/internal insurance dataset  
- Records: 50,000+ customers  
- Features include:
  - `age`, `gender`, `region`, `marital_status`, `number_of_dependants`
  - `bmi_category`, `smoking_status`, `employment_status`
  - `income_level`, `income_lakhs`, `medical_history`, `insurance_plan`
  - Target: `annual_premium_amount`

---

## Key Steps & Methodology

### 1. Data Cleaning & Preprocessing
- Handled **missing values** and removed duplicates  
- Corrected invalid entries like **negative dependents** and **unrealistic ages**  
- Applied **outlier treatment** using business-driven quantiles instead of strict IQR

### 2. Exploratory Data Analysis (EDA)
- Visualized **numerical features** (age, income, dependents) vs premium  
- Analyzed **categorical features** distributions and their effect on premiums  
- Created a **custom risk score** from medical history for better model insights

### 3. Feature Engineering
- **One-hot encoding** for nominal categorical variables  
- **Ordinal encoding** for income levels and insurance plans  
- **Min-Max scaling** for numerical features  
- Checked **multicollinearity using VIF** and removed highly correlated features  

### 4. Model Development
- Models trained: **Linear Regression, Ridge Regression, XGBoost**  
- Initial XGBoost model achieved **R² ≈ 0.97**  
- Conducted **residual and extreme error analysis** to evaluate over/undercharging risks

### 5. Segmented Modeling
- Identified **young customers (<25 years)** as high-error segment  
- Built **separate models for young vs rest of customers**  
- Reduced extreme errors for young segment from **30% → 2%**  
- Achieved **R² ≈ 0.99** for young and high accuracy for the rest

### 6. Model Deployment
- Exported models using **Joblib** (`model_young.joblib` and `model_rest.joblib`)  
- Exported **scalers with feature info** for production usage  
- Integrated models into an **application** for premium prediction

---

## Results
- **High predictive accuracy** with R² > 0.98 for both segments  
- Significant **reduction in extreme errors**, ensuring fair premium estimation  
- Insights show young customers benefit from **segmented modeling**  

---

## Technologies & Tools
- **Python**: pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, statsmodels  
- **Model Deployment**: Joblib  
- **Version Control**: Git & GitHub  

---

## Key Learnings
- Importance of **feature engineering** and **risk-aware analysis** in insurance modeling  
- How **segmented modeling** can reduce prediction errors for specific demographics  
- Combining **ML modeling with business understanding** ensures actionable and reliable results
