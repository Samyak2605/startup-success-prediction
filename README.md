# 🚀 Startup Success Prediction (Machine Learning Project)

## 📌 Project Overview
Most startups fail due to multiple factors such as poor funding, lack of market fit, or weak team composition.  
This project applies **Machine Learning** to predict whether a startup is likely to **succeed or fail**, based on key features such as location, funding, and category.  

The goal is to help **investors, founders, and VCs** make data-driven decisions.

---

## 📊 Dataset
- Source: `data/startups.csv`  
- Features include:
  - `State` – Location of the startup  
  - `Category` – Industry/domain  
  - `Funding Amount` – Total funding received  
  - `Employee Count` – Approximate team size  
  - `Success` – Target variable (1 = Success, 0 = Failure)  

---

## 🛠️ Tech Stack
- **Python**
- **Pandas & NumPy** – Data preprocessing & analysis  
- **Matplotlib & Seaborn** – Data visualization  
- **Scikit-learn** – ML models (Logistic Regression, Random Forest, etc.)  
- **XGBoost** – Advanced classification model  

---

## 📈 Methodology
1. **Data Cleaning**
   - Handled missing values  
   - Removed outliers (e.g., extremely high funding values)  
   - Encoded categorical variables (State, Category)  

2. **Exploratory Data Analysis (EDA)**
   - Distribution of funding by category and state  
   - Correlation heatmaps  
   - Success rate trends  

3. **Modeling**
   - Train-test split (80-20)  
   - Models tested: Logistic Regression, Random Forest, XGBoost  
   - Performance evaluated using **accuracy, precision, recall, F1-score**  

4. **Results**
   - XGBoost achieved the **highest accuracy**  
   - Important features: Funding Amount, Category, Employee Count  

---

