# 🍷 Wine Review Score Prediction

## 📌 Project Overview

This project aims to **predict wine review scores** (points) based on various features such as **country, variety, price, description**, and more. The goal is to develop a **machine learning model** that accurately estimates wine ratings and provides **explainability insights** using **SHAP (SHapley Additive Explanations).**

## 🚀 Key Features

- **Data Preprocessing**: Cleaning, handling missing values, feature engineering.
- **Machine Learning Models**:
    - **XGBoost Regressor** (Final model)
    - **LightGBM & Gradient Boosting** (Experiments)
- **Explainable AI (XAI)**:
    - **SHAP for global & local feature importance**
    - **LIME for individual prediction explanations**
- **Flask Web App**:
    - Allows users to input wine details and receive a predicted score.
    - Displays SHAP visualizations for interpretability.

---

## 📂 Dataset

The dataset comes from a collection of **130,000+ wine reviews** containing the following features:

|Feature|Description|
|---|---|
|`country`|Country of origin|
|`description`|Text review of the wine|
|`designation`|Specific vineyard or label name|
|`price`|Wine price in USD|
|`province`|Region within the country|
|`region_1`|Sub-region of the wine|
|`taster_name`|Name of the wine taster|
|`variety`|Type of grape used|
|`winery`|Name of the winery|
|`points`|Wine rating (Target variable)|

---

## 📊 Data Preprocessing

- **Dropped irrelevant or highly missing columns** (e.g., `taster_twitter_handle`).
- **Handled missing values**:
    - Categorical features filled with `'Unknown'`.
    - `price` imputed with the **median value**.
- **Feature Engineering**:
    - `boxcox_price` - used BoxCox transformation to normalize `price` and hadle outliers.
    - **TF-IDF vectorization** for text descriptions.
    - **Count Encoding** for categorical variables.

---

## 🔥 Model Training & Evaluation

### **🏆 Final Model: XGBoost Regressor**

#### **Pipeline**

```python
pipeline_points = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ('cat', ce.CountEncoder(), categorical_features),
            ('text_1', TfidfVectorizer(max_features=5000), text_features)
        ]
    )),
    ('regressor', xgb.XGBRegressor(objective='tweedie', n_estimators=100, random_state=42))
])
```

#### **Model Performance**

- **Baseline (Median Prediction)**: **88 points**
- **XGBoost RMSE**: **Better than baseline (lower RMSE)**
- **SHAP Analysis**: Key predictors include **price, variety, province, and key descriptive words**.

---

## 🤖 Explainable AI (XAI)

### **SHAP Analysis Results**

- **Top Positive Words:** "intensely", "classic", "elegantly" → Increase wine score
- **Top Negative Words:** "disappointing", "awkward", "underdeveloped" → Decrease score
- **`price` and `province` are significant predictors.**

#### **SHAP Summary Plot Example:**

```python
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)
```

### **LIME Local Interpretability**

- Used to **explain individual predictions**.
- Shows how each feature **affects a single wine score prediction**.

---

## 💻 Flask Web Application

A web app was developed using **Flask**, where users can:

- Input wine details and get a **predicted rating**.
- View SHAP explanations for their prediction.

#### **Run the App**

```bash
python app.py
```

Then visit http://127.0.0.1:5000/ in your browser. Also hosted at https://ml-week.ew.r.appspot.com/

---

## 📌 Business Insights

### **Key Takeaways from SHAP Analysis**

1. **Marketing Optimization**:
    - Use positive words like **"intensely", "classic"** in wine descriptions to boost perception.
    - Avoid negative terms like **"awkward", "disappointing"**.
2. **Dynamic Pricing Strategy**:
    - High prices do **not always** mean high ratings.
    - Optimize pricing based on region and wine variety.
3. **Improve Wine Quality**:
    - Identify negative flavor descriptors and adjust production methods.
4. **Wine Recommender System**:
    - Build a **personalized recommendation model** based on user preferences and reviews.

---

## 📜 Future Work

- Improve text processing with **BERT or Word2Vec embeddings**.
- Experiment with **Neural Networks for better predictions**.
- Deploy model as a **REST API for integration with e-commerce platforms**.

---

## 🛠 Tech Stack

- **Python** (Pandas, NumPy, Scikit-Learn, XGBoost, SHAP, LIME)
- **Flask** (Web app for model predictions & SHAP visualization)
- **Matplotlib & Seaborn** (Data visualization)
- **Joblib** (Model persistence)

---

## 🎯 How to Run the Project

1. **Install dependencies**:
    
    ```bash
    pip install -r requirements.txt
    ```
    
2. **Train the model**:
    
    ```python
    python train.py
    ```
    
3. **Run Flask app**:
    
    ```bash
    python app.py
    ```
    
4. **Test Predictions**:
    
    - Use `predict.py` to test individual wine descriptions.
    
    ```python
    python predict.py --description "A bold red wine with hints of oak and spice"
    ```
    

---

## 📄 Authors

- **[Сазоненко Євгеній]** - MLE & Team Lead
- **[Казаков Дмитро]** - SE
- **[Кордан Павло]** - DE/DA

---

## 📌 Acknowledgments

- Data sourced from **[Wine Enthusiast Dataset](https://www.kaggle.com/datasets/zynicide/wine-reviews/data)**.

---

## 📜 License

This project is licensed under the **MIT License**.
