**Predicting Illicit Drug Use Using Supervised Learning**
=========================================================

**1\. Problem Statement**
-------------------------

### **Objective**

The goal of this project is to leverage supervised machine learning to predict whether an individual is a user of illicit drugs based on demographic attributes, personality traits, and behavioral characteristics. This is a **binary classification task**, where the target variable indicates drug use:

-   `1` = User

-   `0` = Non-user

### **Why It Matters**

Drug abuse is a significant public health issue with far-reaching **social, psychological, and economic consequences**. By identifying strong predictors of drug abuse and evaluating model performance, this project aims to:

1.  Provide actionable insights for **targeted interventions**.

2.  Inform **educational programs and public health policies**.

3.  Develop a model to accurately predict if an individual is at risk for drug abuse, using key metrics such as **precision, recall, ROC-AUC, and F1-score**.

**2\. Models Used**
-------------------

The following machine learning models will be evaluated for their ability to predict drug abuse:

1.  **Logistic Regression** - Simple and interpretable for identifying significant predictors.

2.  **Random Forest** - Handles non-linear relationships and provides feature importance analysis.

3.  **Support Vector Machine (SVM)** - Suitable for high-dimensional data and non-linear boundaries.

4.  **k-Nearest Neighbors (KNN)** - Captures local patterns and is easy to implement.

**3\. Dataset Information**
---------------------------

### **Data Source**

The data used for this project is the **"Drug Consumption (Quantified)"** dataset:

-   **Citation**: Fehrman, E., Egan, V., & Mirkes, E. (2015). Drug Consumption (Quantified) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TC7S.

### **Dataset Overview**

-   **Number of Samples**: 1,885

-   **Number of Features**: 31 (12 numerical, 19 categorical)

### **Target Variable (Will be created later on)**: `drug_use` (Binary: `1`=User, `0`=Non-user)

### **Feature Categories**

1.  **Demographics**: Age, Gender, Education, Country, Ethnicity

2.  **Personality Traits**: Big Five Personality Scores (nscore, escore, oscore, ascore, cscore)

3.  **Behavioral Characteristics**: Impulsivity Score, Sensation-Seeking Score

4.  **Substance Use Variables**: Frequency of use for 19 different substances

**4\. Data Cleaning**
---------------------

### **4.1 Steps Taken**

1.  **Checked for missing values** (None found).

2.  **Fixed a typo** (`impuslive` → `impulsive`).

3.   **Checked for significant outliers** (None found)

4. **Dropped bad data** (Removed `semer` column and removed rows that reported use; semer is a fictitious drug meant to identify over-reporting).

5. **Converted drug use into a binary target variable (`drug_use`).** 

6. **Mapped categorical features** to improve readability (e.g., age groups, education levels, ethnicity).

### **4.2 Data Cleaning Summary**

-   **No missing values** found.

-   **No outliers removed**.

-   **Slight class imbalance** (Users = 58.7%, Non-users = 41.3%).

-   **3 rows dropped** because of semer use

**5\. Exploratory Data Analysis (EDA)**
--------------------------------------

Exploratory Data Analysis (EDA) helps us understand patterns in the dataset before applying machine learning models. 
In this section, we will:
- Analyze demographic and personality feature distributions.
- Check correlations between features.
- Identify potential biases and trends in drug use.

### **5.1 Feature Distributions**

   **a. Demographics vs. Drug Use**: Age, gender, education, ethnicity, and country distributions.

   **b. Personality Traits**: Box plots for Big Five traits.

   **c. Correlation Analysis & VIF Scores**: Heatmap of numerical feature correlations and VIF scores to measure collinearity.

   **d. Pairwise Relationships**: Scatter plots of selected features.

### **5.2 Findings**

-   **Younger individuals (18-24) have a higher proportion of drug use**.

-   **Males show higher drug use rates than females**.

-   **Lower education levels correlate with higher drug use prevalence**.

-   **Certain personality traits (e.g., high sensation-seeking score) correlate with higher drug use**.

**6\. Model Training and Evaluation**
-------------------------------------

### **6.1 Model Performance Metrics**

Each model was evaluated based on:
**Accuracy, Precision, Recall, F1-score, and ROC-AUC**—because they provide a **comprehensive evaluation**. Here's why each metric is relevant to my analysis:

---

### **1. Accuracy**  
**Why?**  
- Accuracy is the most intuitive metric, measuring the percentage of correctly predicted instances.  
- It is useful when **classes are balanced**, but can be misleading if there’s an imbalance, such as in this case.

---

### **2. Precision**  
**Why?**  
- Precision measures how many of the positive predictions were actually correct.  
- It is critical in cases where **false positives are costly** (e.g., medical diagnoses, fraud detection).


---

### **3. Recall**  
**Why?**  
- Recall measures how many actual positive cases were correctly identified.  
- It is important when **false negatives are costly**.


---

### **4. F1-Score**  
**Why?**  
- F1-score balances Precision and Recall, making it a **good compromise** when you need both metrics to be strong.  
- It is useful when **there’s class imbalance**—ensuring neither false positives nor false negatives dominate performance evaluation.

---

### **5. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**  
**Why?**  
- ROC-AUC evaluates how well the model differentiates between positive and negative cases **at different threshold levels**.  
- It is useful when **comparing models** because it looks beyond a fixed threshold.

---

### **Summary: Why These Metrics?**  
- **Accuracy** → General performance but unreliable for class imbalance.  
- **Precision** → Avoid false positives (flagging the wrong person).  
- **Recall** → Avoid false negatives (missing a real case).  
- **F1-Score** → Best for imbalanced data when both Precision & Recall matter.  
- **ROC-AUC** → Best for ranking predictions and evaluating probability-based classification.  

___

### As a reminder we will be looking at the following models:

a.  **Logistic Regression**

b.  **Random Forest**

c.  **Support Vector Machine (SVM)** 

d.  **k-Nearest Neighbors (KNN)**

### Collinearity

- As previously established, there is no severe collinearity within the dataset, so no further action is required to accomodate this.

# Results and Analysis

This study evaluated four supervised learning models to predict illicit drug use: **Logistic Regression, Random Forest, k-Nearest Neighbors (KNN), and Support Vector Machine (SVM)**. The models were assessed using multiple evaluation metrics: **accuracy, precision, recall, F1-score, and ROC-AUC**. Given the imbalanced nature of the dataset, **F1-score and ROC-AUC** were prioritized over accuracy.

## 1. Logistic Regression (Accuracy: 82.23%)

- A linear model that is well-suited for **interpretability**.
- Achieved a **high precision (87.75%)**, meaning fewer false positives.
- **Recall (81.0%)** was the lowest among all models, indicating more false negatives.
- **F1-score (84.24%)** shows a reasonable trade-off between precision and recall.
- **ROC-AUC (0.92)** suggests that Logistic Regression ranks predictions well but struggles with recall.

## 2. Random Forest (Best Overall Accuracy: 84.35%)

- Performs well with **non-linear relationships** and is robust to noise.
- **Precision (88.21%)** is high, indicating strong positive predictions.
- **Best recall (84.62%)** among all models, meaning it identifies more actual drug users correctly.
- **F1-score (86.37%)** is the highest, making this model the most balanced in terms of predictive performance.
- **ROC-AUC (0.92)** confirms strong ranking ability.

## 3. k-Nearest Neighbors (KNN) (Accuracy: 83.29%)

- Captures **complex decision boundaries** well but is sensitive to noise.
- **Precision (87.98%)** and **recall (82.81%)** show a good balance.
- **F1-score (85.31%)** is slightly lower than Random Forest but better than Logistic Regression.
- **ROC-AUC (0.92)** indicates a strong ability to rank positive cases correctly.

## 4. Support Vector Machine (SVM) (Accuracy: 82.23%)

- Provides a strong balance between **precision (85.98%)** and **recall (83.26%)**.
- **F1-score (84.60%)** is slightly better than Logistic Regression but lower than Random Forest.
- **ROC-AUC (0.92)** matches all other models, showing no advantage in probability ranking.

## 5. Model Comparison and Insights

- **Random Forest performed best overall** in accuracy (84.35%), recall (84.62%), and F1-score (86.37%), making it the most effective at identifying drug users.
- **Logistic Regression and SVM had similar accuracy (82.23%) but lower recall**, meaning they miss more actual drug users.
- **KNN provides a good balance but is computationally expensive**.
- **All models achieved the same ROC-AUC (0.92)**, meaning they rank predictions equally well.

## 6. Metric Selection Justification

- **Accuracy alone is misleading** due to the dataset’s imbalance.
- **Recall is crucial** since missing actual drug users (false negatives) may be more concerning than false positives.
- **F1-score was prioritized** as it balances precision and recall.
- **ROC-AUC helped compare ranking performance**, but it did not differentiate models significantly in this case.

## 7. Best Model

- **Random Forest is the best-performing model**, offering the highest F1-score and recall while maintaining high precision.

# Discussion and Conclusion

## Key Learnings
- The models were not significantly different from each other, with all having **similar performance metrics**. 
- More time was spent on **finding and cleaning a good dataset** than on actual modeling, reinforcing the importance of **high-quality data**.

## Model Limitations
- No model stood out as a clear leader, which suggests that the dataset’s inherent patterns might **not be complex enough** for major performance differences.
- Computational power was a limiting factor. **Hyperparameter tuning** using GridSearchCV placed a heavy load on my computer, restricting deeper experimentation. Perhaps running these experiments in environments with more compute can lead to interesting outcomes.

## Ethical Considerations
- A significant issue was the dataset's **lack of diversity**. It was predominantly composed of white individuals.
- Traditional **outlier detection methods flagged data from non-white individuals as outliers**, which is a serious bias concern.
- This highlights the **need for diverse datasets**, especially in social sciences, where racial and demographic representation is crucial.

## Future Improvements
- A **more powerful machine** would enable deeper **hyperparameter tuning** and experimentation with more advanced techniques.
- Future work should focus on **identifying and using more representative datasets** that better match the demographics of the **U.S. population**.
- Exploring ensemble models or deep learning approaches might improve predictive performance.

## Final Thoughts
- While the findings may not have immediate real-world intervention applications, they emphasize the **importance of dataset quality and diversity** in machine learning.
- The experience highlighted the computational challenges of tuning models and the necessity of **ethical AI considerations** when working with biased datasets.
