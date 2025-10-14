# Lead Scoring Classification Project

This project aims to build a binary classification model to predict whether a lead will convert (sign up for a platform) or not. The analysis and modeling are performed on the Bank Marketing dataset.

## Objective

The primary goal is to identify the key factors that influence a lead's decision to convert and to build a predictive model that can accurately classify new leads. This involves data cleaning, exploratory data analysis, feature engineering, and model training.

## Dataset

The dataset used is the "Course Lead Scoring" dataset. It can be downloaded from [this link](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv).

The target variable for our classification task is `converted`.
*   `1`: The client signed up.
*   `0`: The client did not sign up.

## Project Steps

The project follows a structured approach to data analysis and model building:

### 1. Data Preparation and Cleaning

The first step was to load the data and handle inconsistencies.
*   **Missing Values:** We identified columns with missing values (`NaN`).
*   **Imputation Strategy:**
    *   For categorical features (`lead_source`, `industry`, etc.), missing values were replaced with the string `'NA'` to treat them as a distinct category.
    *   For numerical features (`annual_income`), missing values were filled with `0.0`. This is a simple approach, and more complex methods (like mean/median imputation) could be considered in a future iteration.

### 2. Exploratory Data Analysis (EDA)

Before modeling, we explored the data to uncover initial insights:

*   **Most Frequent Industry (Question 1):** After data preparation, the most frequent value (mode) in the `industry` column was **`NA`**. This indicates that a significant number of leads do not have an associated industry in the dataset.

*   **Feature Correlation (Question 2):** We calculated the correlation matrix for all numerical features. The analysis revealed a very strong positive correlation (**~0.91**) between `number_of_courses_viewed` and `lead_score`. This suggests these two features are highly codependent (multicollinearity) and likely provide redundant information.

*   **Mutual Information (Question 3):** To understand the relationship between categorical features and the target variable `converted`, we used mutual information. The **`lead_source`** column had the highest score, indicating it is the most informative categorical feature for predicting whether a lead will convert.

### 3. Model Training and Evaluation

#### a. Data Splitting
The prepared dataset was split into three parts for robust model evaluation:
*   Training set (60%)
*   Validation set (20%)
*   Test set (20%)
A `random_state` of 42 was used to ensure the split is reproducible.

#### b. One-Hot Encoding
Categorical features were converted into a numerical format using `DictVectorizer` from Scikit-Learn. This process, known as one-hot encoding, creates binary columns for each category, making the data suitable for a logistic regression model.

#### c. Baseline Model Accuracy (Question 4)
A logistic regression model was trained on the prepared training data.
*   **Model:** `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)`
*   **Result:** The model achieved an accuracy of **0.84** on the validation set.

### 4. Feature and Model Optimization

#### a. Feature Elimination (Question 5)
We assessed the importance of individual features by training the model repeatedly, each time excluding one feature. We measured the change in accuracy compared to the baseline model.
*   **Finding:** Removing the **`employment_status`** feature resulted in the smallest change (a slight increase) in model accuracy. This suggests it is the least useful feature among the options tested for this specific model.

#### b. Regularization Tuning (Question 6)
To prevent overfitting and find the optimal model complexity, we tuned the regularization parameter `C`. We tested several values: `[0.01, 0.1, 1, 10, 100]`.
*   **Finding:** The best validation accuracy was achieved with **`C=1`**. Values higher than 1 did not improve the score, and values lower than 1 resulted in slightly worse performance.

## Key Takeaways (Takeaways)

1.  **Data Quality is Crucial:** A significant portion of the data had missing `industry` information, which became the most common category after cleaning. This highlights the importance of understanding and properly handling missing data.
2.  **Redundant Features Exist:** The extremely high correlation between `lead_score` and `number_of_courses_viewed` suggests that one of these could potentially be removed in future modeling steps to simplify the model without a significant loss of information.
3.  **Source Matters:** `lead_source` was the most predictive categorical feature. This implies that marketing efforts should focus on channels that have historically generated high-converting leads.
4.  **Simplicity is Effective:** A simple logistic regression model provided a strong baseline accuracy of **84%**, demonstrating that complex models are not always necessary to achieve good results.
5.  **Optimization Provides Minor Gains:** While feature elimination and regularization tuning are important steps, in this case, they provided only marginal improvements over the baseline model, which was already quite effective.

## How to Reproduce

1.  Create a Python virtual environment.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn jupyterlab
    ```
3.  Download the dataset:
    ```bash
    wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv
    ```
4.  Launch Jupyter Lab:
    ```bash
    jupyter lab
    ```
5.  Open the notebook (`.ipynb` file) and run the cells sequentially.
