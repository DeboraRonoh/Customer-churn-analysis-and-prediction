📊 Customer Churn Analysis & Prediction
📌Overview
Customer churn is an important business problem for telecommunications companies because losing existing customers can affect revenue and increase the need for customer acquisition.
This project analyzes customer data to understand the factors associated with churn and develops a machine learning classification model to predict whether a customer is likely to churn.
The project covers the complete workflow from data cleaning and exploratory analysis to feature selection, class-imbalance handling, model comparison, hyperparameter tuning, model evaluation, and model persistence.
The final model is a tuned XGBoost classification model with feature selection and SMOTE, selected primarily for its ability to identify customers who churn.

 Business Problem
Customer churn is an important challenge for telecommunications companies, as losing customers can negatively impact business performance and customer retention.
This project focuses on analyzing customer churn within a telecommunications company and developing predictive models to identify customers who are at risk of churning.
The analysis aims to understand the factors and customer characteristics associated with churn, while the predictive modeling component aims to identify customers who are more likely to leave.
The ultimate goal is to use the findings from the analysis and predictive models to provide actionable insights and recommendations that can help reduce customer churn and improve customer retention.

Objectives
The project objectives were to:
•	Understand the structure and characteristics of the customer dataset.
•	Clean and prepare the data for analysis.
•	Calculate the overall customer churn rate.
•	Explore patterns and relationships between customer characteristics and churn.
•	Segment customers based on tenure and monthly charges.
•	Analyze churn across different customer segments.
•	Identify important features for churn prediction.
•	Build classification models for customer churn.
•	Address class imbalance using SMOTE.
•	Compare model performance before and after feature selection.
•	Tune the best-performing model using GridSearchCV.
•	Evaluate the final model using classification metrics.
•	Save the final trained model for future use.
•	Translate the findings into business recommendations.
📊 Dataset
The dataset contains 7,043 customer records and 21 original columns.
The variables cover customer demographics, account information, services, payment methods, and financial information.
Customer Information
•	customerID
•	gender
•	SeniorCitizen
•	Partner
•	Dependents
Account Information
•	tenure
•	Contract
•	PaperlessBilling
•	PaymentMethod
Services
•	PhoneService
•	MultipleLines
•	InternetService
•	OnlineSecurity
•	OnlineBackup
•	DeviceProtection
•	TechSupport
•	StreamingTV
•	StreamingMovies
Financial Information
•	MonthlyCharges
•	TotalCharges
Target Variable
•	Churn
The target contains two classes:
No  → Customer did not churn
Yes → Customer churned
Therefore, the problem was treated as a binary classification problem.

🔄 Project Workflow
Raw Dataset
     ↓
Data Inspection
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Customer Segmentation
     ↓
Feature Preparation
     ↓
Train/Test Split
     ↓
Preprocessing Pipeline
     ↓
Random Forest
     ↓
Feature Importance & Feature Selection
     ↓
XGBoost + SMOTE
     ↓
XGBoost + Feature Selection + SMOTE
     ↓
Hyperparameter Tuning
     ↓
Final Model Evaluation
     ↓
Model Persistence
 1. Data Cleaning & Preparation
The first stage involved inspecting the dataset structure, data types, missing values, duplicate records, and descriptive statistics.
TotalCharges Data Type
The TotalCharges column was initially stored as an object rather than a numerical variable.
It was converted to numeric using:
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
This revealed 11 missing values in TotalCharges.
Handling Missing TotalCharges
The missing values were calculated using:
TotalCharges = MonthlyCharges × tenure
This allowed the missing financial values to be filled while retaining the affected customer records.
Duplicate Records
The dataset was checked for duplicate rows.
The analysis found:
0 duplicate records
Descriptive Statistics
Summary statistics were generated for the numerical variables, including:
•	SeniorCitizen
•	tenure
•	MonthlyCharges
•	TotalCharges
The average customer tenure was approximately 32.37 months, while the average monthly charge was approximately 64.76.

📈 2. Exploratory Data Analysis
Exploratory Data Analysis was conducted to understand the customer base and investigate patterns associated with churn.
The analysis examined:
•	Overall churn rate
•	Partner status
•	Dependents
•	Gender
•	Customer tenure
•	Contract type
•	Payment method
•	Internet service
•	Customer tenure and churn
•	Customer segments and churn

📌 Overall Churn Rate
The overall customer churn rate calculated from the dataset was:
26.54%
This indicates that approximately one in four customers in the dataset had churned.

👥 Partner Status
The distribution of customers by partner status was analyzed to understand the composition of the customer base and investigate its relationship with churn.

👨👩👧 Dependents
Customer dependents were analyzed as another demographic characteristic to investigate whether customers with or without dependents showed different churn patterns.

⚧ Gender
Gender distribution was examined as part of the exploratory analysis.

 Customer Tenure
Customer tenure was analyzed to understand how long customers had remained with the company.
The analysis examined the distribution of tenure and compared customer tenure with churn status.

📄 Contract Type
Churn was analyzed across the different contract categories:
•	Month-to-month
•	One year
•	Two year
Contract type was an important part of the churn analysis because it provides insight into the relationship between customer commitment and retention.

💳 Payment Method
The analysis also investigated churn across different payment methods.
The payment methods included:
•	Electronic check
•	Mailed check
•	Bank transfer (automatic)
•	Credit card (automatic)

🌐 Internet Service
Churn was compared across internet service types, including:
•	DSL
•	Fiber optic
•	No internet service

👥 3. Customer Segmentation
Additional customer segmentation was performed to explore churn patterns across groups with similar characteristics.
Two segmentation variables were created.

Tenure Groups
Customers were grouped according to their tenure:
Tenure Group
0–12 Months
13–24 Months
25–48 Months
49–72 Months
Churn was then analyzed across these tenure groups.

💰 Monthly Charge Groups
Customers were also grouped based on their monthly charges:
Monthly Charge Group
Low
Medium
High
Very High
Churn patterns were examined across these groups.

🔀 Tenure × Contract Analysis
The project also examined the interaction between:
Tenure Group × Contract Type
This provided a more detailed view of how customer tenure and contract commitment relate to churn.
The segmentation variables were used for analysis but were subsequently removed before machine learning because they were created for exploratory purposes.

⚙️ 4. Preparing Data for Machine Learning
Before training the models, the dataset was prepared for prediction.
Removed Variables
The following columns were removed:
customerID
TenureGroup
MonthlyChargeGroup
customerID is an identifier and does not provide meaningful predictive information.
TenureGroup and MonthlyChargeGroup were created during exploratory analysis and were removed before modeling.
Features and Target
The dataset was separated into:
X = df.drop(columns=["Churn"])
y = df["Churn"]
The target variable was converted to binary format:
No  → 0
Yes → 1

 5. Train-Test Split
The dataset was divided into:
•	80% training data
•	20% testing data
using:
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
The test set contained 1,409 customers, including:
•	1,036 non-churn customers
•	373 churn customers
This imbalance was taken into consideration during model development.

 6. Data Preprocessing
A ColumnTransformer was used to process numerical and categorical variables separately.
Numerical Variables
The numerical variables were:
•	SeniorCitizen
•	tenure
•	MonthlyCharges
•	TotalCharges
These were standardized using:
StandardScaler
Categorical Variables
Categorical variables were encoded using:
OneHotEncoder
with:
drop="first"
handle_unknown="ignore"
The preprocessing was incorporated into machine learning pipelines to ensure consistent transformation during training and prediction.

 7. Random Forest Model
A Random Forest Classifier was initially developed.
The model used:
RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)
The class_weight="balanced" parameter was used to account for the imbalance between churn and non-churn classes.
The Random Forest achieved:
Metric	Score
Training Accuracy	99.84%
Test Accuracy	79.21%



The large difference between training and test accuracy was also an indication that the Random Forest was fitting the training data very strongly and required careful evaluation on unseen data.

🔎 8. Feature Importance & Feature Selection
Feature importance was examined using the Random Forest model to understand which variables contributed to churn prediction.
Feature importance was visualized and used as a basis for feature selection.
SelectFromModel was then used to select important features.
Feature selection allowed the project to compare model performance using:
All features
vs.
Selected features
This process was later incorporated into the XGBoost modeling pipeline.

9. XGBoost + SMOTE
An XGBoost classification model was developed as an alternative to Random Forest.
Because the dataset contained fewer churn customers than non-churn customers, SMOTE (Synthetic Minority Oversampling Technique) was introduced into the pipeline.
The workflow was:
Preprocessing
      ↓
SMOTE
      ↓
XGBoost
The XGBoost model achieved the following results before feature selection:
Class	Precision	Recall	F1-score
Non-Churn	0.85	0.86	0.86
Churn	0.60	0.59	0.60
Overall accuracy:
79%

 10. XGBoost + Feature Selection + SMOTE
A second XGBoost pipeline was developed by adding feature selection.
The workflow became:
Preprocessing
      ↓
Feature Selection
      ↓
SMOTE
      ↓
XGBoost
Feature selection was implemented using SelectFromModel with an XGBoost estimator and a median importance threshold.
The model achieved:
Class	Precision	Recall	F1-score
Non-Churn	0.87	0.82	0.85
Churn	0.57	0.66	0.61
Overall accuracy:
78%
Although the overall accuracy was slightly lower than the previous XGBoost model, recall for the churn class improved from 59% to 66%.
This made the model more useful for the project’s objective of identifying customers who may churn.

 11. Hyperparameter Tuning
The XGBoost model with feature selection and SMOTE was selected for further optimization.
GridSearchCV was used with:
•	5-fold cross-validation
•	F1-score as the optimization metric
•	n_jobs=-1
The search evaluated:
•	n_estimators
•	max_depth
•	learning_rate
•	subsample
•	colsample_bytree
•	min_child_weight
The search evaluated:
324 parameter combinations
1,620 total fits

 12. Best Hyperparameters
The best-performing parameter combination was:
{
    "classifier__colsample_bytree": 0.8,
    "classifier__learning_rate": 0.01,
    "classifier__max_depth": 7,
    "classifier__min_child_weight": 1,
    "classifier__n_estimators": 200,
    "classifier__subsample": 0.8
}
The best cross-validation F1-score was:
0.6383

📈 13. Final Model Performance
The tuned XGBoost model was evaluated on the unseen test dataset.
Overall Accuracy
76.65%
Classification Performance
Class	Precision	Recall	F1-score	Support
Non-Churn	0.90	0.76	0.83	1,036
Churn	0.54	0.78	0.64	373
Overall	—	—	—	1,409

🎯 Why the Tuned XGBoost Model Was Selected
The final model was selected based primarily on its ability to identify customers who actually churn.
Although its overall accuracy was lower than the Random Forest and some of the earlier XGBoost results, the tuned model achieved:
78% recall for the churn class
This means it correctly identified approximately 78% of the customers in the test set who actually churned.
For a churn-retention use case, correctly identifying customers who are at risk of leaving can be more valuable than maximizing overall accuracy alone.

14. Business Recommendations
The analysis can support several customer retention strategies.
1: Target Month-to-Month Customers
Evidence
Feature importance analysis showed that Contract Type is among the strongest predictors of churn.
EDA also showed that month-to-month customers churn more frequently than customers on one-year or two-year contracts.
Recommendation
Offer incentives to encourage customers to switch from month-to-month contracts to longer-term contracts.
Examples:
•	10% discount for the first six months 
•	Free streaming subscription 
•	Free installation of premium services 
•	Loyalty reward points 
 Impact
Long-term contracts generally increase customer retention and create more predictable recurring revenue.
2: Focus on New Customers
Evidence
Tenure was one of the top predictors.
Customers with short tenure are typically more likely to churn.
Recommendation
Create a customer onboarding program.
Examples:
•	Welcome emails 
•	Regular follow-up calls during the first three months 
•	Tutorials explaining available services 
•	Personalized customer support 
Impact
Improving the early customer experience can increase satisfaction and reduce early churn.
3: Review High Monthly Charges
Evidence
MonthlyCharges was one of the most important variables.
Customers paying higher monthly fees may perceive lower value.
Recommendation
Offer personalized discounts or bundled service packages.
Examples:
•	Family packages 
•	Internet + TV bundles 
•	Loyalty discounts 
Impact
Customers are more likely to remain when they perceive better value for money.
4: Target Fiber Optic Customers
The model identified Fiber Optic Internet Service as an important predictor.
 EDA confirms that fiber customers churn more often:
Recommendation
Conduct customer satisfaction surveys among fiber customers.
Possible improvements:
•	Improve network reliability 
•	Increase internet speed 
•	Provide proactive technical support
5: Improve Payment Experience
Electronic Check appears among the important features.
Analysis shows higher churn among customers using Electronic Check:
Recommendation
Encourage customers to switch to automatic payments.
Offer incentives such as:
•	Small monthly discount 
•	Bonus loyalty points 
•	Cashback rewards 
Automatic payments reduce missed payments and often improve retention.
6: Prioritize High-Value Customers
Using segmentation analysis:
High-value customers could be defined as:
•	Long tenure 
•	High Total Charges 
•	High Monthly Charges 
These customers generate significant revenue and should receive proactive retention efforts.
Examples:
•	VIP customer support 
•	Exclusive loyalty offers 
•	Personalized account management

15. Model Persistence
The final tuned model was saved using Joblib:
joblib.dump(
    best_model,
    "Churn_model.pkl"
)
This allows the trained model to be reused for future predictions without having to retrain it from scratch.

 Technologies Used
Programming Language
•	Python
Data Analysis
•	Pandas
•	NumPy
Data Visualization
•	Matplotlib
Machine Learning
•	Scikit-learn
•	XGBoost
•	Imbalanced-learn
Model Persistence
•	Joblib
Development Environment
•	Jupyter Notebook

📌 Conclusion
This project demonstrates an end-to-end approach to customer churn analysis and prediction.
The project progressed from data inspection and cleaning, through exploratory analysis and customer segmentation, to feature selection, class-imbalance handling, model comparison, hyperparameter tuning, and final model evaluation.
The final tuned XGBoost model achieved 76.65% overall accuracy and 78% recall for the churn class on the test dataset.
The project demonstrates the ability to combine technical machine learning skills with business-oriented analysis to identify potential customer retention opportunities.

👩💻 Author
Debora Chepkoech
 Data Analyst / Data Scientist
Skills: 
Python | SQL | Excel | Power BI | Data Analysis | Machine Learning
