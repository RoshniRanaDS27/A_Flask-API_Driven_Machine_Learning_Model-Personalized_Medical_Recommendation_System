# Personalized Medical Recommendation System
![Recording 2024-08-28 145948 (3)](https://github.com/user-attachments/assets/7896a626-f78c-4c7f-ac57-48fe12cf5cd6)
#
# Dashboard Demo (click on below Dashboard video screen)
https://github.com/user-attachments/assets/104ad424-d87a-44bd-9279-11644025efd1

## Personalized Medical Recommendation System
Welcome to our Flask API-based Machine Learning model, designed as a comprehensive Recommendation and Prediction System. Today, I'm excited to present this one-screen dashboard that exemplifies our solution.  

Our system employs a Linear Regression model at its core, focusing on creating an intelligent, user-friendly interface for medical diagnosis and recommendations. Users can input their symptoms into the dashboard, and the system will provide accurate predictions along with relevant recommendations.

## Key Features: 
- ### Comprehensive Recommendations:
  Beyond identifying potential diseases, the system offers a detailed disease description, a list of precautions, suggested workout plans, a tailored medication list, and a diet plan. This holistic approach ensures users not only understand their diagnosis but also receive actionable steps to enhance their health.

- ### Synonym Dictionary:
  A unique feature of our system is the synonym dictionary that maps user-friendly words to medical terms. This allows users to describe their symptoms in their own words while still receiving precise predictions.

- ### Auto-Spelling Correction:
  Integrated spelling correction mechanisms automatically fix any mistakes made during symptom entry, further improving the user experience.  
  ![image](https://github.com/user-attachments/assets/d82f81d9-9762-4630-8742-e5c8bd131db4)


## Overview
The **Personalized Medical Recommendation System** leverages advanced machine learning techniques to provide tailored healthcare guidance. By integrating user-provided symptoms and health data, this system predicts potential diseases, recommends personalized medications, and suggests suitable workout routines. Implemented as a Flask API, it ensures accessibility and scalability across various devices.

- Link to Presentation: 
- ML Model Notebook: 
- Flask-APP-(windowsapi)Dashboard:
    
  ![image](https://github.com/user-attachments/assets/64f71d88-f397-4b84-b03d-41be4a7edb8e)


# Technology Used

### Data Science Libraries:
- scikit-learn: RandomForestClassifier, GradientBoostingClassifier, KNeighborsClassifier, SVC, MultinomialNB, StandardScaler, LabelEncoder, RFE
- Data Handling: pandas, numpy
- Visualization: matplotlib.pyplot, seaborn
- Text Processing: textblob
### Web Development:
- Backend: flask
### Utilities: pickle, warnings, os

# ML Model Selection and Accuracy 
![image](https://github.com/user-attachments/assets/47b52ebe-fd32-46f3-a733-5d94c6f2a868)

### Confusion Matrix 
![image](https://github.com/user-attachments/assets/d176fe4d-3e7d-4f78-96a2-e1f0ab452c30)
![image](https://github.com/user-attachments/assets/de7f27ac-3651-4251-b687-964594279df0)
![image](https://github.com/user-attachments/assets/56c84213-60cf-4e70-a752-19a86ce86eec)


# Runtime Before & after Data cleaning 
![image](https://github.com/user-attachments/assets/b872416e-3d89-4b55-95f6-9f7e825c4579)

# Data Description 

### Symptoms Presence in the Data 
![image](https://github.com/user-attachments/assets/1545bc70-207a-465e-b4db-ebbd76b543db)
![image](https://github.com/user-attachments/assets/e8bb0c77-124f-4aa2-8f2c-f1d9c0cb3964)


### # Most Frequent Diseases
![image](https://github.com/user-attachments/assets/91b49363-3ac1-4d2e-9e76-4ff4e448c80b)

# Most Frequent Symptom_1, Symptom_2, Symptom_3, Symptom_4
![image](https://github.com/user-attachments/assets/e43bc5d6-95c2-411a-be24-c93ed15d1248)
![image](https://github.com/user-attachments/assets/b5aff0c4-22b8-402b-8f26-d49f6ae49112)
![image](https://github.com/user-attachments/assets/8ac70efe-1ba9-4b65-9911-e95c4818af94)
![image](https://github.com/user-attachments/assets/e33d5670-e862-49cb-9f59-98a8747c1cc3)

# Precaution Counts 
![image](https://github.com/user-attachments/assets/984e15ae-9d70-44b6-aa51-683d08d11c51)
![image](https://github.com/user-attachments/assets/52064897-f026-49c2-9114-c8759e1ff725)

## ML The Model
This project utilizes diagnoses data to:

![image](https://github.com/user-attachments/assets/58dc29ed-68d9-48d8-b3a1-8b2db40be144)

- Liner regression Model
- Create clear links between symptoms and diseases.
- Provide recommendations on diet and exercise according to the predicted disease.

# Stats 
![image](https://github.com/user-attachments/assets/65ab9e1a-ca97-425e-8659-30d00f7d8f10)

![image](https://github.com/user-attachments/assets/52b34841-764c-48de-9252-b627cdb12e93)



# Comparison of RFE Rank and RF Rank for Each Feature

![image](https://github.com/user-attachments/assets/31a23b5e-12e0-43b1-bedd-f5c88d365c85)
![image](https://github.com/user-attachments/assets/fa99fa62-ab15-44c5-8c28-a0b8e2d85710)



## Data Processing
- Data Loading: Loaded into Python using Pandas.
- Data Cleaning: Ensured optimal performance with Matplotlib and Seaborn.
- Model Training: Implemented using a Single Vector Machine model in a Flask-hosted app with input validation and synonym dictionaries.

## Data Sources
- Kaggle Diagnoses Data: Dataset
- CBC Healthcare System: Article

## Evaluation Standards
- Data Model Implementation

1. Initialization, training, and evaluation of the model 
2. Data cleaning, normalization, and standardization 
3. Model uses SQL or Spark data 
4. Predictive power: ≥75% classification accuracy or 0.80 R-squared 

- Data Model Optimization 

1. Documented optimization and evaluation process 
2. Model performance display
![image](https://github.com/user-attachments/assets/8d1dcaee-7c0d-4a24-857e-e70df4613298)
![image](https://github.com/user-attachments/assets/d5e54f23-eced-4a72-b95a-be6868726451)



- GitHub Documentation 

1. Repository cleanliness and .gitignore 
2. Polished README presentation

- Presentation 

1. Group member participation 
2. Smooth content transitions and relevance 
3. Audience engagement

## Challenges and Solutions
- Data Cleaning

1. Optimization: Replaced inefficient loops with vectorized operations.
2. Logging: Added logging for cleaning steps like missing values and outliers.
3. Formatting: Standardized formats across datasets.

- Model Building

1. Optimization: Used Grid Search and Random Search for hyperparameter tuning.
2. Logging: Detailed logs for model training and validation.
3. Formatting: Consistent feature scaling and encoding.

- Dictionaries and Synonyms

1. Optimization: Used hash maps for faster synonym mapping.
2. Logging: Version control for dictionary updates.
3. Formatting: Standardized synonym formats.

- Auto-Correcting Functions

1. Optimization: Improved algorithm processing speed.
2. Logging: Tracked applied corrections for error analysis.
3. Formatting: Ensured alignment with standardized formats.

- Schema Diagrams and Relationships

1. Optimization: Normalized data for efficient queries.
2. Logging: Version control for schema changes.
3. Formatting: Regularized data types and clarified relationships.

# Conclusion
AI/ML presents a significant opportunity in the healthcare industry. Despite some risks of inaccuracies, the model demonstrates high initial accuracy rates and potential for use as a diagnostic filtering mechanism.
