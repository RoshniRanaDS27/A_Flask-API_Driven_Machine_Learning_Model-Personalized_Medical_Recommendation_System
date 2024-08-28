# Personalized Medical Recommendation System
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

## Overview
The **Personalized Medical Recommendation System** leverages advanced machine learning techniques to provide tailored healthcare guidance. By integrating user-provided symptoms and health data, this system predicts potential diseases, recommends personalized medications, and suggests suitable workout routines. Implemented as a Flask API, it ensures accessibility and scalability across various devices.

- Link to Presentation: 
- ML Model Notebook: 
- Flask-API Dashboard:

# Technology Used

### Data Science Libraries:
- scikit-learn: RandomForestClassifier, GradientBoostingClassifier, KNeighborsClassifier, SVC, MultinomialNB, StandardScaler, LabelEncoder, RFE
- Data Handling: pandas, numpy
- Visualization: matplotlib.pyplot, seaborn
- Text Processing: textblob
### Web Development:
- Backend: flask
### Utilities: pickle, warnings, os


## The Model
This project utilizes diagnoses data to:

- Create clear links between symptoms and diseases.
- Provide recommendations on diet and exercise according to the predicted disease.

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

- GitHub Documentation 

1. Repository cleanliness and .gitignore 
2. Polished README presentation

- Group Presentation 

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
