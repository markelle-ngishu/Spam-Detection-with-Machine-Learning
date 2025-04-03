# Import required libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# Load the dataset from the CSV file
try:
    file_path = 'spambase.csv'
    dataset = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(dataset.head())

# Step 1: Prepare the data for modeling
# Select relevant features and target variable
selected_features = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
    'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
    'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
    'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
    'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
    'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
    'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
    'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_%3B', 'char_freq_%28',
    'char_freq_%5B', 'char_freq_%21', 'char_freq_%24', 'char_freq_%23', 'capital_run_length_average',
    'capital_run_length_longest', 'capital_run_length_total'
]
target_variable = 'class'

# Extract features and target variable
X = dataset[selected_features]
y = dataset[target_variable]
# DATA PREPROCESSING

# Step 2: Handle missing values
imputer = SimpleImputer(strategy='mean')  # Use mean imputation for missing values
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=selected_features)

# Step 3: Detect and handle outliers
outlier_detector = IsolationForest(contamination=0.05, random_state=42)
outliers = outlier_detector.fit_predict(X_imputed)
X_no_outliers = X_imputed[outliers != -1]
y_no_outliers = y[outliers != -1]

# Step 4: Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_no_outliers)
X_normalized = pd.DataFrame(X_normalized, columns=selected_features)

# Step 5: Feature selection using RandomForest importance
feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
X_selected = feature_selector.fit_transform(X_normalized, y_no_outliers)
selected_features = np.array(selected_features)[feature_selector.get_support()]
# MODEL TRAINING AND EVALUATION

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_no_outliers, test_size=0.2, random_state=42
)

# Step 7: Hyperparameter tuning using GridSearchCV for RandomForest
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Regularization: Limit tree depth
    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4],    # Regularization: Minimum samples per leaf
    'max_features': ['sqrt', 'log2']  # Feature selection during splits
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),  # Use RandomForestClassifier
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',  # Use accuracy for classification
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best model after hyperparameter tuning
best_rf_model = grid_search.best_estimator_

# Step 8: Hyperparameter tuning using RandomizedSearchCV for Gradient Boosting
gb_param_dist = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]  # Stochastic gradient boosting
}

gb_random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),  # Use GradientBoostingClassifier
    param_distributions=gb_param_dist,
    n_iter=10,  # Number of parameter settings to sample
    cv=5,
    scoring='accuracy',  # Use accuracy for classification
    random_state=42,
    n_jobs=-1
)

gb_random_search.fit(X_train, y_train)
best_gb_model = gb_random_search.best_estimator_

# Step 9: Implement Bagging to reduce variance
bagging_model = BaggingClassifier(
    n_estimators=50,  # Increase from 10 to 50
    random_state=42,
    n_jobs=-1
)

# Train the bagging model
bagging_model.fit(X_train, y_train)

# Step 10: Add Gradient Boosting with Early Stopping
# Use the best Gradient Boosting model from RandomizedSearchCV
best_gb_model.set_params(
    n_estimators=1000,  # Increase the number of estimators
    validation_fraction=0.2,  # Use 20% of training data for validation
    n_iter_no_change=10,  # Early stopping after 10 iterations without improvement
    tol=1e-4  # Tolerance for early stopping
)

# Train the Gradient Boosting model with early stopping
best_gb_model.fit(X_train, y_train)

# Step 11: Implement a Stacking Classifier for Improved Performance
stacked_model = StackingClassifier(
    estimators=[
        ('bagging', bagging_model),  # Bagging for variance reduction
        ('et', ExtraTreesClassifier(n_estimators=100, random_state=42)),  # ExtraTrees for randomness
        ('gb', best_gb_model),  # Gradient Boosting with early stopping
        ('logistic', LogisticRegression(penalty='l1', solver='liblinear')),  # Logistic regression with L1 regularization
        ('svc', SVC(kernel='rbf', C=1.0)),  # Add SVC for diversity
        ('knn', KNeighborsClassifier(n_neighbors=5))  # Add KNN for diversity
    ],
    final_estimator=LogisticRegression(),  # Logistic regression as the meta-model
    n_jobs=-1
)

# Train the optimized stacked model
stacked_model.fit(X_train, y_train)

# Step 12: Evaluate the model's performance
print("\nEvaluating Model Performance")

predictions = stacked_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Display results
print("\nOptimized Stacking Model with Bagging and Early Stopping Performance:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Best Hyperparameters (RandomForest): {grid_search.best_params_}")
print(f"Best Hyperparameters (Gradient Boosting): {gb_random_search.best_params_}")
print(f"Selected Features: {selected_features}")