# Import required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import IsolationForest


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