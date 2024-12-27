# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree

# Load the dataset (update the file path if necessary)
df = pd.read_csv('Dataset/bank.csv', sep=',')  # Adjust file path if needed

# Display the first few rows of the dataset
print("Dataset Head:")
print(df.head())

# Check the column names
print("\nColumn Names:")
print(df.columns)

# Ensure 'deposit' exists in columns
if 'deposit' not in df.columns:
    df.columns = df.columns[0].split(',')
    df = df.iloc[1:].reset_index(drop=True)

# Check again
print("\nCorrected Column Names:")
print(df.columns)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Encode categorical columns using LabelEncoder
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Verify the dataset after encoding
print("\nEncoded Data Head:")
print(df.head())

# Split the data into features (X) and target (y)
if 'deposit' in df.columns:
    X = df.drop(columns=['deposit'])  # Drop the target column 'deposit'
    y = df['deposit']  # 'deposit' is the target column
else:
    raise KeyError("'deposit' column not found in dataset.")

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simplified Decision Tree Classifier model
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limit tree depth
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plotting a simplified Decision Tree
plt.figure(figsize=(20, 15))
plot_tree(
    dt_classifier,
    filled=True,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    rounded=True,
    fontsize=10
)
plt.title("Simplified Decision Tree Visualization (Max Depth = 3)")
plt.show()

# Visualizing the feature importance
feature_importance = dt_classifier.feature_importances_
features = X.columns

# Plotting Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importance, y=features)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
