# Q1. Preprocess the dataset by handling missing values, encoding categorical variables, and scaling the numerical features if necessary.
Here's how you can preprocess the dataset:

python
Copy code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Assuming you have a dataset named 'heart_disease_data.csv'
# Load the dataset
df = pd.read_csv('heart_disease_data.csv')

# Handle missing values
# Example assuming 'age' and 'chol' are numerical features, 'sex' and 'cp' are categorical features
numerical_features = ['age', 'chol']
categorical_features = ['sex', 'cp']

# Use SimpleImputer for numerical features
numerical_transformer = SimpleImputer(strategy='mean')

# Use SimpleImputer and OneHotEncoder for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Splitting the dataset into features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Preprocess and split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit and transform the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Optionally, scale numerical features if needed (depends on the algorithm used)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
Q2. Split the dataset into a training set (70%) and a test set (30%).
Already done in the preprocessing step above using train_test_split.

Q3. Train a random forest classifier on the training set using 100 trees and a maximum depth of 10 for each tree. Use the default values for other hyperparameters.
python
Copy code
# Train the random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_classifier.fit(X_train, y_train)
Q4. Evaluate the performance of the model on the test set using accuracy, precision, recall, and F1 score.
python
Copy code
# Predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
Q5. Use the feature importance scores to identify the top 5 most important features in predicting heart disease risk. Visualize the feature importances using a bar chart.
python
Copy code
# Get feature importances
feature_importances = rf_classifier.feature_importances_
features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names(categorical_features).tolist() + numerical_features

# Create DataFrame to visualize feature importances
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
top_5_features = feature_importance_df.nlargest(5, 'Importance')

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(top_5_features['Feature'], top_5_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Top 5 Features Importance')
plt.gca().invert_yaxis()  # Invert y-axis to display most important feature at the top
plt.show()
Q6. Tune the hyperparameters of the random forest classifier using grid search or random search. Try different values of the number of trees, maximum depth, minimum samples split, and minimum samples leaf. Use 5-fold cross-validation to evaluate the performance of each set of hyperparameters.
python
Copy code
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Accuracy:", grid_search.best_score_)
Q7. Report the best set of hyperparameters found by the search and the corresponding performance metrics. Compare the performance of the tuned model with the default model.
After running the grid search, you can compare the performance metrics of the best model (grid_search.best_estimator_) with the default model trained earlier.

Q8. Interpret the model by analyzing the decision boundaries of the random forest classifier. Plot the decision boundaries on a scatter plot of two of the most important features. Discuss the insights and limitations of the model for predicting heart disease risk.
To analyze decision boundaries and visualize them, you can plot a scatter plot with the two most important features and color code the regions based on the predictions made by the model. This visualization helps understand how the model separates different classes based on the chosen features.

python
Copy code
# Example plot decision boundaries (assuming 'age' and 'chol' are two most important features)
# Assuming X_test_2d is the 2D data with just two features
X_test_2d = X_test[:, :2]  # Selecting first two features for demonstration

# Plot decision boundaries
x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = rf_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, s=20, edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Decision Boundaries of Random Forest Classifier')
plt.show()
This plot visualizes how the random forest classifier separates the data points into different classes based on 'age' and 'cholesterol' (or any other chosen features). Insights from this plot can help understand how the model predicts heart disease risk based on these features. Limitations may include overfitting if the model is too complex or if features are not sufficiently informative.

These steps provide a comprehensive approach to building, evaluating, tuning, and interpreting a random forest classifier for predicting heart disease risk based on the given dataset. Adjustments may be needed based on specific requirements and characteristics of the dataset used.






