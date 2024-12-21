from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
reconstructed_data = pd.read_csv("./Churn_Modelling.csv")
reconstructed_data["income_cat"] = pd.cut(
    reconstructed_data["EstimatedSalary"],
    bins=[0, 50000, 100000, 150000, 200000, np.inf],
    labels=[1, 2, 3, 4, 5]
)

# Stratified train-test split
strat_train_set, strat_test_set = train_test_split(
    reconstructed_data,
    test_size=0.20,
    stratify=reconstructed_data["income_cat"],
    random_state=42
)

# Drop the 'income_cat' column
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Prepare training data
recon_data = strat_train_set.copy()
recon_data = recon_data.drop(['CustomerId', 'Surname'], axis=1)

# Define feature groups
numeric_features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'EstimatedSalary']
categorical_features = ['Geography', 'Gender']
target = 'Exited'

# Split into features and target
X_train = recon_data.drop(target, axis=1)
y_train = recon_data[target]

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Define the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predict on the training data
churn_prediction = pipeline.predict(X_train)
original_rmse = np.sqrt(mean_squared_error(y_train, churn_prediction))
print('Original Model RMSE:', original_rmse)

# Save the pipeline
joblib.dump(pipeline, 'random_forest_model_v1.pkl')

# Reload and test the pipeline
reloaded_model = joblib.load('random_forest_model_v1.pkl')
reloaded_rmse = np.sqrt(mean_squared_error(y_train, reloaded_model.predict(X_train)))
print("Reloaded Model RMSE:", reloaded_rmse)



