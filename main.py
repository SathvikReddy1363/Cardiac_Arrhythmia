from src.preprocess import load_data, binaryConverter, balancing_data, Scaling_data
from src.model import Logistic_model, KNN_model
from src.evaluate import evaluate_model

# Step 1 - Load data
train_df, test_df = load_data()

# Step 2 - Convert to binary
train_df, test_df = binaryConverter(train_df, test_df)

# Step 3 - Balance data
train_df_balanced = balancing_data(train_df)

# Step 4 - Scale data
X_train, X_test, y_train, y_test = Scaling_data(train_df_balanced, test_df)

# Step 5 - Train models
lr_model, lr_pred = Logistic_model(X_train, X_test, y_train)
knn_model, knn_pred = KNN_model(X_train, X_test, y_train)

# Step 6 - Evaluate
evaluate_model(y_test, lr_pred, "Logistic Regression")
evaluate_model(y_test, knn_pred, "KNN")