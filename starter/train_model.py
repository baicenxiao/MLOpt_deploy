# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv('../data/census_nowhitespace.csv').iloc[:, 1:]

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb)

# Train and save a model.
model = train_model(X_train, y_train)
with open('../model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('saved model')

# Save onehot encoder
with open('../model/onehotencoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
print('saved onehot encoder')

# Slice
print("overall performance:")
preds = inference(model, X_test)
print(" ", compute_model_metrics(preds, y_test))

results = []


def slice_feature(feature_to_slice):
    for feature_slice in test[feature_to_slice].unique():
        slice_idx = test.loc[test[feature_to_slice] == feature_slice].index

        test_slice = test.loc[slice_idx]

        x_test_slice, y_test_slice, _, _ = process_data(
            test_slice, categorical_features=cat_features, label="salary", training=False,
            encoder=encoder, lb=lb
        )

        slice_preds = inference(model, x_test_slice)

        metrics = compute_model_metrics(slice_preds, y_test_slice)

        results.append([feature_to_slice, feature_slice] + list(metrics))


for feature in cat_features:
    slice_feature(feature)

results_df = pd.DataFrame(results)
results_df.columns = ['feature', 'slice', 'precision', 'recall', 'f1beta']
results_df.to_csv('slice_output.txt', index=False)
