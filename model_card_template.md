# Model Card
This is project 3 for MLOpt Undacity nano-degree
## Model Details
The model uses decision tree to predict whether income exceeds $50K/yr based on census data.

## Intended Use
This model is for study purposes and should not be used for commercial purposes.

## Training Data
The data used for this exercise was obtained from the UCI Machine Learning Repository:

[https://archive.ics.uci.edu/ml/datasets/census+income](https://archive.ics.uci.edu/ml/datasets/census+income)
## Evaluation Data
The base training data was split using a 80%/20% train and test split

## Metrics
We use precision, recall and the f1 score to measure model performance.

## Ethical Considerations
There have been historic income disparities among gender racial and ethnic groups. The data may be reflective of this and whatever models may trained will reinforce these inherent disparities.

## Caveats and Recommendations
No adjustments have been made to de-bias the model or the datasets.
This project is only for MLOpt study purpose.