import logging

import mlflow
import mlflow.data
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.pandas_dataset import PandasDataset
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logging.getLogger('mlflow').setLevel(logging.DEBUG)


mlflow.set_tracking_uri('http://127.0.0.1:8080')

# Logging datasets with mlflow.log_input() API
california_housing = fetch_california_housing()
california_data: pd.DataFrame = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
california_target: pd.DataFrame = pd.DataFrame(california_housing.target, columns=['Target'])

california_housing_df: pd.DataFrame = pd.concat([california_data, california_target], axis=1)

dataset_source_url: str = 'https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz'
dataset_source: DatasetSource = HTTPDatasetSource(url=dataset_source_url)
dataset_name: str = 'California Housing Dataset'
dataset_target: str = 'Target'
dataset_tags = {
    'description': california_housing.DESCR,
}

dataset: PandasDataset = mlflow.data.from_pandas(
    df=california_housing_df, source=dataset_source, targets=dataset_target, name=dataset_name
)

print(f'Dataset name: {dataset.name}')
print(f'Dataset digest: {dataset.digest}')
print(f'Dataset source: {dataset.source}')
print(f'Dataset schema: {dataset.schema}')
print(f'Dataset profile: {dataset.profile}')
print(f'Dataset targets: {dataset.targets}')
print(f'Dataset predictions: {dataset.predictions}')
print(dataset.df.head())

for k, v in dataset.to_dict().items():
    print(f'{k}: {v}')

with mlflow.start_run():
    mlflow.log_input(dataset=dataset, context='training', tags=dataset_tags)

# Logging datasets when evaluating with mlflow.evaluate() API
X_train, X_test, y_train, y_test = train_test_split(california_data, california_target, test_size=0.25, random_state=42)

training_dataset_name: str = 'California Housing Training Dataset'
training_dataset_target: str = 'Target'
eval_dataset_name: str = 'California Housing Evaluation Dataset'
eval_dataset_target: str = 'Target'
eval_dataset_prediction: str = 'Prediction'


training_df: pd.DataFrame = pd.concat([X_train, y_train], axis=1)
training_dataset: PandasDataset = mlflow.data.from_pandas(
    df=training_df, source=dataset_source, targets=training_dataset_target, name=training_dataset_name
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train.to_numpy().flatten())


y_test_pred: pd.Series = model.predict(X=X_test)
eval_df: pd.DataFrame = X_test.copy()
eval_df[eval_dataset_target] = y_test.to_numpy().flatten()
eval_df[eval_dataset_prediction] = y_test_pred

eval_dataset: PandasDataset = mlflow.data.from_pandas(
    df=eval_df, targets=eval_dataset_target, name=eval_dataset_name, predictions=eval_dataset_prediction
)


mlflow.sklearn.autolog()
with mlflow.start_run():
    mlflow.log_input(dataset=training_dataset, context='training')

    mlflow.sklearn.log_model(model, artifact_path='rf', input_example=X_test)

    result = mlflow.evaluate(
        data=eval_dataset,
        predictions=None,
        model_type='regressor',
    )

    print(f'metrics: {result.metrics}')
