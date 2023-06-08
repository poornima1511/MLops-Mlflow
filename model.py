from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix    
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error, r2_score, roc_auc_score, classification_report
from sklearn import preprocessing
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
import warnings
import sys
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import load_iris
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.sklearn.autolog()

# project_name=input("project_name:")
project_name=sys.argv[1]
try:
    exp = mlflow.create_experiment(name=project_name)
except:
    print("Experiment name already exsists")

exp_id= mlflow.set_experiment(experiment_name=project_name)


# Define a list of classification models and their corresponding hyperparameters
classification_models = [
    {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 4, 6]
        }
    },
    {
        'model': GradientBoostingClassifier(),
        'params': {
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [50, 100, 150],
            'max_depth' : [3,5,7]
           }
    },
    {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7]
        }
    }
]

# Define a list of regression models and their corresponding hyperparameters
regression_models = [
    {
        'model': GradientBoostingRegressor(),
        'params': {
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [50, 100, 150],
            'max_depth' : [3,5,7]
           }
    },
    {
        'model': DecisionTreeRegressor(),
        'params': {
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 4, 6]
        }
    },
    {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7]
        }
    }
]
# Define the evaluation metrics
classification_metrics = {
    'accuracy': make_scorer(accuracy_score),
    # 'roc': make_scorer(roc_auc_score),
    # 'classification_report' : make_scorer(classification_report)    
}
regression_metrics = {
    'mean_squared_error': make_scorer(mean_squared_error),
    'r2_score': make_scorer(r2_score)
}
# n_estimator = int(sys.argv[1])



if project_name=='regression':
    df = pd.read_csv('sale.csv')
    df["deal_id"]=df["deal_id"].apply(hash) 
    df["start_date"]=df["start_date"].apply(hash)
    X_regression=pd.read_csv('regree.csv')
    # X_regression=df.drop(["treatment"],axis=1)
    y_regression = df["treatment"]
    X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42
    )
    print("Regression being executed")
    
        # Train the data on each regression model with hyperparameter tuning
    for model_config in regression_models:
        model = model_config['model']
        model_name = model.__class__.__name__
        params = model_config['params']
        print(f'Training {model_name} with hyperparameter tuning for regression...')
        with mlflow.start_run(experiment_id=exp_id.experiment_id) as run: 
            # Perform grid search with cross-validation and multiple evalusation metrics
            grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3, scoring=regression_metrics, refit='mean_squared_error')
            
            # Train the model with the best parameters
            grid_search.fit(X_train_regression, y_train_regression)
            print(grid_search)
            best_model = grid_search.best_estimator_
            
            # Make predictions on the test set
            y_pred = best_model.predict(X_test_regression)
            
            # Calculate and print the evaluation metrics
            for metric_name, scorer in regression_metrics.items():
                metric_value = scorer._score_func(y_test_regression, y_pred)
                print(f'{metric_name} of {model_name}: {metric_value}')
            
            print(f'Best parameters: {grid_search.best_params_}\n')

else:
    data = load_breast_cancer()  
    X_classification = pd.read_csv('classifi.csv')
    y_classification = data.target
    X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42)
    for model_config in classification_models:
        model = model_config['model']
        model_name = model.__class__.__name__
        params = model_config['params']
        print(f'Training {model_name} with hyperparameter tuning for classification...')
        with mlflow.start_run(experiment_id=exp_id.experiment_id) as run: 
            # Perform grid search with cross-validation and multiple evaluation metrics
            grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3, scoring=classification_metrics, refit='accuracy')
            
            # Train the model with the best parameters
            grid_search.fit(X_train_classification, y_train_classification)
            best_model = grid_search.best_estimator_
            
            # Make predictions on the test set
            y_pred = best_model.predict(X_test_classification)
            fpr, tpr, thresholds = roc_curve(y_test_classification, y_pred)
            roc_auc = auc(fpr, tpr)
            # plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
            plt.plot(fpr, tpr, label=model_name)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig("roc.png")
            mlflow.log_artifact("roc.png")
            for metric_name, scorer in classification_metrics.items():
                metric_value = scorer._score_func(y_test_classification, y_pred)
                print(f'{metric_name} of {model_name}: {metric_value}')
            
            print(f'Best parameters: {grid_search.best_params_}\n')






