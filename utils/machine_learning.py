import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from utils.engineering import calculate_distance
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats
import pickle

mlflow_tracking_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
uri = os.environ.get("uri")

class DataPreprocess:
    """
    A class for preprocessing data, including feature engineering, transformation, and splitting data into train-test sets

    Methods:
    - __init__: Initializes the DataPreprocess object
    - save_preprocessor: Saves the preprocessor object to a file
    - load_preprocessor: Loads the preprocessor object from a file
    - get_feature_names: Retrieves feature names after applying transformations in the preprocessor pipeline
    - preprocessor: Creates and returns a preprocessor pipeline for data preprocessing
    - preprocess_data: Preprocesses data for training, applying transformations, feature engineering, and splitting into train-test sets
    """
    def __init__(self):
        pass

    def save_preprocessor(self, preprocessor, bairro_column):
        """
        Saves the preprocessor object to a file

        Parameters:
        - preprocessor (sklearn.pipeline.Pipeline): The preprocessor object to be saved
        - bairro_column (bool): Flag indicating if the input data contains the "bairro" column or not
        """
        if not os.path.exists("../artifacts"):
            os.makedirs("../artifacts")
        if not bairro_column:
            with open("../artifacts/preprocessor_without_bairro.pkl", "wb") as f_all:
                pickle.dump(preprocessor, f_all)
        else:
            with open("../artifacts/preprocessor_with_bairro.pkl", "wb") as f_all:
                pickle.dump(preprocessor, f_all)
    
    def load_preprocessor(self, preprocessor):
        """
        Loads the preprocessor object from a file

        Parameters:
        - preprocessor (str): The name of the preprocessor to be loaded

        Returns:
        - preprocessor: The loaded preprocessor object
        """
        with open(f"../artifacts/{preprocessor}.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        return preprocessor
    
    def get_feature_names(self, preprocessor, num_only_scale_cols, log_cols, cbrt_cols, one_hot_cols):
        """
        Retrieves the feature names after preprocessing is applied

        Parameters:
        - preprocessor (sklearn.pipeline.Pipeline): The preprocessor object used for transformations
        - num_only_scale_cols (list): List of numeric columns without transformations
        - log_cols (list): List of column names for which log transformation is applied
        - cbrt_cols (list): List of column names for which cubic root transformation is applied
        - one_hot_cols (list): List of categorical columns for which OneHotEncoding is applied

        Returns:
        - feature_names (list): List of feature names after preprocessing
        """
        numeric_features = num_only_scale_cols + log_cols + cbrt_cols
        one_hot_features = list(preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(one_hot_cols))
        feature_names = numeric_features + one_hot_features
        return feature_names

    def preprocessor(self, num_only_scale_cols, log_cols, cbrt_cols, one_hot_cols):
        """
        Creates and returns a preprocessor pipeline for data preprocessing

        Parameters:
        - num_only_scale_cols (list): List of numeric columns without transformations
        - log_cols (list): List of column names for which log transformation is applied
        - cbrt_cols (list): List of column names for which cubic root transformation is applied
        - one_hot_cols (list): List of categorical columns for which OneHotEncoding is applied

        Returns:
        - preprocessor (sklearn.compose.ColumnTransformer): Preprocessor pipeline for data preprocessing
        """
        # Define transformers for numeric columns
        log_transformer = Pipeline(steps=[
            ("log_transformation", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", RobustScaler())
        ])
        cubic_transformer = Pipeline(steps=[
            ("sqrt_transformation", FunctionTransformer(np.cbrt, validate=True)),
            ("scaler", RobustScaler())
        ])

        #Define transformer for categorical columns
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder())
        ])

        # Combine transformers for numeric and categorical columns
        preprocessor = ColumnTransformer(
            transformers=[("num_only_scale", RobustScaler(), num_only_scale_cols),
                          ("num_log", log_transformer, log_cols),
                          ("num_sqrt", cubic_transformer, cbrt_cols), 
                          ("cat", categorical_transformer, one_hot_cols)], verbose_feature_names_out=False)
        return preprocessor

    def preprocess_data(self, data, target_name=None, test_size=None, bairro_column=False, test_data=False, preprocessor=None):
        """
        Preprocesses the input data, including feature engineering, transformation, and splitting into train-test sets

        Parameters:
        - data (pandas.DataFrame): The input DataFrame containing raw data
        - target_name (str): The name of the target variable to predict
        - test_size (float): Proportion of the dataset to include in the test split
        - bairro_column (bool): Flag indicating if the input data contains the "bairro" column or not
        - test_data (bool): Flag indicating if the input data is test data. If true, the function returns only the test dataframe
        - preprocessor (str): The name of the preprocessor to be loaded

        Returns:
        - X (pandas.Dataframe): Preprocessed test data for predictions
        - X_train (pandas.DataFrame): Preprocessed features for training set
        - X_test (pandas.DataFrame): Preprocessed features for testing set
        - y_train (pandas.Series): Target labels for training set
        - y_test (pandas.Series): Target labels for testing set
        """
        # Drop unnecessary columns and specify columns for transformations
        data_process = data.drop(columns=["id", "host_id", "host_name", "ultima_review", "nome", "reviews_por_mes", "disponibilidade_365", target_name], errors="ignore")
        if not bairro_column:
            data_process = data_process.drop(columns="bairro")
        log_cols = ["minimo_noites", "numero_de_reviews", "calculado_host_listings_count"]
        cbrt_cols = ["distance_to_empire_state"]
        num_only_scale_cols = ["latitude", "longitude"]
        one_hot_cols = data_process.select_dtypes("object").columns

        # Apply preprocessor to test data
        if test_data:
            data_process["distance_to_empire_state"] = data_process.apply(lambda row: calculate_distance(row["latitude"], row["longitude"], (40.7484053, -73.9856019)), axis=1)
            preprocessor = self.load_preprocessor(preprocessor)
            X = preprocessor.transform(data_process)
            if bairro_column:
                X = X.toarray()
            feature_names = self.get_feature_names(preprocessor, num_only_scale_cols, log_cols, cbrt_cols, one_hot_cols)
            X = pd.DataFrame(X, columns=feature_names)
            return X

        # Build preprocessor, fit and transform data, get feature names and create final dataframe of preprocessed data
        preprocessor = self.preprocessor(num_only_scale_cols, log_cols, cbrt_cols, one_hot_cols)
        data_preprocessed = preprocessor.fit_transform(data_process)
        if bairro_column:
            data_preprocessed = data_preprocessed.toarray()
        feature_names = self.get_feature_names(preprocessor, num_only_scale_cols, log_cols, cbrt_cols, one_hot_cols)
        data_preprocessed = pd.DataFrame(data_preprocessed, columns=feature_names)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(data_preprocessed, data[target_name], test_size=test_size, shuffle=True, random_state=42)

        # Save preprocessor if not already saved
        self.save_preprocessor(preprocessor, bairro_column)

        return X_train, X_test, y_train, y_test

class ModelTraining:
    """
    A class for training machine learning models, evaluating their performance, and saving the best one

    Methods:
    - __init__: Initializes the ModelTraining object
    - save_model: Saves the specified model to a pkl file
    - initiate_model_trainer: Initiates the model training process and evaluates multiple models
    - evaluate_models: Evaluates multiple models using random search cross-validation and logs the results with MLflow
    """
    def __init__(self):
        pass

    def save_model(self, model_name, version, save_folder, save_filename):
        """
        Save the specified model to a pkl file

        Parameters:
        - model_name (str): The name of the model to save
        - version (int): The version of the model to save
        - save_folder (str): The folder path where the model will be saved
        - save_filename (str): The filename for the pkl file
        """
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)

        # Get the correct version of the registered model
        model_versions = client.search_model_versions(f"name='{model_name}'")
        model_versions_sorted = sorted(model_versions, key=lambda v: int(v.version))
        requested_version = model_versions_sorted[version - 1]
        
        # Construct the logged model path
        run_id = requested_version.run_id
        artifact_path = requested_version.source.split("/")[-1]
        logged_model = f"runs:/{run_id}/{artifact_path}"

        # Load the model from MLflow and saves it to a pkl file
        loaded_model = mlflow.sklearn.load_model(logged_model)
        file_path = os.path.join(save_folder, f"{save_filename}.pkl")
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, "wb") as f:
            pickle.dump(loaded_model, f)

    def initiate_model_trainer(self, train_test, experiment_name, bairro_column=False):
        """
        Initiates the model training process

        Parameters:
        - train_test (tuple): A tuple containing the train-test split data in the format (X_train, y_train, X_test, y_test)
        - experiment_name (str): Name of the MLflow experiment where the results will be logged
        - bairro_column (bool): A boolean indicating whether the model is trained using the "bairro" column or not

        Returns:
        - dict: A dictionary containing the evaluation report for each model
        """
        mlflow.set_tracking_uri(uri)
        X_train, y_train, X_test, y_test = train_test
        
        models = {
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42)
        }
        
        params = {
            "Ridge": {
                "alpha":[0.01, 0.1, 0.5, 1], 
                "max_iter":[1000, 3000, 5000], 
                "tol": [0.0001, 0.001, 0.01, 0.1]
            },
            "Lasso":{
                "alpha":[0.01, 0.1, 0.5, 1], 
                "max_iter":[1000, 3000, 5000], 
                "tol": [0.0001, 0.001, 0.01, 0.1]
            },
            "Random Forest":{
                "criterion":["squared_error", "absolute_error", "poisson"],
                "max_features":["sqrt","log2"],
                "n_estimators": [25, 50, 100, 150, 200],
                "max_depth": [5, 10, 20, 30]
            },
            "XGBoost":{
                "n_estimators": [25, 50, 100, 150, 200],
                "max_depth": [2, 5, 10, 20],
                "learning_rate": [0.01, 0.1, 0.2, 0.3]
            }
        }
        
        model_report = self.evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                           models=models, params=params, experiment_name=experiment_name, bairro_column=bairro_column)
        
        return model_report

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params, experiment_name, bairro_column):
        """
        Evaluates multiple models using random search cross-validation and logs the results with MLflow

        Parameters:
        - X_train (array-like): Features of the training data
        - y_train (array-like): Target labels of the training data
        - X_test (array-like): Features of the testing data
        - y_test (array-like): Target labels of the testing data
        - models (dict): A dictionary containing the models to be evaluated
        - params (dict): A dictionary containing the hyperparameter grids for each model
        - experiment_name (str): Name of the MLflow experiment where the results will be logged
        - bairro_column (bool): A boolean indicating whether the model is trained using the "bairro" column or not

        Returns:
        - dict: A dictionary containing the evaluation report for each model.
        """
        mlflow.set_experiment(experiment_name)
        report = {}
        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                param = params[model_name]
                rs = RandomizedSearchCV(model, param, cv=5, scoring=["neg_mean_absolute_error", "r2"], refit="neg_mean_absolute_error", random_state=42)
                search_result = rs.fit(X_train, y_train)
                model = search_result.best_estimator_
                y_pred = model.predict(X_test)
                mlflow.set_tags({"model_type": f"{model_name}-{experiment_name}", "bairro_column": bairro_column})

                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = root_mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Log metrics to MLflow
                mlflow.log_params(search_result.best_params_)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.sklearn.log_model(model, model_name, registered_model_name=f"{model_name} - {experiment_name}")
                
                # Store the model for visualization
                report[model_name] = {"model": model, "y_pred": y_pred, "mae": mae, "rmse": rmse, "r2": r2}        
        return report


class MetricsVisualizations:
    """
    A class for visualizing model evaluation metrics and results

    Attributes:
    - models (dict): A dictionary containing the trained models, with metrics and predictions for each model

    Methods:
    - __init__: Initializes the MetricsVisualizations object with a dictionary of models
    - create_subplots: Creates a figure and subplots with common settings
    - plot_pred_x_real: Plots predicted vs real values for each model
    - plot_feature_importance: Plots feature importance for each model
    - plot_residuals: Plots residuals, residual autocorrelation, and residual distribution for each model
    """
    def __init__(self, models):
        """
        Initializes the MetricsVisualizations object with a dictionary of models

        Parameters:
        - models (dict): A dictionary containing the trained models, with metrics and predictions for each model
        """
        self.models = models

    def create_subplots(self, rows, columns, figsize=(18,12)):
        """
        Creates a figure and subplots with common settings

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        - figsize (tuple): Figure size. Default is (18, 12)
        
        Returns:
        - fig (matplotlib.figure.Figure): The figure object
        - ax (numpy.ndarray): Array of axes objects
        """
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        ax = ax.ravel()
        return fig, ax

    def plot_pred_x_real(self, y_test, rows, columns):
        """
        Plots predicted vs real values for each model

        Parameters:
        - y_test (array-like): True labels of the test data
        - rows (int): Number of rows for subplots
        - columns (int): Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns)
        for i, (model_name, model_data) in enumerate(self.models.items()):
            # Get predicted values and make a copy of y_test
            y_pred = pd.Series(model_data["y_pred"])
            y_test_copy = y_test.copy()

            # Reset indices for easy plotting
            y_pred.reset_index(drop=True, inplace=True)
            y_test_copy.reset_index(drop=True, inplace=True)

            # Create DataFrame for plotting
            df_plot = pd.DataFrame({"Predicted Values": y_pred.values, "Real Values": y_test_copy.values})

            # Plot scatter plot and regression line
            sns.scatterplot(data=df_plot, x="Predicted Values", y="Real Values", ax=ax[i])
            sns.regplot(data=df_plot, x="Predicted Values", y="Real Values", ax=ax[i], scatter=False, color="red", line_kws={"linewidth": 2})
            ax[i].set_title(f"Predicted x Real Values: {model_name}")
            ax[i].set_xlabel("Predicted Values")
            ax[i].set_ylabel("Real Values")

        fig.tight_layout()
        plt.show()

    def plot_feature_importance(self, y_test, X_test, rows, columns):
        """
        Plots feature importance for each model using permutation importance

        Parameters:
        - y_test (array-like): True labels of the test data
        - X_test (DataFrame): Features of the test data, where each column represents a feature
        - rows (int): Number of rows for the subplot grid
        - columns (int): Number of columns for the subplot grid
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(16, 10))
        for i, (model_name, model_data) in enumerate(self.models.items()):
            # Calculate and sort permutation importances
            result = permutation_importance(model_data["model"], X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring="neg_mean_absolute_error")
            sorted_importances_idx = result["importances_mean"].argsort()
            importances = pd.DataFrame(result.importances[sorted_importances_idx].T, columns=X_test.columns[sorted_importances_idx])

            # If X_test has more than 30 columns, limit to the top 15 features
            if X_test.shape[1] > 30:
                importances = importances.iloc[:, -15:]

            # Plot boxplot of feature importances
            box = importances.plot.box(vert=False, whis=10, ax=ax[i])
            box.set_title(f"Feature Importance - {model_name}")
            box.axvline(x=0, color="k", linestyle="--")
            box.set_xlabel(f"Increase in Mean Absolute Error")
            box.figure.tight_layout()

        fig.tight_layout()
        plt.show()

    def plot_residuals(self, y_test, rows, columns):
        """
        Plots residuals, residual autocorrelation, and residual distribution for each model

         Parameters:
        - y_test (array-like): True labels of the test data
        - rows (int): Number of rows for subplots
        - columns (int): Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns)
        for i, (model_name, model_data) in enumerate(self.models.items()):
            # Calculate residuals
            y_pred = model_data["y_pred"]
            residuals = y_test - y_pred

            # Plot residual plot
            sns.scatterplot(x=y_pred, y=residuals, ax=ax[i * columns])
            ax[i * columns].set_xlabel("Predicted Values")
            ax[i * columns].set_ylabel("Residuals")
            ax[i * columns].axhline(y=0, color="r", linestyle="--")
            ax[i * columns].set_title(f"Residual Plot - {model_name}")

            # Plot residual autocorrelation
            plot_acf(residuals, lags=40, ax=ax[i * columns + 1])
            ax[i * columns + 1].set_title(f"Residual Autocorrelation - {model_name}")
            ax[i * columns + 1].set_xlabel("Lags")
            ax[i * columns + 1].set_ylabel("Autocorrelation")

            # Plot residual distribution
            stats.probplot(residuals, dist="norm", plot=ax[i * columns + 2])
            ax[i * columns + 2].set_title(f"Residual Distribution - {model_name}")

        fig.tight_layout()
        plt.show()