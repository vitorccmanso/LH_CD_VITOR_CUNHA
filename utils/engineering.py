import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from math import radians, sin, cos, sqrt, asin
from scipy import stats
from scipy.stats import skew, mode
from sklearn.feature_selection import f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

class Plots:
    """
    A class for plotting data distributions and correlations

    Attributes:
    - data (DataFrame): The dataset to be visualized

    Methods:
    - __init__: Initialize the Plots object with a DataFrame
    - plot_transformed_distributions: Plot distributions of a numeric column and its transformed versions
    - plot_corr: Plot a heatmap of the correlation matrix for numerical columns in the DataFrame
    """
    def __init__(self, data):
        """
        Initialize the Plots object with a DataFrame

        Parameters:
        - data (Dataframe): DataFrame containing the data
        """
        self.data = data

    def plot_transformed_distributions(self, col):
        """
        Plot distributions of a numeric column and its transformed versions

        Parameters:
        - col (str): Name of the column to plot
        """
        data = self.data[col]
        fig, ax = plt.subplots(2, 2, figsize=(16, 8))
        ax = ax.ravel()

        # Plot original distribution
        sns.histplot(data, kde=True, ax=ax[0])
        ax[0].set_title(f"Original Distribution\n Skewness:{round(skew(data), 2)}")
        self.plot_legends(ax, 0, data)

        # Check for non-positive values before log transformation
        log_transformed = None
        has_non_positive = np.any(data < 0)

        # Plot log transformation if valid
        if not has_non_positive:
            log_transformed = np.log1p(data)
            sns.histplot(log_transformed, kde=True, ax=ax[1])
            ax[1].set_title(f"Log Transformation\n Skewness: {round(skew(log_transformed), 2)}")
            self.plot_legends(ax, 1, log_transformed)
        else:
            ax[1].text(0.5, 0.5, "Log Transformation\n Not Applicable", fontsize=15, ha="center", va="center")
            ax[1].set_title("Log Transformation\n Not Applicable")

        # Apply cubic root transformation
        sqrt_transformed = np.cbrt(data)
        sns.histplot(sqrt_transformed, kde=True, ax=ax[2])
        ax[2].set_title(f"Cubic Transformation\n Skewness: {round(skew(sqrt_transformed), 2)}")
        self.plot_legends(ax, 2, sqrt_transformed)

        skewness_values = [skew(data)]
        if log_transformed is not None:
            skewness_values.append(skew(log_transformed))
        skewness_values.append(skew(sqrt_transformed))
        closest_to_zero_idx = np.argmin(np.abs(skewness_values))

        # Plot boxplot of the transformation with the lowest skewness
        if closest_to_zero_idx == 0:
            sns.boxplot(x=data, ax=ax[3])
        elif closest_to_zero_idx == 1 and log_transformed is not None:
            sns.boxplot(x=log_transformed, ax=ax[3])
        else:
            sns.boxplot(x=sqrt_transformed, ax=ax[3])
        ax[3].set_title(f"Boxplot: Best Distribution")
        plt.tight_layout()
        plt.show()

    def plot_legends(self, ax, pos, data):
        """
        Plot legends on a given axis for a specific plot position

        Parameters:
        - ax: Axis object to plot on
        - pos (int): Position in the subplot grid
        - data: Data for which legends are plotted
        """
        ax[pos].axvline(data.mean(), color="r", linestyle="--", label="Mean: {:.2f}".format(data.mean()))
        ax[pos].axvline(mode(data)[0], color="g", linestyle="--", label="Mode: {:.2f}".format(mode(data)[0]))
        ax[pos].axvline(data.median(), color="b", linestyle="--", label="Median: {:.2f}".format(data.median()))
        ax[pos].legend()

    def plot_corr(self, method, figsize):
        """
        Plots a heatmap of the correlation matrix for numerical columns in the DataFrame

        Parameters:
        - method (str): Correlation method to use ("pearson", "kendall", or "spearman")
        """
        plt.figure(figsize=figsize)
        sns.heatmap(self.data.select_dtypes(include="number").corr(method=method), annot=True, fmt=".2f", cmap="RdYlGn")
        plt.show()

def calculate_distance(lat, lon, point_coordinates):
    """
    Calculates the distance between two points on Earth using the Haversine formula

    Parameters:
    - lat (float): Latitude of the first point (ad) in degrees
    - lon (float): Longitude of the first point (ad) in degrees
    - point_coordinates (tuple): Latitude and longitude of the second point (reference point) as a tuple (latitude, longitude) in degrees

    Returns:
    - float: The distance between the two points in kilometers
    """
    point_lat = radians(point_coordinates[0])
    point_lon = radians(point_coordinates[1])
    lat = radians(lat)
    lon = radians(lon)

    earth_radius = 6378
    difference_lat = lat - point_lat
    difference_lon = lon - point_lon

    a = sin(difference_lat / 2) * sin(difference_lat / 2) + cos(point_lat) * cos(lat) * sin(difference_lon / 2) * sin(difference_lon / 2)
    central_angle = 2 * asin(sqrt(a))

    distance = earth_radius * central_angle

    return distance

def statistical_tests(data, target):
    """
    Performs statistical tests to evaluate feature relevance and multicollinearity

    Parameters:
    - data (Dataframe): DataFrame containing the dataset
    - target (str): The name of the target column

    Returns:
    - f_regression_results (Dataframe): DataFrame containing F-Scores and P-Values for numerical features using f_regression
    - f_oneway_results (Dataframe): DataFrame containing P-Values for categorical features using ANOVA (f_oneway)
    - vif_data (Dataframe): DataFrame containing Variance Inflation Factor (VIF) values for numerical features to check multicollinearity
    """
    # Perform F-regression test for numerical features to measure their relevance to the target
    f_regression_data = data.select_dtypes(include="number").drop(columns=target)
    f_scores, p_values = f_regression(f_regression_data, data[target])
    f_regression_results = pd.DataFrame({'Feature': f_regression_data.columns, 'F-Score': f_scores, 'P-Value': p_values})

    # Perform ANOVA test (f_oneway) for categorical features to assess their relationship with the target
    f_oneway_results = {}
    cat_features = data.select_dtypes(include="object").columns
    for feature in cat_features:
        f_value, p_value = stats.f_oneway(*(data[data[feature] == value][target] for value in data[feature].unique()))
        f_oneway_results[feature] = p_value
    f_oneway_results = pd.DataFrame(f_oneway_results.items(), columns=["Feature", "p-value"])

    # Calculate Variance Inflation Factor to detect multicollinearity among numerical features
    f_regression_data['intercept'] = 1 
    vif_data = pd.DataFrame()
    vif_data["Feature"] = f_regression_data.columns
    vif_data["VIF"] = [variance_inflation_factor(f_regression_data.astype(float), i) for i in range(f_regression_data.shape[1])]

    return f_regression_results, f_oneway_results, vif_data

def saving_dataset(data, save_folder, save_filename):
    """
    Saves the dataset, and creates the specified folder if it doesn't exist

    Parameters:
    - data (Dataframe): DataFrame containing the original dataset
    - save_folder (str): Folder path where the datasets will be saved
    - save_filename (str): Base filename for the saved datasets
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    original_save_path = os.path.join(save_folder, f"{save_filename}.csv")
    data.to_csv(original_save_path, index=False)