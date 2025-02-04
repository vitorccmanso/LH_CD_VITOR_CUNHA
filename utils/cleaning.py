import matplotlib.pyplot as plt
import seaborn as sns
import os

def detect_outliers_iqr(data, column):
    """
    Detect outliers in a specified column using the Interquartile Range (IQR) method

    Parameters:
    - data (pandas.DataFrame): The input DataFrame
    - column (str): The name of the column to check for outliers

    Returns:
    - outliers (pandas.DataFrame): The rows where values in the column are outliers
    - upper_bound (float): The upper bound for outlier detection
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)], upper_bound

def plot_boxplots(df, columns_to_exclude, rows, cols, figsize=(16, 16), show_outliers=True):
    """
    Function to plot boxplots of numerical columns in a DataFrame and optionally detect outliers

    Parameters:
    - df (pandas.DataFrame): The input DataFrame
    - columns_to_exclude (list): List of columns to exclude from plotting
    - rows (int): Number of rows for subplots
    - cols (int): Number of columns for subplots
    - figsize (tuple): Size of the figure (default is (16, 16))
    - show_outliers (bool): Whether to print outlier information (default is False)
    """
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    ax = ax.ravel()
    
    # Iterate over numerical columns (excluding specified ones)
    for i, col in enumerate(df.drop(columns=columns_to_exclude).select_dtypes("number")):
        sns.boxplot(y=df[col], ax=ax[i])
        ax[i].axhline(df[col].mean(), color="orange", linestyle="--")
        ax[i].set_title(f"Boxplot: {col}")
        # Print outliers if show_outliers is True
        if show_outliers:
            outliers, upper_bound = detect_outliers_iqr(df, col)
            if len(outliers) > 0:
                print(f"Outliers in {col}: {len(outliers)} | Upper bound value: {upper_bound}")
    plt.tight_layout()
    plt.show()

def cleaning_pipeline(data, list_drop, save_folder, save_filename):
    """
    Perform preprocessing on the input DataFrame and save the cleaned data to a CSV file

    Parameters:
    - data (pandas.DataFrame): The input DataFrame
    - list_drop (list): List of columns to drop
    - save_folder (str): The folder where the CSV file will be saved
    - save_filename (str): The name of the CSV file

    Returns:
    - processed_data (pandas.DataFrame): The preprocessed DataFrame
    """
    # Drop columns and fill NaN values
    data.drop(columns=list_drop, inplace=True)
    data["nome"].fillna("No name", inplace=True)
    data["reviews_por_mes"].fillna(0, inplace=True)

    # Calculate the 99th percentile for selected columns and filter the dataframe
    percentile_99 = data[["price", "numero_de_reviews", "reviews_por_mes", "calculado_host_listings_count"]].quantile(0.99)
    mask = (data[percentile_99.index] <= percentile_99.values).all(axis=1)
    data = data[mask]

    # Drop unrealistic values from "minimo_noites" and free ads
    data = data[(data["minimo_noites"] <= 365) & (data["price"] > 0)]

    # Save cleaned dataset to predefined folder
    save_path = os.path.join(save_folder, f"{save_filename}.csv")
    processed_df = data.to_csv(save_path, index=False)

    return processed_df