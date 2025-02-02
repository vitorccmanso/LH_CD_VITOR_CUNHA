import os

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
    data.drop(columns=list_drop, inplace=True)
    data["nome"].fillna("No name", inplace=True)
    data["reviews_por_mes"].fillna(0, inplace=True)
    data = data[~(data["minimo_noites"] > 365)]
    save_path = os.path.join(save_folder, f"{save_filename}.csv")
    processed_df = data.to_csv(save_path, index=False)

    return processed_df