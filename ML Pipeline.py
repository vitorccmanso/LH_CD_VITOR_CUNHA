import pandas as pd
from math import radians, sin, cos, sqrt, asin
import pickle
import warnings
import tkinter as tk
from tkinter import filedialog
warnings.filterwarnings("ignore")

class PredictPipeline:
    """
    A class for predicting listing rent prices using a pre-trained model and preprocessing pipeline

    Methods:
    - __init__: Initializes the PredictPipeline object by loading the preprocessor and model from .pkl files
    - calculate_distance: Calculates the distance between two points on Earth using the Haversine formula
    - preprocess_data: Preprocesses the input data, including feature engineering and transformation
    - predict: Predicts rent prices
    """
    def __init__(self):
        """
        Initializes the PredictPipeline object by loading the preprocessor and model from .pkl files
        """
        with open("./artifacts/preprocessor_without_bairro.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)
        with open("./artifacts/model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def calculate_distance(self, lat, lon):
        """
        Calculates the distance between two points on Earth using the Haversine formula

        Parameters:
        - lat (float): Latitude of the first point (ad) in degrees
        - lon (float): Longitude of the first point (ad) in degrees

        Returns:
        - float: The distance between the two points in kilometers
        """
        point_lat = radians(40.7484053)
        point_lon = radians(-73.9856019)
        lat = radians(lat)
        lon = radians(lon)

        earth_radius = 6378
        difference_lat = lat - point_lat
        difference_lon = lon - point_lon

        a = sin(difference_lat / 2) * sin(difference_lat / 2) + cos(point_lat) * cos(lat) * sin(difference_lon / 2) * sin(difference_lon / 2)
        central_angle = 2 * asin(sqrt(a))

        distance = earth_radius * central_angle

        return distance

    def preprocess_data(self, input_data):
        """
        Preprocesses the input data, including feature engineering and transformation

        Parameters:
        - input_data (pandas.DataFrame): The input data to be processed

        Returns:
        - pandas.DataFrame: The processed input data
        """
        one_hot_cols = input_data.select_dtypes(include="object").columns

        # Create the new feature "distance_to_empire_state"
        input_data["distance_to_empire_state"] = input_data.apply(lambda row: self.calculate_distance(row["latitude"], row["longitude"]), axis=1)

        # Apply preprocessor object to input_data
        input_data = self.preprocessor.transform(input_data)
        one_hot_features = list(self.preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(one_hot_cols))
        feature_names = ["latitude", "longitude", "minimo_noites", "numero_de_reviews", "calculado_host_listings_count", "distance_to_empire_state"] + one_hot_features
        return pd.DataFrame(input_data, columns=feature_names)

    def predict(self, data):
        """
        Predicts clients churn

        Parameters:
        - data: The input data for prediction
        """
        # Check if the uploaded dataset contains all required columns
        columns = ["latitude", "longitude", "minimo_noites", "numero_de_reviews", "calculado_host_listings_count", "bairro_group", "room_type"]
        if not set(columns).issubset(data.columns):
            raise ValueError("Dataset must contain all the columns listed above")
        data = data[columns]

        # Preprocess data and make predictions
        transformed_data = self.preprocess_data(data.copy())
        prediction = self.model.predict(transformed_data)
        results_df = data
        results_df["Predicted Price"] = prediction
        results_df.to_csv("results.csv", index=False)
        print("Predictions saved to results.csv")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    dataset_path = filedialog.askopenfilename(
        title="Select Dataset",
        filetypes=[("CSV files", "*.csv"), ("Excel Files", "*.xlsx *.xls")]
    )
    if dataset_path:
        try:
            data = pd.read_csv(dataset_path, sep=None, engine="python")
            results = PredictPipeline().predict(data)           
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("No file was selected")