import pandas as pd
import pickle
from math import radians, sin, cos, sqrt, asin

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
        with open("app/artifacts/preprocessor.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)
        with open("app/artifacts/model.pkl", "rb") as f:
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

    def predict(self, data, path, manual_data=False):
        """
        Predicts rent prices

        Parameters:
        - data: The input data for prediction
        - path (str): The file path to save the CSV file
        - manual_data: Indicates whether it is a dataset prediction or a manual data input prediction

        Returns:
        - The predicted rent price(s)
        """
        # Check if the uploaded dataset contains all required columns
        columns = ["latitude", "longitude", "minimo_noites", "numero_de_reviews", "calculado_host_listings_count", "bairro_group", "room_type"]
        if not set(columns).issubset(data.columns):
            raise ValueError("Dataset must contain all the columns listed above")
        data = data[columns]

        # Preprocess data and make predictions
        transformed_data = self.preprocess_data(data.copy())
        prediction = self.model.predict(transformed_data)
        if manual_data:
            return prediction[0]
        results_df = data
        results_df["Predicted Price"] = prediction

        # Save results to a temporary CSV file and convert DataFrame to a list of dictionaries for rendering in HTML
        results_df.to_csv(path, index=False)
        results = results_df.to_dict(orient="records")
        return results

class CustomData:
    """
    A class representing custom datasets

    Attributes:
    - bairro_group: The neighborhood group of the ad
    - latitude: The latitude coordinate of the ad
    - longitude: The longitude coordinate of the ad
    - room_type: The type of room in the ad
    - minimo_noites: The minimum number of nights required for booking the ad
    - numero_de_reviews: The number of reviews for the ad
    - calculado_host_listings_count: The calculated host listings count for the ad
    """
    def __init__(self, bairro_group: str,
                    latitude: float,
                    longitude: float,
                    room_type: str,
                    minimo_noites: int,
                    numero_de_reviews: int,
                    calculado_host_listings_count: int):
        """
        Initializes the CustomData object with the provided attributes
        """
        self.bairro_group = bairro_group
        self.latitude = latitude
        self.longitude = longitude
        self.room_type = room_type
        self.minimo_noites = minimo_noites
        self.numero_de_reviews = numero_de_reviews
        self.calculado_host_listings_count = calculado_host_listings_count

    def get_data_as_dataframe(self):
        """
        Converts the CustomData object into a pandas DataFrame

        Returns:
        - pd.DataFrame: The CustomData object as a DataFrame
        """
        custom_data_input_dict = {
            "bairro_group": [self.bairro_group],
            "latitude": [self.latitude],
            "longitude": [self.longitude],
            "room_type": [self.room_type],
            "minimo_noites": [self.minimo_noites],
            "numero_de_reviews": [self.numero_de_reviews],
            "calculado_host_listings_count": [self.calculado_host_listings_count]
        }
        return pd.DataFrame(custom_data_input_dict)