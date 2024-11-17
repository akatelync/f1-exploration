import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import resample
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from datetime import datetime


@dataclass
class F1TelemetryAnalyzer:
    """
    A class for analyzing Formula 1 telemetry data, providing methods for preprocessing,
    analysis, and visualization of lap-by-lap data.

    This analyzer supports filtering by events and drivers, resampling of telemetry data,
    distance matrix calculations, and comprehensive statistical analysis including PCA.

    Attributes:
        n_samples (int): Number of samples to resample each lap's telemetry data to
        feature_cols (List[str]): List of telemetry features to analyze
    """
    n_samples: int = 300
    feature_cols = ["RPM", "Speed", "nGear", "Throttle", "Brake", "DRS"]

    def filter_data(self,
                    telemetry_data: pd.DataFrame,
                    events: Optional[List[str]] = None,
                    drivers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter telemetry data by events and/or drivers.

        Args:
            telemetry_data: Raw telemetry DataFrame containing all events and drivers
            events: Optional list of event names to include in the filtered data
            drivers: Optional list of driver names to include in the filtered data

        Returns:
            pd.DataFrame: Filtered telemetry data containing only specified events and drivers
        """
        filtered_data = telemetry_data.copy()

        if events:
            filtered_data = filtered_data[filtered_data["Event"].isin(events)]
        if drivers:
            filtered_data = filtered_data[filtered_data["Driver"].isin(
                drivers)]

        return filtered_data

    def preprocess_lap_data(self,
                            telemetry_data: pd.DataFrame,
                            events: Optional[List[str]] = None,
                            drivers: Optional[List[str]] = None):
        """
        Preprocess telemetry data by aligning and resampling lap data to fixed length.

        This method performs several preprocessing steps:
        1. Filters data by specified events and drivers
        2. Resamples each lap's telemetry to a fixed number of points
        3. Calculates various lap metrics (duration, speeds, brake applications, etc.)

        Args:
            telemetry_data: Raw telemetry DataFrame to process
            events: Optional list of events to include
            drivers: Optional list of drivers to include

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing:
                - Processed telemetry data with resampled features
                - DataFrame of lap metrics including duration, speeds, etc.
        """
        filtered_data = self.filter_data(telemetry_data, events, drivers)

        grouped = filtered_data.groupby(["Event", "Driver", "LapNumber"])
        processed_laps = []
        lap_metrics = []

        for (event, driver, lap), lap_data in grouped:
            lap_data = lap_data.sort_values("Time")
            lap_duration = (lap_data["Time"].max() -
                            lap_data["Time"].min()).total_seconds()
            resampled_data = {}

            for feature in self.feature_cols:
                if feature in lap_data.columns:
                    resampled = resample(lap_data[feature], self.n_samples)
                    resampled_data[feature] = resampled

            resampled_df = pd.DataFrame(resampled_data)
            resampled_df["Event"] = event
            resampled_df["Driver"] = driver
            resampled_df["LapNumber"] = lap
            resampled_df["normalized_time"] = np.linspace(0, 1, self.n_samples)
            processed_laps.append(resampled_df)

            metrics = {
                "event": event,
                "driver": driver,
                "lap": lap,
                "duration": lap_duration,
                "max_speed": lap_data["Speed"].max(),
                "avg_speed": lap_data["Speed"].mean(),
                "brake_applications": len(lap_data[lap_data["Brake"] > 0]),
                "drs_zones": len(lap_data[lap_data["DRS"] > 0])
            }
            lap_metrics.append(metrics)

        return pd.concat(processed_laps), pd.DataFrame(lap_metrics)

    def calculate_distance_matrix(self,
                                  processed_data: pd.DataFrame,
                                  feature: str = "Speed",
                                  by_event: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Calculate Euclidean distances between average lap profiles for each driver.

        Computes distance matrices showing how similar or different drivers' lap profiles
        are from each other, based on a specified telemetry feature.

        Args:
            processed_data: Preprocessed telemetry data
            feature: Telemetry feature to use for distance calculation (default: "Speed")
            by_event: Whether to calculate separate distance matrices for each event

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping event names to distance matrices.
                If by_event is False, contains single matrix under key "all"
        """
        distance_matrices = {}

        if by_event:
            events = processed_data["Event"].unique()
            for event in events:
                event_data = processed_data[processed_data["Event"] == event]
                distance_matrices[event] = self._calculate_single_distance_matrix(
                    event_data, feature)
        else:
            distance_matrices["all"] = self._calculate_single_distance_matrix(
                processed_data, feature)

        return distance_matrices

    def _calculate_single_distance_matrix(self, data: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Helper method to calculate distance matrix for a single dataset.

        Computes Euclidean distances between average lap profiles for each pair of drivers
        in the dataset.

        Args:
            data: Telemetry data for a single event or all events combined
            feature: Telemetry feature to use for distance calculation

        Returns:
            pd.DataFrame: Square matrix of distances between drivers' average lap profiles
        """
        drivers = data["Driver"].unique()
        distance_matrix = np.zeros((len(drivers), len(drivers)))
        driver_profiles = {}

        for driver in drivers:
            driver_data = data[data["Driver"] == driver]
            profile_matrix = np.array(driver_data.groupby(
                "LapNumber")[feature].apply(list).tolist())
            mean_profile = np.mean(profile_matrix, axis=0)
            driver_profiles[driver] = mean_profile

        for i, driver1 in enumerate(drivers):
            for j, driver2 in enumerate(drivers):
                if i <= j:
                    distance = np.sqrt(
                        np.sum((driver_profiles[driver1] - driver_profiles[driver2]) ** 2))
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance

        return pd.DataFrame(distance_matrix, index=drivers, columns=drivers)

    def analyze_laps(self,
                     telemetry_data: pd.DataFrame,
                     events: Optional[List[str]] = None,
                     drivers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of lap telemetry data.

        This method provides a complete analysis suite including:
        - Data preprocessing and resampling
        - Speed profile analysis with confidence intervals
        - Distance matrix calculations between drivers
        - PCA analysis of telemetry features
        - Statistical summaries of lap times and telemetry features
        - Visualization of all analyses

        Args:
            telemetry_data: Raw telemetry data to analyze
            events: Optional list of events to include in analysis
            drivers: Optional list of drivers to include in analysis

        Returns:
            Dict[str, Any]: Dictionary containing:
                - processed_data: Resampled and processed telemetry data
                - lap_metrics: Calculated metrics for each lap
                - distance_matrices: Driver similarity matrices
                - pca_transformed: PCA-transformed telemetry data
                - pca_explained_variance: Explained variance ratios for PCA
                - statistics: Summary statistics for drivers and laps
                - figure: Matplotlib figure containing all visualizations
        """
        processed_data, lap_metrics = self.preprocess_lap_data(
            telemetry_data, events, drivers)
        distance_matrices = self.calculate_distance_matrix(
            processed_data, "Speed", by_event=True)

        fig = plt.figure(figsize=(20, 12))

        plt.subplot(221)
        for event in processed_data["Event"].unique():
            event_data = processed_data[processed_data["Event"] == event]
            for driver in processed_data["Driver"].unique():
                driver_data = event_data[event_data["Driver"] == driver]
                grouped = driver_data.groupby("normalized_time")
                mean_speed = grouped["Speed"].mean()
                std_speed = grouped["Speed"].std()

                plt.plot(driver_data["normalized_time"].unique(),
                         mean_speed, label=driver)
                plt.fill_between(driver_data["normalized_time"].unique(),
                                 mean_speed - std_speed,
                                 mean_speed + std_speed,
                                 alpha=0.2)

        plt.xlabel("Normalized Lap Time")
        plt.ylabel("Speed")
        plt.title("Speed Profiles with Confidence Intervals")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(processed_data[self.feature_cols])
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        processed_data["PC1"] = X_pca[:, 0]
        processed_data["PC2"] = X_pca[:, 1]

        plt.subplot(222)
        for event in processed_data["Event"].unique():
            event_data = processed_data[processed_data["Event"] == event]
            for driver in event_data["Driver"].unique():
                mask = (processed_data["Event"] == event) & (
                    processed_data["Driver"] == driver)
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                            label=f"{event} - {driver}",
                            alpha=0.6)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Analysis by Event and Driver")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.subplot(223)
        sns.boxplot(data=lap_metrics, x="event", y="duration", hue="driver")
        plt.xticks(rotation=45)
        plt.title("Lap Time Distributions by Event and Driver")
        plt.tight_layout()

        plt.subplot(224)
        feature_importance = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": np.abs(pca.components_[0])
        }).sort_values("importance", ascending=True)
        sns.barplot(data=feature_importance, x="importance", y="feature")
        plt.title("Feature Importance (PC1)")
        plt.tight_layout()

        stats = {
            "driver_stats": processed_data.groupby(["Event", "Driver"]).agg({
                "Speed": ["mean", "std", "max", "min"],
                "Throttle": ["mean", "std"],
                "Brake": ["mean", "std"],
                "nGear": ["mean", "std"],
            }).round(2),
            "lap_time_stats": lap_metrics.groupby(["event", "driver"]).agg({
                "duration": ["mean", "std", "min", "max"],
                "max_speed": ["mean", "max"],
                "avg_speed": ["mean"],
                "brake_applications": ["mean"],
                "drs_zones": ["mean"]
            }).round(2)
        }

        return {
            "processed_data": processed_data,
            "lap_metrics": lap_metrics,
            "distance_matrices": distance_matrices,
            "pca_transformed": X_pca,
            "pca_explained_variance": pca.explained_variance_ratio_,
            "statistics": stats,
            "figure": fig
        }
