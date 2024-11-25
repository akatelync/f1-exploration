from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import resample
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


driver_colors = {
    "VER": "#3671c6",  # Blue
    "NOR": "#ff8000",  # Orange
    "LEC": "#e8002d",
    "PIA": "#ff8000",   # Red
}

driver_markers = {
    "LEC": "o",  # circle
    "VER": "x",  # square
    "NOR": "*",  # triangle up
    "PIA": "D"   # diamond
}


@dataclass
class F1TelemetryAnalyzer:
    """
    A comprehensive class for analyzing Formula 1 telemetry data, including preprocessing,
    statistical analysis, PCA, and visualization capabilities.

    Attributes:
        n_samples (int): Number of samples to resample each lap's telemetry data to
        feature_cols (List[str]): List of telemetry features to analyze
        driver_colors (Dict[str, str]): Mapping of driver codes to their colors
        variance_threshold (float): Threshold for cumulative explained variance in PCA
    """
    n_samples: int = 300
    feature_cols: List[str] = None
    variance_threshold: float = 0.95
    show_plots: bool = True

    def __post_init__(self):
        """Initialize default values after dataclass initialization"""
        if self.feature_cols is None:
            self.feature_cols = ["RPM", "Speed",
                                 "nGear", "Throttle", "Brake", "DRS"]
        self.scaler = StandardScaler()
        self.pca_model = None

    def filter_data(self,
                    telemetry_data: pd.DataFrame,
                    rounds: Optional[List[str]] = None,
                    drivers: Optional[List[str]] = None) -> pd.DataFrame:
        """Filter telemetry data by rounds and/or drivers."""
        filtered_data = telemetry_data.copy()

        if rounds:
            filtered_data = filtered_data[filtered_data["Round"].isin(rounds)]
        if drivers:
            filtered_data = filtered_data[filtered_data["Driver"].isin(
                drivers)]

        return filtered_data

    def preprocess_lap_data(self,
                            telemetry_data: pd.DataFrame,
                            rounds: Optional[List[int]] = None,
                            drivers: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess telemetry data by aligning and resampling lap data."""
        filtered_data = self.filter_data(telemetry_data, rounds, drivers)

        grouped = filtered_data.groupby(["Round", "Driver", "LapNumber"])
        processed_laps = []
        lap_metrics = []

        for (round_name, driver, lap), lap_data in grouped:
            lap_data = lap_data.sort_values("Time")
            lap_duration = (lap_data["Time"].max() -
                            lap_data["Time"].min()).total_seconds()

            lap_data["normalized_time"] = np.linspace(0, 1, self.n_samples)
            processed_laps.append(lap_data)

            metrics = {
                "round": round_name,
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
                                  by_round: bool = True) -> Dict[str, pd.DataFrame]:
        """Calculate Euclidean distances between average lap profiles for each driver."""
        distance_matrices = {}

        if by_round:
            rounds = processed_data["Round"].unique()
            for round_name in rounds:
                round_data = processed_data[processed_data["Round"]
                                            == round_name]
                distance_matrices[round_name] = self._calculate_single_distance_matrix(
                    round_data, feature)
        else:
            distance_matrices["all"] = self._calculate_single_distance_matrix(
                processed_data, feature)

        return distance_matrices

    def _calculate_single_distance_matrix(self,
                                          data: pd.DataFrame,
                                          feature: str) -> pd.DataFrame:
        """Helper method to calculate distance matrix for a single dataset."""
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

    def determine_optimal_components(self,
                                     X_scaled: np.ndarray) -> Tuple[int, PCA]:
        """Determines optimal number of PCA components using scree plot."""
        pca_all = PCA()
        pca_all.fit(X_scaled)

        if self.show_plots:
            plt.figure(figsize=(12, 6), dpi=150)
            cum_var_ratio = np.cumsum(pca_all.explained_variance_ratio_)
            plt.plot(range(1, len(cum_var_ratio) + 1), cum_var_ratio, 'bo-')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance Ratio')
            plt.title(
                'Scree Plot: Cumulative Explained Variance vs. Number of Components')
            plt.grid(True)
            plt.show()

        n_components = np.argmax(
            np.cumsum(pca_all.explained_variance_ratio_) >= self.variance_threshold) + 1
        print(
            f"Number of components needed for {self.variance_threshold*100}% variance: {n_components}")

        return n_components, pca_all

    def analyze_drivers_pca(self,
                            reshaped_data: pd.DataFrame,
                            selected_drivers: List[str],
                            session_type: str) -> Dict[str, Any]:
        """Perform PCA analysis on selected drivers with dynamic component selection."""
        # Filter data for selected drivers
        X = reshaped_data.loc[(slice(None), selected_drivers), :]

        # Scale data
        X_scaled = self.scaler.fit_transform(X)

        # Get optimal components
        n_components, _ = self.determine_optimal_components(X_scaled)
        self.pca_model = PCA(n_components=n_components)
        X_pca = self.pca_model.fit_transform(X_scaled)

        # Create PCA DataFrame
        pca_cols = [f"PC{i+1}" for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        pca_df["Driver"] = X.index.get_level_values("Driver")

        # Plot PCA results
        self._plot_pca_results(pca_df, selected_drivers, session_type)

        # Calculate and plot centroids
        centroids, spreads = self._calculate_centroids_and_spreads(
            pca_df, pca_cols)
        self._plot_centroids_with_spreads(
            centroids, spreads, selected_drivers, session_type)

        # Print analysis results
        self._print_analysis_results(centroids, spreads, selected_drivers)

        return {
            "pca": self.pca_model,
            "pca_df": pca_df,
            "centroids": centroids,
            "spreads": spreads
        }

    def _plot_pca_results(self,
                          pca_df: pd.DataFrame,
                          selected_drivers: List[str],
                          session_type: str):
        """Helper method to plot PCA results."""
        if not self.show_plots:
            return

        plt.figure(figsize=(10, 6), dpi=150)
        for driver in selected_drivers:
            driver_data = pca_df[pca_df["Driver"] == driver]
            plt.scatter(driver_data["PC1"], driver_data["PC2"],
                        c=driver_colors[driver],
                        marker=driver_markers[driver],
                        label=driver, s=100, alpha=0.6)

        # plt.title(
        #     f"{session_type} Driver Comparison: {', '.join(selected_drivers)}")
        plt.xlabel(
            f"PC1 ({self.pca_model.explained_variance_ratio_[0]:.1%} variance)", fontsize=14)
        plt.ylabel(
            f"PC2 ({self.pca_model.explained_variance_ratio_[1]:.1%} variance)", fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=13)
        plt.tight_layout()
        plt.savefig(
            f"imgs/{session_type} Driver Comparison: {', '.join(selected_drivers)}.png")
        plt.show()

    def _calculate_centroids_and_spreads(self,
                                         pca_df: pd.DataFrame,
                                         pca_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate centroids and spreads for PCA results."""
        centroids = pca_df.groupby("Driver")[pca_cols].mean()
        spreads = pca_df.groupby("Driver")[pca_cols].std()
        return centroids, spreads

    def _plot_centroids_with_spreads(self,
                                     centroids: pd.DataFrame,
                                     spreads: pd.DataFrame,
                                     selected_drivers: List[str],
                                     session_type: str):
        """Plot centroids with error bars."""
        if not self.show_plots:
            return

        plt.figure(figsize=(10, 6), dpi=150)
        for driver in selected_drivers:
            plt.errorbar(
                centroids.loc[driver, "PC1"],
                centroids.loc[driver, "PC2"],
                xerr=spreads.loc[driver, "PC1"],
                yerr=spreads.loc[driver, "PC2"],
                label=driver,
                c=driver_colors[driver],
                marker=driver_markers[driver],
                fmt="o",
                capsize=5,
                capthick=2,
                markersize=10,
                alpha=0.6
            )

        # plt.title(
        #     f"{session_type} Driving Styles Comparison: {', '.join(selected_drivers)}")
        plt.xlabel(
            f"PC1 ({self.pca_model.explained_variance_ratio_[0]:.1%} variance)", fontsize=14)
        plt.ylabel(
            f"PC2 ({self.pca_model.explained_variance_ratio_[1]:.1%} variance)", fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=13)
        plt.tight_layout()
        plt.savefig(
            f"imgs/{session_type} Driving Styles Comparison: {', '.join(selected_drivers)}.png")
        plt.show()

    def _print_analysis_results(self,
                                centroids: pd.DataFrame,
                                spreads: pd.DataFrame,
                                selected_drivers: List[str]):
        """Print analysis results including centroids and pairwise distances."""
        print("\nDriver Centroids (average position in PCA space):")
        print(centroids)
        print("\nDriver Variability (spread in PCA space):")
        print(spreads)

        print("\nPairwise Distances between Drivers:")
        for i, driver1 in enumerate(selected_drivers):
            for driver2 in selected_drivers[i+1:]:
                distance = np.linalg.norm(
                    centroids.loc[driver1] - centroids.loc[driver2]
                )
                print(f"{driver1} vs {driver2}: {distance:.2f}")

    def _reshape_telemetry_data(self, processed_laps: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape telemetry data from long format to wide format where each lap's metrics
        are spread across columns with sequential numbering.

        Args:
            processed_laps: DataFrame containing processed lap telemetry data

        Returns:
            pd.DataFrame: Reshaped DataFrame where each row represents a unique lap and
                        columns are named as {metric}_{index}
        """
        grouped_data = []

        for (round_num, driver, lap_num), lap_data in processed_laps.groupby(["Round", "Driver", "LapNumber"]):
            reshaped_data = {
                "Round": round_num,
                "Driver": driver,
                "LapNumber": lap_num
            }

            for metric in self.feature_cols:
                for idx, value in enumerate(lap_data[metric]):
                    reshaped_data[f"{metric}_{idx}"] = value

            grouped_data.append(reshaped_data)

        reshaped_df = pd.DataFrame(grouped_data)
        reshaped_df.set_index(["Round", "Driver", "LapNumber"], inplace=True)

        return reshaped_df

    def analyze_laps(self,
                     telemetry_data: pd.DataFrame,
                     session_type: str,
                     rounds: Optional[List[str]] = None,
                     drivers: Optional[List[str]] = None,
                     pca: Optional[bool] = None) -> Dict[str, Any]:
        """Perform comprehensive analysis of lap telemetry data."""
        # Process telemetry data
        processed_data, lap_metrics = self.preprocess_lap_data(
            telemetry_data, rounds, drivers)

        # Reshape telemetry data
        reshaped_data = self._reshape_telemetry_data(processed_data)

        # Calculate distance matrices
        distance_matrices = self.calculate_distance_matrix(
            processed_data, "Speed", by_round=True)

        # Perform PCA analysis if drivers are specified
        pca_results = None
        if pca:
            pca_results = self.analyze_drivers_pca(
                reshaped_data, drivers, session_type)

        return {
            "processed_data": processed_data,
            "lap_metrics": lap_metrics,
            "distance_matrices": distance_matrices,
            "pca_results": pca_results
        }
