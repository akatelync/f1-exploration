import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import resample
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class F1TelemetryAnalyzer:
    n_samples: int = 300
    feature_cols = ["RPM", "Speed", "nGear", "Throttle", "Brake", "DRS"]

    def preprocess_lap_data(self, telemetry_data, lap_column="Lap", time_column="Time"):
        """
        Preprocess telemetry data by aligning and resampling lap data to fixed length
        """
        grouped = telemetry_data.groupby(["Driver", lap_column])
        processed_laps = []
        lap_metrics = []

        for (driver, lap), lap_data in grouped:
            lap_data = lap_data.sort_values(time_column)
            lap_duration = (lap_data[time_column].max(
            ) - lap_data[time_column].min()).total_seconds()
            resampled_data = {}

            for feature in self.feature_cols:
                if feature in lap_data.columns:
                    resampled = resample(lap_data[feature], self.n_samples)
                    resampled_data[feature] = resampled

            resampled_df = pd.DataFrame(resampled_data)
            resampled_df["Driver"] = driver
            resampled_df["Lap"] = lap
            resampled_df["normalized_time"] = np.linspace(0, 1, self.n_samples)
            processed_laps.append(resampled_df)

            metrics = {
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

    def calculate_distance_matrix(self, processed_data, feature="Speed"):
        """
        Calculate Euclidean distances between average lap profiles for each driver
        """
        drivers = processed_data["Driver"].unique()
        distance_matrix = np.zeros((len(drivers), len(drivers)))
        driver_profiles = {}

        for driver in drivers:
            driver_data = processed_data[processed_data["Driver"] == driver]
            profile_matrix = np.array(driver_data.groupby("Lap")[
                                      feature].apply(list).tolist())
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

    def analyze_laps(self, telemetry_data):
        """
        Perform comprehensive analysis of lap telemetry data
        """
        processed_data, lap_metrics = self.preprocess_lap_data(telemetry_data)
        distance_matrix = self.calculate_distance_matrix(
            processed_data, "Speed")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(processed_data[self.feature_cols])

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        fig = plt.figure(figsize=(20, 12))

        plt.subplot(231)
        for driver in processed_data["Driver"].unique():
            driver_data = processed_data[processed_data["Driver"] == driver]
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
        plt.legend()

        plt.subplot(232)
        sns.heatmap(distance_matrix, cmap="YlOrRd")
        plt.title("Profile Distances Between Drivers")

        plt.subplot(233)
        for driver in processed_data["Driver"].unique():
            mask = processed_data["Driver"] == driver
            plt.plot(X_pca[mask, 0], X_pca[mask, 1], label=driver, alpha=0.6)
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.2, s=10)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Trajectories")
        plt.legend()

        plt.subplot(234)
        for driver in processed_data["Driver"].unique():
            driver_data = processed_data[processed_data["Driver"] == driver]
            sns.kdeplot(data=driver_data, x="Throttle",
                        y="Brake", label=driver, alpha=0.5)
        plt.xlabel("Throttle")
        plt.ylabel("Brake")
        plt.title("Throttle vs Brake Density")
        plt.legend()

        plt.subplot(235)
        sns.violinplot(data=lap_metrics, x="driver", y="duration")
        plt.title("Lap Time Distributions")
        plt.xticks(rotation=45)

        plt.subplot(236)
        feature_importance = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": np.abs(pca.components_[0])
        }).sort_values("importance", ascending=True)

        sns.barplot(data=feature_importance, x="importance", y="feature")
        plt.title("Feature Importance (PC1)")

        plt.tight_layout()

        stats = {
            "driver_stats": processed_data.groupby("Driver").agg({
                "Speed": ["mean", "std", "max", "min"],
                "Throttle": ["mean", "std"],
                "Brake": ["mean", "std"],
                "nGear": ["mean", "std"],
            }).round(2)
        }

        return {
            "processed_data": processed_data,
            "lap_metrics": lap_metrics,
            "distance_matrix": distance_matrix,
            "pca_transformed": X_pca,
            "pca_explained_variance": pca.explained_variance_ratio_,
            "statistics": stats,
            "figure": fig
        }
