from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from core.extract import F1DataProcessor
from core.telemetry import F1TelemetryAnalyzer


@dataclass
class F1DriverPerformanceIndex:
    """
    A comprehensive class for analyzing Formula 1 driver performance by combining
    telemetry analysis, lap metrics, and PCA-based driving style analysis.

    Attributes:
        data_processor: F1DataProcessor instance for raw data processing
        telemetry_analyzer: F1TelemetryAnalyzer instance for detailed analysis
        session_type: Type of session to analyze ("race" or "qualifying")
        feature_weights: Dictionary of feature weights for index calculation
    """
    data_processor: F1DataProcessor
    telemetry_analyzer: F1TelemetryAnalyzer
    session_type: str
    feature_weights: Dict[str, float] = None

    def __post_init__(self):
        """Initialize default feature weights if none provided"""
        if self.feature_weights is None:
            # Simplified weight keys to match metric names directly
            self.feature_weights = {
                "consistency": 0.25,
                "style": 0.25,
                "technical": 0.25,
                "pace": 0.25
            }

        self.scaler = MinMaxScaler()

    def calculate_performance_index(
        self,
        rounds: List[int],
        drivers: List[str]
    ) -> pd.DataFrame:
        """
        Calculate comprehensive performance index for specified drivers across rounds.

        Args:
            rounds: List of round numbers to analyze
            drivers: List of driver codes to analyze

        Returns:
            DataFrame containing performance metrics and final index
        """
        # Get session data based on type
        if self.session_type.lower() == "race":
            session_data, telemetry_data = self.data_processor.get_race_session(
                rounds=rounds,
                drivers=drivers,
                normalize_telemetry=True
            )
        else:
            session_data, telemetry_data = self.data_processor.get_quali_session(
                rounds=rounds,
                drivers=drivers,
                normalize_telemetry=True
            )

        # Analyze telemetry data
        analysis_results = self.telemetry_analyzer.analyze_laps(
            telemetry_data=telemetry_data,
            session_type=self.session_type,
            rounds=rounds,
            drivers=drivers,
            pca=True
        )

        # Calculate individual performance metrics
        consistency_scores = self._calculate_consistency_scores(session_data)
        style_scores = self._calculate_driving_style_scores(analysis_results)
        technical_scores = self._calculate_technical_scores(
            analysis_results["processed_data"],
            analysis_results["lap_metrics"]
        )
        pace_scores = self._calculate_pace_scores(session_data)

        # Combine metrics into final index
        performance_index = self._compute_final_index(
            consistency_scores,
            style_scores,
            technical_scores,
            pace_scores
        )

        return performance_index

    def _calculate_consistency_scores(self, session_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate lap time consistency scores for each driver"""
        consistency_metrics = []

        for driver in session_data["Driver"].unique():
            driver_laps = session_data[session_data["Driver"] == driver]

            metrics = {
                "Driver": driver,
                "lap_time_std": driver_laps["LapTime"].dt.total_seconds().std(),
                "sector_consistency": np.mean([
                    driver_laps["Sector1Time"].std(),
                    driver_laps["Sector2Time"].std(),
                    driver_laps["Sector3Time"].std()
                ])
            }
            consistency_metrics.append(metrics)

        consistency_df = pd.DataFrame(consistency_metrics)

        # Scale metrics (lower is better)
        consistency_df["consistency_score"] = 1 - self.scaler.fit_transform(
            consistency_df[["lap_time_std", "sector_consistency"]]
        ).mean(axis=1)

        return consistency_df

    def _calculate_driving_style_scores(self, analysis_results: Dict) -> pd.DataFrame:
        """Calculate driving style distinctiveness scores using PCA results"""
        pca_results = analysis_results["pca_results"]
        style_metrics = []

        for driver in pca_results["centroids"].index:
            # Calculate distance from mean driving style
            centroid = pca_results["centroids"].loc[driver]
            mean_centroid = pca_results["centroids"].mean()
            style_distinctiveness = np.linalg.norm(centroid - mean_centroid)

            # Calculate consistency of driving style
            spread = pca_results["spreads"].loc[driver]
            style_consistency = 1 / np.mean(spread)

            metrics = {
                "Driver": driver,
                "style_distinctiveness": style_distinctiveness,
                "style_consistency": style_consistency
            }
            style_metrics.append(metrics)

        style_df = pd.DataFrame(style_metrics)

        # Scale metrics
        style_df["style_score"] = self.scaler.fit_transform(
            style_df[["style_distinctiveness", "style_consistency"]]
        ).mean(axis=1)

        return style_df

    def _calculate_technical_scores(
        self,
        processed_data: pd.DataFrame,
        lap_metrics: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate technical execution scores based on telemetry data"""
        technical_metrics = []

        for driver in processed_data["Driver"].unique():
            driver_data = processed_data[processed_data["Driver"] == driver]
            driver_metrics = lap_metrics[lap_metrics["driver"] == driver]

            # Calculate various technical metrics
            metrics = {
                "Driver": driver,
                "throttle_control": driver_data["Throttle"].std(),
                "brake_efficiency": (
                    driver_metrics["brake_applications"].mean() /
                    driver_metrics["duration"].mean()
                ),
                "gear_changes": driver_data["nGear"].diff().abs().mean(),
                "drs_usage": driver_metrics["drs_zones"].mean()
            }
            technical_metrics.append(metrics)

        technical_df = pd.DataFrame(technical_metrics)

        # Scale metrics
        technical_df["technical_score"] = self.scaler.fit_transform(
            technical_df[[
                "throttle_control",
                "brake_efficiency",
                "gear_changes",
                "drs_usage"
            ]]
        ).mean(axis=1)

        return technical_df

    def _calculate_pace_scores(self, session_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate overall pace performance scores"""
        pace_metrics = []

        for driver in session_data["Driver"].unique():
            driver_laps = session_data[session_data["Driver"] == driver]
            driver_laps["LapTime"] = driver_laps["LapTime"].dt.total_seconds()

            if self.session_type.lower() == "qualifying":
                # For qualifying, focus on best lap times
                best_lap = driver_laps["LapTime"].min()
                relative_pace = best_lap / driver_laps["LapTime"].min().mean()
            else:
                # For race, consider average lap times and position
                avg_lap = driver_laps["LapTime"].mean()
                relative_pace = avg_lap / driver_laps["LapTime"].mean().mean()

            metrics = {
                "Driver": driver,
                "relative_pace": relative_pace,
                "position": driver_laps["Position"].iloc[0]
            }
            pace_metrics.append(metrics)

        pace_df = pd.DataFrame(pace_metrics)

        # Scale metrics (lower is better for both)
        pace_df["pace_score"] = 1 - self.scaler.fit_transform(
            pace_df[["relative_pace", "position"]]
        ).mean(axis=1)

        return pace_df

    def _compute_final_index(
        self,
        consistency_scores: pd.DataFrame,
        style_scores: pd.DataFrame,
        technical_scores: pd.DataFrame,
        pace_scores: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute final performance index by combining all metrics"""
        final_index = pd.DataFrame()
        final_index["Driver"] = consistency_scores["Driver"]

        # Add individual component scores
        final_index["Consistency"] = consistency_scores["consistency_score"]
        final_index["Style"] = style_scores["style_score"]
        final_index["Technical"] = technical_scores["technical_score"]
        final_index["Pace"] = pace_scores["pace_score"]

        # Calculate weighted performance index using simplified weight keys
        final_index["PerformanceIndex"] = (
            final_index["Consistency"] * self.feature_weights["consistency"] +
            final_index["Style"] * self.feature_weights["style"] +
            final_index["Technical"] * self.feature_weights["technical"] +
            final_index["Pace"] * self.feature_weights["pace"]
        )

        return final_index.sort_values("PerformanceIndex", ascending=False)

    def plot_performance_breakdown(self, performance_index: pd.DataFrame) -> None:
        """Create visualization of performance index breakdown for each driver"""
        plt.figure(figsize=(12, 6))

        metrics = ["Consistency", "Style", "Technical", "Pace"]
        bottom = np.zeros(len(performance_index))

        for metric in metrics:
            # Simplified weight key lookup
            weight = self.feature_weights[metric.lower()]

            plt.bar(
                performance_index["Driver"],
                performance_index[metric] * weight,
                bottom=bottom,
                label=metric
            )
            bottom += performance_index[metric] * weight

        plt.title(f"{self.session_type} Performance Index Breakdown")
        plt.xlabel("Driver", fontsize=14)
        plt.ylabel("Performance Index", fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=14)
        plt.tight_layout()
        # plt.savefig(f"imgs/{self.session_type} Performance Index Breakdown")
        plt.show()

        return performance_index
