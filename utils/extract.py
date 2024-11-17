import fastf1

import pandas as pd
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import Union, List, Dict, Optional
from scipy.interpolate import interp1d


@dataclass
class F1DataProcessor:
    cache_dir: Union[str, Path]
    year: int
    numeric_columns = ["RPM", "Speed", "nGear",
                       "Throttle", "Brake", "CumulativeDistance"]

    def __post_init__(self):
        fastf1.Cache.enable_cache(self.cache_dir)
        self.schedule = fastf1.get_event_schedule(self.year)

    def get_quali_session(self, rounds: Optional[List[int]] = None,
                          include_sprint_quali: bool = False) -> pd.DataFrame:
        if rounds is None:
            rounds = self.schedule["RoundNumber"].tolist()

        for round_num in rounds:
            quali_data, lap_data = self._process_quali_session(round_num, "Q")
            return quali_data, lap_data

            if include_sprint_quali:
                sprint_quali_data, lap_data = self._process_quali_session(
                    round_num, "SQ")
                sprint_quali_data["SessionType"] = "SprintQuali"
                return sprint_quali_data, lap_data

    def _process_quali_session(self, round_num: int, session_type: str) -> Optional[pd.DataFrame]:
        cols_to_keep = ["Event", "Driver", "DriverNumber", "Team", "LapTime", "LapNumber", "Stint", "Sector1Time", "Sector2Time",
                        "Sector3Time", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "IsPersonalBest", "Compound", "TyreLife",
                        "FreshTyre", "TrackStatus", "Deleted", "DeletedReason", "LapStartDate", "LapEndDate"]

        try:
            session = fastf1.get_session(self.year, round_num, session_type)
            session.load()

            session_df = session.laps
            session_df = session_df[(session_df["PitInTime"].isna()) & (
                session_df["PitOutTime"].isna())]
            session_df["LapEndDate"] = session_df["LapStartDate"] + \
                session_df["LapTime"]

            laps_df = session_df.copy()
            laps_df["Event"] = f"{self.year} {round_num}"
            laps_df = laps_df[cols_to_keep]

            race_control_df = session.race_control_messages
            quali_sessions = race_control_df[(race_control_df.Message.str.contains("GREEN LIGHT")) |
                                             (race_control_df.Message.str.contains("CHEQUERED FLAG")) |
                                             (race_control_df.Message.str.contains("WILL NOT BE RESUMED"))]

            quali_sessions["GreenLightCount"] = (quali_sessions["Message"].str.contains("GREEN LIGHT")
                                                 .groupby((quali_sessions["Message"] != quali_sessions["Message"].shift()).cumsum())
                                                 .cumsum())
            # first instance of green light means start of session
            quali_sessions = quali_sessions[quali_sessions["GreenLightCount"] == 1]

            quali_sessions["QualiSession"] = [
                "Q" + str(i) for i in range(1, len(quali_sessions) + 1)]
            quali_sessions = quali_sessions[["Time", "QualiSession"]]

            q1_mask = (laps_df["LapStartDate"] >= quali_sessions.loc[quali_sessions["QualiSession"] == "Q1", "Time"].iloc[0]) & \
                (laps_df["LapStartDate"] <
                 quali_sessions.loc[quali_sessions["QualiSession"] == "Q2", "Time"].iloc[0])

            q2_mask = (laps_df["LapStartDate"] >= quali_sessions.loc[quali_sessions["QualiSession"] == "Q2", "Time"].iloc[0]) & \
                (laps_df["LapStartDate"] <
                 quali_sessions.loc[quali_sessions["QualiSession"] == "Q3", "Time"].iloc[0])

            q3_mask = laps_df["LapStartDate"] >= quali_sessions.loc[quali_sessions["QualiSession"]
                                                                    == "Q3", "Time"].iloc[0]

            laps_df.loc[q1_mask, "QualiSession"] = 1
            laps_df.loc[q2_mask, "QualiSession"] = 2
            laps_df.loc[q3_mask, "QualiSession"] = 3

            quali_df = self._rank_quali_laps(laps_df)
            for i in ["Sector1Time", "Sector2Time", "Sector3Time"]:
                quali_df[i] = quali_df[i].dt.total_seconds()

            telemetry_dfs = []
            dfs = []
            drivers = session_df["Driver"].unique()
            for driver in drivers:
                car_df = session_df[session_df["Driver"] ==
                                    driver].get_car_data().add_distance()
                lap_df = quali_df[quali_df["Driver"]
                                  == driver].reset_index(drop=True)

                pivot_telemetry_df = self._process_quali_telemetry(
                    car_df, lap_df)

                pivot_telemetry_df["Driver"] = driver

                # lap_telemetry_df = lap_df.join(pivot_telemetry_df)
                # telemetry_dfs.append(lap_telemetry_df)
                dfs.append(pivot_telemetry_df)

            return quali_df, pd.concat(dfs)

        except Exception as e:
            print(f"Error processing qualifying round {round_num}: {str(e)}")
            return None

    def _rank_quali_laps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ranks lap times by speed within each QualiSession.
        The fastest lap gets rank 1, updating as new laps are set.

        Parameters:
        df (pandas.DataFrame): DataFrame containing qualifying data with LapTime and QualiSession columns

        Returns:
        pandas.DataFrame: Original DataFrame with new 'LapRank' and 'CurrentFastestLap' columns
        """

        # Sort by QualiSession and LapStartDate to maintain chronological order
        df = df.sort_values(["Event", "QualiSession", "LapStartDate"])

        def update_rankings(group):
            # Create a running ranking based on laptimes seen so far
            group = group.copy()
            for idx in range(len(group)):
                current_slice = group.iloc[:idx + 1]
                # Rank the laptimes
                current_ranks = current_slice["LapTime"].rank(
                    method="min", ascending=True)
                group.iloc[idx, group.columns.get_loc(
                    "LapRank")] = current_ranks.iloc[-1]

                # Store the current fastest lap time for reference
                group.iloc[idx, group.columns.get_loc("CurrentFastestLap")] = \
                    current_slice["LapTime"].min()

            return group

        # Initialize new columns
        df["LapRank"] = float("nan")
        df["CurrentFastestLap"] = pd.Timedelta(0)

        # Apply the ranking function to each QualiSession group
        df = df.groupby("QualiSession", group_keys=False).apply(
            update_rankings)

        # Calculate the delta between CurrentFastestLap and LapTime
        df["DeltaFastestLap"] = (
            df["LapTime"] - df["CurrentFastestLap"]).dt.total_seconds()

        return df

    def _process_quali_telemetry(self, car_df: pd.DataFrame, lap_df: pd.DataFrame) -> pd.DataFrame:
        car_df["Brake"] = car_df["Brake"].astype(int)

        lap_starts = lap_df["LapStartDate"].values
        lap_ends = lap_df["LapEndDate"].values

        car_df["Lap"] = 0
        for i, (lap_start, lap_end) in enumerate(zip(lap_starts, lap_ends), 1):
            car_df.loc[(car_df["Date"] >= lap_start) & (
                car_df["Date"] <= lap_end), "Lap"] = i

        car_df = car_df[car_df["Lap"] != 0]

        grouped = car_df.groupby("Lap")
        result_df = grouped.apply(lambda x: x.assign(CumulativeDistance=x["Distance"].sub(
            x["Distance"].iloc[0]).cumsum())).reset_index(level=0, drop=True)
        result_df["CumulativeDistance"].fillna(0, inplace=True)

        # normalized_df = self._normalize_telemetry_data(result_df)

        # pivoted_dfs = []
        # grouped = normalized_df.groupby("Lap")
        # for lap, lap_data in grouped:
        #     pivoted_data = {}

        #     for col in self.numeric_columns:
        #         series_values = lap_data[col].values
        #         col_names = [f"{col}_{i}" for i in range(len(series_values))]
        #         pivoted_data.update(dict(zip(col_names, series_values)))

        #     pivoted_dfs.append(pd.DataFrame([pivoted_data]))

        # final_pivoted_df = pd.concat(
        #     pivoted_dfs, axis=0).reset_index(drop=True)

        # return final_pivoted_df
        return result_df

    def _normalize_telemetry_data(self, df: pd.DataFrame, target_points: int = 300) -> pd.DataFrame:
        laps = df["Lap"].unique()

        normalized_data = []

        for lap in laps:
            lap_data = df[df["Lap"] == lap].copy()

            if len(lap_data) < 2:
                continue

            original_points = np.linspace(0, 100, len(lap_data))
            new_points = np.linspace(0, 100, target_points)

            normalized_lap = {}

            for col in self.numeric_columns:
                interpolator = interp1d(
                    original_points,
                    lap_data[col].values,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate"
                )

                normalized_lap[col] = interpolator(new_points)

            normalized_df = pd.DataFrame(normalized_lap)
            normalized_df["Lap"] = lap
            normalized_data.append(normalized_df)

        result = pd.concat(normalized_data, ignore_index=True)

        return result
