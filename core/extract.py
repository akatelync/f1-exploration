from scipy.interpolate import interp1d
from typing import Union, List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import fastf1
import logging
import warnings
warnings.filterwarnings("ignore")


@dataclass
class F1DataProcessor:
    """
    A class for processing Formula 1 race and qualifying session data using the FastF1 API.

    This class provides functionality to fetch, process, and analyze telemetry data
    from Formula 1 sessions, including both qualifying and race sessions.

    Attributes:
        cache_dir (Union[str, Path]): Directory path for caching FastF1 data
        year (int): The F1 season year to process
        numeric_columns (List[str]): List of telemetry columns that contain numeric data
    """

    cache_dir: Union[str, Path]
    year: int
    numeric_columns = ["RPM", "Speed", "nGear", "DRS",
                       "Throttle", "Brake", "CumulativeDistance"]

    def __post_init__(self):
        """
        Initialize the FastF1 cache and load the season schedule quietly.
        Called automatically after class instantiation.
        """
        logging.getLogger("fastf1").setLevel(logging.WARNING)
        fastf1.Cache.enable_cache(self.cache_dir)
        self.schedule = fastf1.get_event_schedule(self.year)

    def get_quali_session(self, rounds: Union[List[int], int], drivers: Optional[List[str]],
                          normalize_telemetry: bool = False, target_points: int = 300) -> pd.DataFrame:
        """
        Retrieve and process qualifying session data for specified rounds.

        Args:
            rounds (Optional[List[int]]): List of round numbers to process. If None, processes all rounds.
            include_sprint_quali (bool): Flag to include sprint qualifying sessions (not implemented).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing qualifying session data and associated telemetry data.
        """
        if rounds is None:
            rounds = self.schedule["RoundNumber"].tolist()

        all_quali_data = []
        all_lap_data = []

        for round_num in rounds:
            quali_data, lap_data = self._process_quali_session(
                round_num,
                "Q",
                normalize_telemetry=normalize_telemetry,
                target_points=target_points
            )

            if quali_data is not None:
                if drivers:
                    lap_data = lap_data[lap_data["Driver"].isin(drivers)]
                    quali_data = quali_data[quali_data["Driver"].isin(drivers)]

                lap_data["Round"] = round_num
                all_quali_data.append(quali_data)
                all_lap_data.append(lap_data)

        return pd.concat(all_quali_data), pd.concat(all_lap_data)

    def get_race_session(self, rounds: Union[List[int], int], drivers: Optional[List[str]],
                         normalize_telemetry: bool = False, target_points: int = 300) -> pd.DataFrame:
        """
        Retrieve and process race session data for specified rounds.

        Args:
            rounds (Optional[List[int]]): List of round numbers to process. If None, processes all rounds.
            include_sprint_quali (bool): Flag to include sprint race sessions (not implemented).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing race session data and associated telemetry data.
        """
        if rounds is None:
            rounds = self.schedule["RoundNumber"].tolist()

        all_race_data = []
        all_lap_data = []

        for round_num in rounds:
            race_data, lap_data = self._process_race_session(
                round_num,
                "R",
                normalize_telemetry=normalize_telemetry,
                target_points=target_points
            )

            if race_data is not None:
                if drivers:
                    lap_data = lap_data[lap_data["Driver"].isin(drivers)]
                    race_data = race_data[race_data["Driver"].isin(drivers)]

                lap_data["Round"] = round_num
                all_race_data.append(race_data)
                all_lap_data.append(lap_data)

        return pd.concat(all_race_data), pd.concat(all_lap_data)

    def _process_quali_session(self, round_num: int, session_type: str = "Q", normalize_telemetry: bool = False, target_points: int = 300) -> Optional[pd.DataFrame]:
        """
        Process a single qualifying session's data.

        Args:
            round_num (int): Round number of the session to process
            session_type (str): Type of session ('Q' for qualifying)

        Returns:
            Optional[Tuple[pd.DataFrame, pd.DataFrame]]: Processed qualifying data and telemetry data,
                                                       or None if processing fails
        """
        cols_to_keep = ["Round", "Driver", "DriverNumber", "Team", "LapTime", "LapNumber", "Stint", "Sector1Time", "Sector2Time",
                        "Sector3Time", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "IsPersonalBest", "Compound", "TyreLife",
                        "FreshTyre", "TrackStatus", "Deleted", "DeletedReason", "LapStartDate", "LapEndDate"]

        try:
            session = fastf1.get_session(self.year, round_num, session_type)
            session.load()

            quali_results = session.results
            quali_results_dict = dict(
                zip(quali_results["Abbreviation"], quali_results["Position"]))

            session_df = session.laps
            session_df = session_df[(session_df["PitInTime"].isna()) & (
                session_df["PitOutTime"].isna())]
            session_df["LapEndDate"] = session_df["LapStartDate"] + \
                session_df["LapTime"]

            laps_df = session_df.copy()
            laps_df["Round"] = round_num
            laps_df = laps_df[cols_to_keep]

            race_control_df = session.race_control_messages
            quali_sessions = race_control_df[(race_control_df.Message.str.contains("GREEN LIGHT")) |
                                             (race_control_df.Message.str.contains("CHEQUERED FLAG")) |
                                             (race_control_df.Message.str.contains("WILL NOT BE RESUMED"))]

            quali_sessions["GreenLightCount"] = (quali_sessions["Message"].str.contains("GREEN LIGHT")
                                                 .groupby((quali_sessions["Message"] != quali_sessions["Message"].shift()).cumsum())
                                                 .cumsum())
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

            quali_df["Position"] = quali_df["Driver"].map(quali_results_dict)

            telemetry_dfs = []
            dfs = []
            drivers = session_df["Driver"].unique()
            for driver in drivers:
                car_df = session_df[session_df["Driver"] ==
                                    driver].get_car_data().add_distance()
                lap_df = quali_df[quali_df["Driver"]
                                  == driver].reset_index(drop=True)
                if len(lap_df) > 1:
                    pivot_telemetry_df = self._process_telemetry(
                        car_df,
                        lap_df,
                        normalize=normalize_telemetry,
                        target_points=target_points
                    )
                    pivot_telemetry_df["Driver"] = driver
                    dfs.append(pivot_telemetry_df)

            return quali_df, pd.concat(dfs)

        except Exception as e:
            print(f"Error processing qualifying round {round_num}: {str(e)}")
            return None

    def _process_race_session(self, round_num: int, session_type: str = "R", normalize_telemetry: bool = False, target_points: int = 300) -> Optional[pd.DataFrame]:
        """
        Process a single race session's data.

        Args:
            round_num (int): Round number of the session to process
            session_type (str): Type of session ('R' for race)

        Returns:
            Optional[Tuple[pd.DataFrame, pd.DataFrame]]: Processed race data and telemetry data,
                                                       or None if processing fails
        """
        cols_to_keep = ["Round", "Driver", "DriverNumber", "Team", "LapTime", "LapNumber", "Stint", "Sector1Time", "Sector2Time",
                        "Sector3Time", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "IsPersonalBest", "Compound", "TyreLife",
                        "FreshTyre", "TrackStatus", "Deleted", "DeletedReason", "LapStartDate", "LapEndDate"]

        try:
            session = fastf1.get_session(self.year, round_num, session_type)
            session.load()

            race_results = session.results
            race_results_dict = dict(
                zip(race_results["Abbreviation"], race_results["Position"]))

            session_df = session.laps
            session_df["LapEndDate"] = session_df["LapStartDate"] + \
                session_df["LapTime"]

            laps_df = session_df.copy()
            laps_df["Round"] = round_num
            laps_df = laps_df[cols_to_keep]

            for i in ["Sector1Time", "Sector2Time", "Sector3Time"]:
                laps_df[i] = laps_df[i].dt.total_seconds()

            laps_df["Position"] = laps_df["Driver"].map(race_results_dict)

            telemetry_dfs = []
            dfs = []
            drivers = session_df["Driver"].unique()
            for driver in drivers:

                car_df = session_df[session_df["Driver"] ==
                                    driver].get_car_data().add_distance()
                lap_df = laps_df[laps_df["Driver"]
                                 == driver].reset_index(drop=True)

                if len(lap_df) > 1:
                    pivot_telemetry_df = self._process_telemetry(
                        car_df,
                        lap_df,
                        normalize=normalize_telemetry,
                        target_points=target_points
                    )
                    pivot_telemetry_df["Driver"] = driver
                    dfs.append(pivot_telemetry_df)

            return laps_df, pd.concat(dfs)

        except Exception as e:
            print(f"Error processing race round {round_num}: {str(e)}")
            return None

    def _rank_quali_laps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ranks qualifying lap times chronologically within each qualifying session.

        Processes a DataFrame of qualifying laps to add rankings and track the fastest
        lap times as they occur during each qualifying session (Q1, Q2, Q3).

        Args:
            df (pd.DataFrame): DataFrame containing qualifying session lap data

        Returns:
            pd.DataFrame: Original DataFrame with added columns for lap rankings and fastest lap deltas
        """
        df = df.sort_values(["Round", "QualiSession", "LapStartDate"])

        def update_rankings(group):
            group = group.copy()
            for idx in range(len(group)):
                current_slice = group.iloc[:idx + 1]
                current_ranks = current_slice["LapTime"].rank(
                    method="min", ascending=True)
                group.iloc[idx, group.columns.get_loc(
                    "LapRank")] = current_ranks.iloc[-1]

                group.iloc[idx, group.columns.get_loc("CurrentFastestLap")] = \
                    current_slice["LapTime"].min()

            return group

        df["LapRank"] = float("nan")
        df["CurrentFastestLap"] = pd.Timedelta(0)

        df = df.groupby("QualiSession", group_keys=False).apply(
            update_rankings)

        df["DeltaFastestLap"] = (
            df["LapTime"] - df["CurrentFastestLap"]).dt.total_seconds()

        return df

    def _process_telemetry(self, car_df: pd.DataFrame, lap_df: pd.DataFrame, normalize: bool = False, target_points: int = 300) -> pd.DataFrame:
        """
        Process raw telemetry data for a single car's laps.

        Assigns lap numbers to telemetry data points and calculates cumulative
        distance within each lap.

        Args:
            car_df (pd.DataFrame): Raw telemetry data for a single car
            lap_df (pd.DataFrame): Lap timing data for the same car

        Returns:
            pd.DataFrame: Processed telemetry data with lap numbers and cumulative distances
        """
        car_df["Brake"] = car_df["Brake"].astype(int)

        lap_starts = lap_df["LapStartDate"].values
        lap_ends = lap_df["LapEndDate"].values

        car_df["LapNumber"] = 0
        for i, (lap_start, lap_end) in enumerate(zip(lap_starts, lap_ends), 1):
            car_df.loc[(car_df["Date"] >= lap_start) & (
                car_df["Date"] <= lap_end), "LapNumber"] = i

        car_df = car_df[car_df["LapNumber"] != 0]

        grouped = car_df.groupby("LapNumber")
        result_df = grouped.apply(lambda x: x.assign(CumulativeDistance=x["Distance"].sub(
            x["Distance"].iloc[0]).cumsum())).reset_index(level=0, drop=True)
        result_df["CumulativeDistance"].fillna(0, inplace=True)

        if normalize:
            return self._normalize_telemetry_data(result_df, target_points)

        return result_df

    def _normalize_telemetry_data(self, df: pd.DataFrame, target_points: int = 300,) -> pd.DataFrame:
        """
        Normalize telemetry data to have a consistent number of data points per lap.

        Interpolates telemetry data to ensure each lap has the same number of data points,
        making it easier to compare laps and perform analysis.

        Args:
            df (pd.DataFrame): Telemetry data to normalize
            target_points (int, optional): Number of points to interpolate to per lap. Defaults to 300.

        Returns:
            pd.DataFrame: Normalized telemetry data with consistent number of points per lap
        """
        laps = df["LapNumber"].unique()

        normalized_data = []

        for lap in laps:
            lap_data = df[df["LapNumber"] == lap].copy()

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

            time_interpolator = interp1d(
                original_points,
                # Convert timedelta to seconds for interpolation
                lap_data["Time"].dt.total_seconds(),
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate"
            )

            interpolated_time_seconds = time_interpolator(new_points)
            normalized_lap["Time"] = pd.to_timedelta(
                interpolated_time_seconds, unit="s")  # Convert back to timedelta

            # Add LapNumber back
            normalized_df = pd.DataFrame(normalized_lap)
            normalized_df["LapNumber"] = lap
            normalized_data.append(normalized_df)

        return pd.concat(normalized_data)

        #     normalized_df = pd.DataFrame(normalized_lap)
        #     normalized_df["LapNumber"] = lap
        #     normalized_data.append(normalized_df)

        # return pd.concat(normalized_data)
