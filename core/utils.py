from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union, Dict, Any

round_dict = {1: 'Bahrain Grand Prix',
              2: 'Saudi Arabian Grand Prix',
              3: 'Australian Grand Prix',
              4: 'Japanese Grand Prix',
              5: 'Chinese Grand Prix',
              6: 'Miami Grand Prix',
              7: 'Emilia Romagna Grand Prix',
              8: 'Monaco Grand Prix',
              9: 'Canadian Grand Prix',
              10: 'Spanish Grand Prix',
              11: 'Austrian Grand Prix',
              12: 'British Grand Prix',
              13: 'Hungarian Grand Prix',
              14: 'Belgian Grand Prix',
              15: 'Dutch Grand Prix',
              16: 'Italian Grand Prix',
              17: 'Azerbaijan Grand Prix',
              18: 'Singapore Grand Prix',
              19: 'United States Grand Prix',
              20: 'Mexico City Grand Prix',
              21: 'São Paulo Grand Prix'
              }

driver_colors = {
    "VER": "#3671c6",  # Blue
    "NOR": "#ff8000",  # Orange
    "LEC": "#e8002d"   # Red
}

drivers = ["LEC", "NOR", "VER"]


def plot_distance_matrices(
    distance_matrices: Dict[np.int64, np.ndarray],
    rounds: Optional[Union[int, List[int]]] = None
) -> Dict[np.int64, np.ndarray]:
    """
    Filter distance matrices based on specified rounds.

    Args:
        distance_matrices: Dictionary mapping round numbers to distance matrices
        rounds: Single round number (int) or list of round numbers to filter by

    Returns:
        Filtered dictionary of distance matrices
    """
    if rounds:
        # Convert single integer to list if necessary
        rounds_list = [rounds] if isinstance(rounds, int) else rounds

    distance_matrices = {
        k: v for k, v in distance_matrices.items() if k in rounds_list}
    num_plots = len(distance_matrices)

    # Dynamic layout calculation
    if num_plots <= 3:
        # For 1-3 plots, use a single column
        num_cols = num_plots
        num_rows = 1
    else:
        # For 4+ plots, use 2 columns
        num_rows = 2
        num_cols = int(np.ceil(num_plots / num_rows))

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))
    fig.suptitle("Distance Matrices by Round", fontsize=14, y=1.02)

    # Convert axes to array for consistent indexing
    axes = np.array(axes).reshape(-1) if num_plots == 1 else np.array(axes)

    for idx, (round_key, matrix_data) in enumerate(distance_matrices.items()):
        current_ax = axes.flatten()[idx]

        matrix = np.array([[matrix_data[i][j] for j in drivers]
                          for i in drivers])
        event_info = round_key
        event_name = round_dict[event_info]

        sns.heatmap(
            matrix,
            ax=current_ax,
            cmap="YlOrRd",
            annot=True,
            fmt=".2f",
            square=True,
            xticklabels=drivers,
            yticklabels=drivers,
            cbar_kws={"label": "Distance"}
        )

        title = f"Round {event_info}\n{event_name}"
        current_ax.set_title(title, pad=10)
        current_ax.set_xticklabels(current_ax.get_xticklabels(), rotation=45)
        current_ax.set_yticklabels(current_ax.get_yticklabels(), rotation=0)

    # Remove empty subplots
    if num_plots < (num_rows * num_cols):
        for idx in range(num_plots, num_rows * num_cols):
            fig.delaxes(axes.flatten()[idx])

    plt.tight_layout()


def analyze_selected_drivers_pca(df, driver_info, selected_drivers):
    """
    Perform PCA analysis on selected drivers, visualize results, and compute centroids and spreads.

    Parameters:
        df (pd.DataFrame): DataFrame containing the dataset for PCA analysis.
        driver_info (pd.DataFrame): DataFrame containing driver information (must include a "Driver" column).
        selected_drivers (list): List of drivers to include in the analysis.

    Returns:
        dict: A dictionary containing:
            - 'pca': Fitted PCA object.
            - 'pca_df': DataFrame with PCA-transformed data and driver labels.
            - 'centroids': DataFrame with centroids (mean positions) for each driver in PCA space.
            - 'spreads': DataFrame with spreads (standard deviations) for each driver in PCA space.
    """
    mask = driver_info["Driver"].isin(selected_drivers)
    X = df[mask].select_dtypes(include=[np.number])
    driver_info_filtered = driver_info[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = 3
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
    pca_df["Driver"] = driver_info_filtered["Driver"].values

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Driver",
                    style="Driver", s=100, alpha=0.6)
    plt.title(f"Driver Comparison: {', '.join(selected_drivers)}")
    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance explained)")
    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance explained)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    driver_centroids = pca_df.groupby("Driver")[["PC1", "PC2", "PC3"]].mean()
    driver_spreads = pca_df.groupby("Driver")[["PC1", "PC2", "PC3"]].std()

    plt.figure(figsize=(10, 6))
    for driver in selected_drivers:
        plt.errorbar(driver_centroids.loc[driver, "PC1"],
                     driver_centroids.loc[driver, "PC2"],
                     xerr=driver_spreads.loc[driver, "PC1"],
                     yerr=driver_spreads.loc[driver, "PC2"],
                     label=driver, fmt="o", capsize=5, capthick=2,
                     markersize=10, alpha=0.6)
    plt.title(f"Driving Styles Comparison: {', '.join(selected_drivers)}")
    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance explained)")
    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance explained)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    print("\nDriver Centroids (average position in PCA space):")
    print(driver_centroids)

    print("\nDriver Variability (spread in PCA space):")
    print(driver_spreads)

    print("\nPairwise Distances between Drivers:")
    for i, driver1 in enumerate(selected_drivers):
        for driver2 in selected_drivers[i+1:]:
            distance = np.linalg.norm(
                driver_centroids.loc[driver1] - driver_centroids.loc[driver2])
            print(f"{driver1} vs {driver2}: {distance:.2f}")

    return {
        "pca": pca,
        "pca_df": pca_df,
        "centroids": driver_centroids,
        "spreads": driver_spreads
    }


def plot_speed_profile(
    processed_laps: pd.DataFrame,
    rounds: Optional[Union[int, List[int]]] = None
) -> None:
    """
    Creates speed profile plots for specified rounds.

    Args:
        processed_laps: DataFrame containing the lap data with columns:
                       ['Round', 'Driver', 'normalized_time', 'Speed']
        rounds: Single round number (int) or list of round numbers to filter by
               If None, plots all available rounds

    Returns:
        None. Displays the speed profile plots.
    """

    if rounds:
        # Convert single integer to list if necessary
        rounds_list = [rounds] if isinstance(rounds, int) else rounds

    processed_laps = processed_laps[processed_laps["Round"].isin(rounds_list)]
    rounds = processed_laps["Round"].unique()
    num_rounds = len(rounds)

    # Set number of columns to 3
    num_cols = 3
    num_rows = (num_rounds + num_cols - 1) // num_cols

    # Create figure with extra space on the right for the legend
    # Slightly wider to accommodate legend
    fig = plt.figure(figsize=(22, 5*num_rows))

    for idx, round_num in enumerate(rounds, 1):
        ax = plt.subplot(num_rows, num_cols, idx)

        round_data = processed_laps[processed_laps["Round"] == round_num]
        for driver in processed_laps["Driver"].unique():
            driver_data = round_data[round_data["Driver"] == driver]
            grouped = driver_data.groupby("normalized_time")
            mean_speed = grouped["Speed"].mean()
            std_speed = grouped["Speed"].std()

            # Use the driver-specific color for both the line and fill
            driver_color = driver_colors.get(driver)
            plt.plot(driver_data["normalized_time"].unique(),
                     mean_speed,
                     label=driver if idx == 1 else "",  # Only include label for first subplot
                     color=driver_color)
            plt.fill_between(driver_data["normalized_time"].unique(),
                             mean_speed - std_speed,
                             mean_speed + std_speed,
                             alpha=0.2,
                             color=driver_color)

            plt.xlabel("Normalized Lap Time")
            plt.ylabel("Speed")
            plt.title(f"Qualifying Speed Profiles - {round_dict[round_num]}")

    fig.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center",
               ncol=len(driver_colors), fontsize=13)
    # Adjusted to leave space at the top for legend
    plt.tight_layout(rect=[0, 0, 1, 1])


def plot_combined_analysis(
    distance_matrices: Dict[np.int64, np.ndarray],
    processed_laps: pd.DataFrame,
    rounds: Optional[Union[int, List[int]]] = None,
) -> None:
    """
    Creates a combined visualization with distance matrices and speed profiles side by side for each round.

    Args:
        distance_matrices: Dictionary mapping round numbers to distance matrices
        processed_laps: DataFrame containing processed lap data with columns:
                       ['Round', 'Driver', 'normalized_time', 'Speed']
        rounds: Single round number (int) or list of round numbers to filter by
        driver_colors: Dictionary mapping driver codes to their color codes
        round_dict: Dictionary mapping round numbers to race names

    Returns:
        None. Displays the combined plots.
    """
    if rounds is not None:
        rounds_list = [rounds] if isinstance(rounds, int) else rounds
        distance_matrices = {
            k: v for k, v in distance_matrices.items() if k in rounds_list}
        processed_laps = processed_laps[processed_laps["Round"].isin(
            rounds_list)]

    num_rounds = len(distance_matrices)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 5 * num_rounds), dpi=300)

    # Loop through each round
    for idx, (round_num, matrix_data) in enumerate(distance_matrices.items(), 1):
        # Distance Matrix subplot (left)
        ax1 = plt.subplot(num_rounds, 2, 2*idx - 1)

        # Create distance matrix heatmap
        matrix = np.array([[matrix_data[i][j] for j in drivers]
                          for i in drivers])
        sns.heatmap(
            matrix,
            ax=ax1,
            cmap="YlOrRd",
            annot=True,
            fmt=".2f",
            square=True,
            xticklabels=drivers,
            yticklabels=drivers,
            cbar_kws={"label": "Distance"}
        )
        ax1.set_title(
            f"Distance Matrix - Round {round_num}\n{round_dict[round_num]}", pad=10)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

        # Speed Profile subplot (right)
        ax2 = plt.subplot(num_rounds, 2, 2*idx)

        # Create speed profile plot
        round_data = processed_laps[processed_laps["Round"] == round_num]
        for driver in drivers:
            driver_data = round_data[round_data["Driver"] == driver]
            if not driver_data.empty:
                grouped = driver_data.groupby("normalized_time")
                mean_speed = grouped["Speed"].mean()
                std_speed = grouped["Speed"].std()

                driver_color = driver_colors.get(driver)
                ax2.plot(
                    driver_data["normalized_time"].unique(),
                    mean_speed,
                    label=driver,
                    color=driver_color
                )
                ax2.fill_between(
                    driver_data["normalized_time"].unique(),
                    mean_speed - std_speed,
                    mean_speed + std_speed,
                    alpha=0.2,
                    color=driver_color
                )

        ax2.set_xlabel("Normalized Lap Time")
        ax2.set_ylabel("Speed")
        ax2.set_title(
            f"Speed Profile - Round {round_num}\n{round_dict[round_num]}")
        ax2.legend(loc="lower left")

    plt.tight_layout()


def reshape_telemetry_data(
    telemetry_df: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
    index_cols: List[str] = ["Round", "Driver", "LapNumber"]
) -> pd.DataFrame:
    """
    Reshapes telemetry data from long format to wide format where each lap's metrics
    are spread across columns with sequential numbering.

    Args:
        telemetry_df: DataFrame containing telemetry data with at least the following columns:
                     - Round
                     - Driver
                     - LapNumber 
                     - Metric columns (e.g., RPM, Speed, etc.)
        metric_cols: List of metric column names to reshape. If None, defaults to 
                    ["RPM", "Speed", "nGear", "Throttle", "Brake", "DRS"]
        index_cols: List of columns to use as index in the final DataFrame.
                   Default is ["Round", "Driver", "LapNumber"]

    Returns:
        pd.DataFrame: Reshaped DataFrame where:
            - Each row represents a unique lap
            - Columns are named as {metric}_{index} where index represents the sequential
              position in the lap
            - DataFrame is indexed by index_cols

    Example:
        >>> metric_columns = ["Speed", "RPM"]
        >>> reshaped_df = reshape_telemetry_data(quali_telemetry, metric_cols=metric_columns)
        >>> # Results in columns like: Speed_0, Speed_1, ..., RPM_0, RPM_1, ...
    """
    # Default metric columns if none provided
    if metric_cols is None:
        metric_cols = ["RPM", "Speed", "nGear", "Throttle", "Brake", "DRS"]

    # Validate input DataFrame has required columns
    required_cols = index_cols + metric_cols
    missing_cols = [
        col for col in required_cols if col not in telemetry_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in telemetry_df: {missing_cols}")

    grouped_data: List[Dict[str, Any]] = []

    # Group by index columns and reshape data
    for group_key, lap_data in telemetry_df.groupby(index_cols):
        # Convert group_key to dict if it's a tuple
        if isinstance(group_key, tuple):
            group_dict = dict(zip(index_cols, group_key))
        else:
            group_dict = {index_cols[0]: group_key}

        reshaped_data = group_dict.copy()

        # Reshape each metric into sequential columns
        for metric in metric_cols:
            for idx, value in enumerate(lap_data[metric]):
                reshaped_data[f"{metric}_{idx}"] = value

        grouped_data.append(reshaped_data)

    # Create DataFrame and set index
    reshaped_df = pd.DataFrame(grouped_data)
    reshaped_df.set_index(index_cols, inplace=True)

    return reshaped_df
