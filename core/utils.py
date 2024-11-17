from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_distance_matrices(results):
    num_plots = len(results["distance_matrices"])

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

    drivers = ["LEC", "NOR", "VER"]

    # Convert axes to array for consistent indexing
    axes = np.array(axes).reshape(-1) if num_plots == 1 else np.array(axes)

    for idx, (round_key, matrix_data) in enumerate(results["distance_matrices"].items()):
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
