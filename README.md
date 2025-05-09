# F1-Exploration

A Python toolkit for analyzing Formula 1 telemetry data and driver performance metrics. This project leverages FastF1 data to provide comprehensive insights into driver performance, racing dynamics, and comparative analysis.

## Features

- **Telemetry Analysis**
  - Process and normalize lap-by-lap telemetry data
  - Generate speed profiles and distance matrices
  - Compare driving styles using PCA (Principal Component Analysis)

- **Driver Performance Index**
  - Calculate comprehensive performance metrics
  - Analyze consistency, technical execution, and pace
  - Generate weighted performance indices for driver comparisons

- **Data Visualization**
  - Speed profile comparisons across drivers and races
  - Distance matrix heatmaps
  - PCA-based driving style analysis plots
  - Performance breakdown visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/akatelync/f1-exploration.git
cd f1-exploration
```

2. Set up environment with UV:
```bash
# Create virtual environment
uv venv .venv

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

3. Register the IPykernel for Jupyter:
```bash
python -m ipykernel install --user --name=f1-exploration --display-name="Python (F1 Exploration)"
```

## Project Structure

```
f1-exploration/
├── core/
│   ├── extract.py      # Data extraction and processing
│   ├── telemetry.py    # Telemetry analysis functions
│   ├── dpi.py          # Driver Performance Index calculations
│   └── utils.py        # Utility functions
├── notebooks/          # Jupyter notebooks for analysis examples
├── tests/              # Unit tests
├── pyproject.toml      # Project dependencies and metadata
└── README.md          # Project documentation
```

## Usage

### Basic Example

```python
from core.extract import F1DataProcessor
from core.telemetry import F1TelemetryAnalyzer
from core.dpi import F1DriverPerformanceIndex

# Initialize data processor
processor = F1DataProcessor(
    cache_dir="./cache",
    year=2024
)

# Create telemetry analyzer
analyzer = F1TelemetryAnalyzer(
    n_samples=300,
    feature_cols=["RPM", "Speed", "nGear", "Throttle", "Brake", "DRS"]
)

# Get qualifying session data
quali_data, telemetry_data = processor.get_quali_session(
    rounds=[1, 2],
    drivers=["VER", "LEC", "NOR"]
)

# Analyze driver performance
analysis_results = analyzer.analyze_laps(
    telemetry_data=telemetry_data,
    session_type="qualifying",
    rounds=[1, 2],
    drivers=["VER", "LEC", "NOR"],
    pca=True
)
```

### Performance Index Calculation

```python
# Initialize performance index calculator
dpi = F1DriverPerformanceIndex(
    data_processor=processor,
    telemetry_analyzer=analyzer,
    session_type="qualifying"
)

# Calculate performance indices
performance_index = dpi.calculate_performance_index(
    rounds=[1, 2],
    drivers=["VER", "LEC", "NOR"]
)

# Visualize results
dpi.plot_performance_breakdown(performance_index)
```

## Features in Detail

### Telemetry Analysis
- **Data Processing**: Normalize and align telemetry data for consistent analysis
- **Speed Profiles**: Generate and compare driver speed profiles across laps
- **Distance Matrices**: Calculate similarity metrics between driving styles
- **PCA Analysis**: Reduce dimensionality of telemetry data to identify key driving style characteristics

### Driver Performance Index
- **Consistency Score**: Evaluate lap time and sector time consistency
- **Style Score**: Measure driving style distinctiveness and consistency
- **Technical Score**: Assess throttle control, braking efficiency, and gear changes
- **Pace Score**: Calculate relative pace and position-based performance

## Development

### Setting up a Development Environment

```bash
# Clone the repository
git clone https://github.com/akatelync/f1-exploration.git
cd f1-exploration

# Create and activate a virtual environment with UV
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
uv pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Dependencies

- fastf1
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy
- jupyter
- ipykernel

## Data Sources

This project uses data from:
- FastF1 API
- OpenF1 Data

## License

This project is licensed under the MIT License - see the LICENSE file for details.