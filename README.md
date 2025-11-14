# Model Estimator

A standalone CLI tool for simple power-law model estimation optimized for small samples. Originally developed for memory usage prediction in computational workflows.

## Features

- **Power-Law Modeling**: Fits models of the form `y = coeff × var1^exp1 × var2^exp2 × ...`
- **Simple OLS Approach**: Straightforward ordinary least squares, no regularization needed
- **Automatic Feature Engineering**: Creates two-way interaction terms between significant predictors
- **Statistical Validation**:
  - Univariate significance testing (relaxed p < 0.2 threshold for small samples)
  - Automatic overfitting protection (limits predictors based on sample size)
  - Mean Absolute Error (MAE) calculation
- **Comprehensive Output**:
  - OLS and QR90 (90th percentile) model coefficients
  - Visualization plots
  - Copy-paste ready Python code
  - Performance metrics

## Installation

```bash
# Install in development mode
pip install -e .

# Or install directly
pip install .
```

## Usage

### Basic Usage

```bash
model-estimator data.csv --target-column ram
```

### With Custom Output

```bash
model-estimator data.csv --target-column memory --output my_model
```

### Options

- `csv_path`: Path to CSV, Excel (.xlsx), or ODS file with data
- `--target-column`, `-t`: Name of the column to estimate/predict (required)
- `--output`, `-o`: Output file prefix (default: power_law_model)

## Input Data Format

The tool accepts CSV, Excel (.xlsx), or ODS files with:
- One target column (specified by `--target-column`)
- Multiple numeric predictor columns
- All values should be positive (for log transformation)

Example:

```csv
ram,n_cells,n_genes,n_features
10.5,1000,500,2000
25.3,5000,1200,8000
45.7,10000,2500,15000
```

## Output Files

The tool generates:

1. **`{output}_model.json`**: JSON file with model coefficients and offset
2. **`{output}_simple_model.png`**: Model fit visualization showing actual vs predicted values
3. **Console output**: Detailed model coefficients, offset for no underestimation, and copy-paste ready Python code

### JSON Format

The `model.json` file contains:
```json
{
  "intercept": 0.0530,
  "offset": 1.8401,
  "features": [
    {
      "feature": "n_cells",
      "coefficient": 1.0,
      "exponent": 0.2466
    }
  ]
}
```

The model formula is: `y = intercept × (feature1 ^ exponent1) × (feature2 ^ exponent2) × ... + offset`

## Example

```bash
# Estimate memory usage from workflow parameters
model-estimator mem_data.csv --target-column ram

# Output:
# - power_law_model_simple_model.png
# - Console output with coefficients and formulas
```

## Model Type

The tool fits an **OLS (Ordinary Least Squares)** model for average predictions. It also calculates an offset value that, when added to predictions, ensures no underestimation on the training data.

## How It Works

The tool follows a simplified approach optimized for small samples:

1. **Univariate Testing**: Tests each predictor individually with relaxed significance threshold (p < 0.2)
2. **Interaction Terms**: Automatically creates two-way interactions between significant predictors
3. **Overfitting Protection**: Limits total predictors to avoid exceeding sample size constraints
4. **Model Fitting**: Fits OLS model for average predictions
5. **Offset Calculation**: Computes the offset needed to ensure no underestimation on training data
6. **Validation**: Calculates MAE and checks for overfitting indicators

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- statsmodels >= 0.13.0
- matplotlib >= 3.4.0
- odfpy >= 1.4.1 (for ODS files)
- openpyxl >= 3.0.0 (for Excel files)

## GitHub Actions Usage

```yaml
- name: Run model estimation
  run: |
    pip install -e .
    model-estimator data.csv --target-column ram --output results
```

## License

[Your License Here]