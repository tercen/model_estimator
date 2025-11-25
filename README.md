# Model Estimator

A standalone CLI tool for simple power-law model estimation optimized for small samples. Originally developed for memory usage prediction in computational workflows.

## Features

- **Power-Law Modeling**: Fits models of the form `y = coeff × var1^exp1 × var2^exp2 × ...`
- **Simple OLS Approach**: Straightforward ordinary least squares, no regularization needed
- **Automatic Feature Engineering**: Creates two-way interaction terms between significant predictors
- **Statistical Validation**:
  - AIC-based predictor selection (configurable ΔAIC threshold, default < 5)
  - Automatic overfitting protection (limits predictors based on sample size)
  - Mean Absolute Error (MAE) calculation
  - Intercept-only fallback when no significant predictors are found
- **Comprehensive Output**:
  - OLS model coefficients
  - Visualization plots
  - Copy-paste ready Python code
  - JSON model file for easy integration
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

### Ignore Specific Columns

```bash
model-estimator data.csv --target-column ram --ignore-columns "id,name,timestamp"
```

### Custom Delta AIC Threshold

```bash
# Stricter selection (fewer predictors)
model-estimator data.csv --target-column ram --max-delta-aic 2

# More relaxed selection (more predictors)
model-estimator data.csv --target-column ram --max-delta-aic 7
```

### Combined Options

```bash
model-estimator data.csv \
  --target-column ram \
  --output my_model \
  --ignore-columns "id,timestamp" \
  --max-delta-aic 4
```

### Docker Usage

```bash
# Build the Docker image
docker build -t model-estimator .

# Run with a local data file
docker run --rm -v $(pwd)/data:/data model-estimator \
  model-estimator /data/data.csv --target-column ram

# With all options
docker run --rm -v $(pwd)/data:/data -v $(pwd)/output:/output model-estimator \
  model-estimator /data/data.csv \
  --target-column ram \
  --output /output/my_model \
  --ignore-columns "id,timestamp" \
  --max-delta-aic 4

# Using pre-built image from GitHub Container Registry
docker pull ghcr.io/tercen/model_estimator:main
docker run --rm -v $(pwd)/data:/data \
  ghcr.io/tercen/model_estimator:main \
  /data/data.csv \
  --target-column ram \
  --output /data/memory_model
```

### Options

- `csv_path`: Path to CSV file with data
- `--target-column`, `-t`: Name of the column to estimate/predict (required)
- `--output`, `-o`: Output file prefix (default: power_law_model)
- `--ignore-columns`, `-i`: Comma-separated list of column names to ignore as predictors (default: none)
- `--max-delta-aic`, `-d`: Maximum ΔAIC for predictor selection (default: 5.0, lower=stricter, ΔAIC<2=strict, ΔAIC<7=relaxed)

### Understanding ΔAIC (Delta AIC)

ΔAIC measures how much worse a model is compared to the best model. The tool fits a univariate model for each predictor and compares them:

**Interpretation:**
- **ΔAIC < 2**: Model has substantial support (essentially equivalent to best model)
- **ΔAIC 2-4**: Model has considerable support (good alternative)
- **ΔAIC 4-7**: Model has some support (worth considering)
- **ΔAIC 7-10**: Model has weak support
- **ΔAIC > 10**: No support (reject)

**Recommendations:**
- **Strict selection (fewer predictors)**: Use `--max-delta-aic 2` - only keeps best predictors
- **Moderate selection (balanced)**: Use `--max-delta-aic 5` (default) - includes reasonably good predictors
- **Relaxed selection (more predictors)**: Use `--max-delta-aic 7` - includes more marginal predictors
- **Very relaxed**: Use `--max-delta-aic 10` - accepts most predictors with any signal

For small samples (n < 50), the default (5.0) provides a good balance between capturing important predictors and avoiding overfitting.

## Input Data Format

The tool accepts CSV files with:
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
# - power_law_model_model.json (JSON model file)
# - power_law_model_simple_model.png (visualization)
# - Console output with coefficients and formulas
```

## Model Type

The tool fits an **OLS (Ordinary Least Squares)** model for average predictions. It also calculates an offset value that, when added to predictions, ensures no underestimation on the training data.

## How It Works

The tool follows a simplified approach optimized for small samples:

1. **Univariate Testing**: Tests each predictor individually and compares models using AIC
2. **Predictor Selection**: Selects predictors with ΔAIC below threshold (default < 5, lower=stricter)
3. **Interaction Terms**: Automatically creates two-way interactions between selected predictors
4. **Overfitting Protection**: Limits total predictors to avoid exceeding sample size constraints
5. **Model Fitting**: Fits OLS model for average predictions (or intercept-only if no predictors are selected)
6. **Offset Calculation**: Computes the offset needed to ensure no underestimation on training data
7. **Validation**: Calculates MAE and checks for overfitting indicators

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- statsmodels >= 0.13.0
- matplotlib >= 3.4.0

## GitHub Actions Usage

```yaml
- name: Run model estimation
  run: |
    pip install -e .
    model-estimator data.csv \
      --target-column ram \
      --output results \
      --ignore-columns "id,timestamp" \
      --max-delta-aic 4
```

## License

[Your License Here]