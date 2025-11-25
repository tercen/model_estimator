#!/usr/bin/env python3
"""
Simplified power-law model estimation for small samples.
Uses basic OLS with main effects and two-way interactions.
"""

import argparse
import sys
import json
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from itertools import combinations
from .data_loader import load_data


def simple_power_law_fit(df, target_column, output_prefix='power_law_model'):
    """Fit a simple power-law model with main effects and two-way interactions."""
    print("=" * 70)
    print("SIMPLE POWER-LAW MODEL (Main Effects + Two-Way Interactions)")
    print("=" * 70)

    # Get numeric columns excluding target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    print(f"Available predictors: {numeric_cols}")

    # Log transform everything
    df_log = df.copy()
    df_log[f'log_{target_column}'] = np.log(df_log[target_column])

    for col in numeric_cols:
        df_log[f'log_{col}'] = np.log(df_log[col])

    # Test each predictor individually first
    print(f"\nUnivariate significance tests:")
    print(f"{'Variable':<15} {'R²':<8} {'p-value':<10} {'Coef':<10}")
    print("-" * 50)

    significant_vars = []
    for col in numeric_cols:
        log_col = f'log_{col}'
        X = sm.add_constant(df_log[log_col])
        y = df_log[f'log_{target_column}']

        try:
            model = sm.OLS(y, X).fit()
            r2 = model.rsquared
            pval = model.pvalues.iloc[1]
            coef = model.params.iloc[1]

            print(f"{col:<15} {r2:<8.3f} {pval:<10.3f} {coef:<10.3f}")

            if pval < 0.2:  # Very relaxed for small sample
                significant_vars.append(col)
        except:
            print(f"{col:<15} {'ERROR':<8} {'ERROR':<10} {'ERROR':<10}")

    print(f"\nSignificant variables (p < 0.2): {significant_vars}")

    if not significant_vars:
        print("No significant predictors found!")
        return None

    # Build interaction terms
    print(f"\nBuilding interaction terms...")
    log_predictors = [f'log_{var}' for var in significant_vars]

    # Add two-way interactions
    interaction_terms = []
    for var1, var2 in combinations(significant_vars, 2):
        interaction_name = f'log_{var1}:log_{var2}'
        df_log[interaction_name] = df_log[f'log_{var1}'] * df_log[f'log_{var2}']
        interaction_terms.append(interaction_name)
        print(f"  Added: {var1} × {var2}")

    all_predictors = log_predictors + interaction_terms
    print(f"\nTotal predictors: {len(log_predictors)} main effects + {len(interaction_terms)} interactions = {len(all_predictors)}")

    # Check if we have too many predictors for sample size
    if len(all_predictors) >= len(df_log) - 2:
        print(f"⚠️  Too many predictors ({len(all_predictors)}) for sample size ({len(df_log)})")
        print("   Using main effects only to avoid overfitting")
        all_predictors = log_predictors

    # Fit multivariate model
    X = sm.add_constant(df_log[all_predictors])
    y = df_log[f'log_{target_column}']

    # OLS model
    ols_model = sm.OLS(y, X).fit()
    print(f"\n" + "="*50)
    print("MULTIVARIATE OLS MODEL")
    print("="*50)
    print(ols_model.summary())

    # Generate predictions
    y_pred_ols = np.exp(ols_model.predict(X))
    y_actual = df[target_column]

    # Simple plot
    plt.figure(figsize=(8, 6))

    plt.scatter(y_actual, y_pred_ols, alpha=0.7, s=100)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
    plt.xlabel(f'Actual {target_column}')
    plt.ylabel(f'Predicted {target_column}')
    plt.title(f'OLS Model (R² = {ols_model.rsquared:.3f})')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = f'{output_prefix}_simple_model.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {plot_file}")

    # Generate simple code
    print(f"\n" + "="*70)
    print("SIMPLE PYTHON CODE")
    print("="*70)

    print("# OLS Model coefficients:")
    print(f"intercept_ols = {np.exp(ols_model.params['const']):.4f}")

    # Main effects
    for var in significant_vars:
        if f'log_{var}' in ols_model.params:
            coef = ols_model.params[f'log_{var}']
            print(f"power_{var} = {coef:.4f}")

    # Interactions
    for term in interaction_terms:
        if term in ols_model.params:
            coef = ols_model.params[term]
            var_pair = term.replace('log_', '').replace(':', '_x_')
            print(f"power_{var_pair} = {coef:.4f}")

    print(f"\n# Calculate {target_column} estimate:")
    print(f"# {target_column} = {np.exp(ols_model.params['const']):.4f}", end="")

    # Main effects
    for var in significant_vars:
        if f'log_{var}' in ols_model.params:
            coef = ols_model.params[f'log_{var}']
            print(f" * ({var} ** {coef:.4f})", end="")

    # Interactions
    for term in interaction_terms:
        if term in ols_model.params:
            coef = ols_model.params[term]
            vars_in_term = term.replace('log_', '').split(':')
            print(f" * (({vars_in_term[0]} * {vars_in_term[1]}) ** {coef:.4f})", end="")

    print()

    # Calculate offset to never underestimate (OLS)
    residuals_ols = y_actual - y_pred_ols
    max_underestimate_ols = residuals_ols.max()
    print(f"\n# Offset to never underestimate (on training data):")
    print(f"offset = {max_underestimate_ols:.4f}")
    print(f"# To ensure no underestimation: {target_column}_safe = {target_column} + {max_underestimate_ols:.4f}")

    # Build JSON model structure
    model_json = {
        "intercept": float(np.exp(ols_model.params['const'])),
        "offset": float(max_underestimate_ols),
        "features": []
    }

    # Add main effects
    for var in significant_vars:
        if f'log_{var}' in ols_model.params:
            coef = ols_model.params[f'log_{var}']
            model_json["features"].append({
                "feature": var,
                "coefficient": 1.0,  # Coefficient is 1 for power-law form
                "exponent": float(coef)
            })

    # Add interaction terms
    for term in interaction_terms:
        if term in ols_model.params:
            coef = ols_model.params[term]
            vars_in_term = term.replace('log_', '').split(':')
            feature_name = f"{vars_in_term[0]}_x_{vars_in_term[1]}"
            model_json["features"].append({
                "feature": feature_name,
                "coefficient": 1.0,  # Coefficient is 1 for power-law form
                "exponent": float(coef)
            })

    # Save JSON file
    json_file = f'{output_prefix}_model.json'
    with open(json_file, 'w') as f:
        json.dump(model_json, f, indent=2)
    print(f"\n✓ Model saved to {json_file}")

    return {
        'ols_model': ols_model,
        'significant_vars': significant_vars,
        'interaction_terms': interaction_terms,
        'all_predictors': all_predictors,
        'r_squared': ols_model.rsquared,
        'offset': max_underestimate_ols,
        'model_json': model_json,
        'predictions': {
            'ols': y_pred_ols,
            'actual': y_actual
        }
    }


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Simplified Power-Law Model Estimation for Small Samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage
  model-estimator data.csv --target-column ram

  # Specify output file prefix
  model-estimator data.csv --target-column memory --output my_model
        '''
    )

    parser.add_argument(
        'csv_path',
        help='Path to CSV, Excel (.xlsx), or ODS file with data'
    )

    parser.add_argument(
        '--target-column',
        '-t',
        required=True,
        help='Name of the column to estimate/predict'
    )

    parser.add_argument(
        '--output',
        '-o',
        default='power_law_model',
        help='Output file prefix (default: power_law_model)'
    )

    parser.add_argument(
        '--ignore-columns',
        '-i',
        default='',
        help='Comma-separated list of column names to ignore as predictors'
    )

    args = parser.parse_args()

    # Print header
    print("=" * 70)
    print("Simplified Power-Law Model Estimation")
    print("=" * 70)
    print("For small samples: Simple model with interactions")
    print("=" * 70)

    # Parse ignored columns
    ignore_columns = [col.strip() for col in args.ignore_columns.split(',') if col.strip()]

    # Load data
    df = load_data(args.csv_path, target_column=args.target_column, ignore_columns=ignore_columns)
    if df is None:
        sys.exit(1)

    print(f"\nData shape: {df.shape}")
    print(f"Sample size: {len(df)} (small sample - keeping it simple!)")

    # Fit simple model
    results = simple_power_law_fit(df, args.target_column, output_prefix=args.output)

    if results:
        print(f"\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✓ Model R² = {results['r_squared']:.3f}")
        print(f"✓ Significant predictors: {results['significant_vars']}")
        print(f"✓ Interaction terms: {len(results['interaction_terms'])}")
        print(f"✓ Total predictors: {len(results['all_predictors'])}")
        print(f"✓ No regularization needed (simple model)")
        print(f"✓ Ready-to-use coefficients generated")

        # Simple validation
        mae_ols = np.mean(np.abs(results['predictions']['actual'] - results['predictions']['ols']))
        print(f"✓ MAE: {mae_ols:.2f}")
        print(f"✓ Offset for no underestimation: {results['offset']:.2f}")

        # Check for overfitting signs
        if results['r_squared'] > 0.95:
            print(f"⚠️  Very high R² ({results['r_squared']:.3f}) - possible overfitting")
        elif results['r_squared'] > 0.8:
            print(f"✓ Good fit without overfitting")
        else:
            print(f"✓ Reasonable fit for small sample")

        print(f"\n" + "="*70)
        print("RECOMMENDATION FOR SMALL SAMPLES:")
        print("="*70)
        print("✓ Use main effects + selected interactions")
        print("✓ No regularization needed for simple models")
        print("✓ Monitor for overfitting (very high R²)")
        print("✓ Focus on interpretability")
        print("="*70)
    else:
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
