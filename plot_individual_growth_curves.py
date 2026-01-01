#!/usr/bin/env python
"""
Visualize Individual Growth Curves
===================================

Plots individual animal growth curves for animals with ≥3 observations,
showing each animal's fitted curve overlaid with a population curve.

Creates comprehensive visualizations showing:
  - Individual data points and fitted curves
  - Population curve from aggregated parameters (Option A)
  - Separate panels for Overall, Males, Females
  - Distribution of individual variation

Usage:
    python plot_individual_growth_curves.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from giraffesurvival.data import (
    load_prepare_wild,
    load_prepare_zoo,
    add_age_class_midpoints,
    assign_initial_ages_from_classes,
    add_vtb_umb_flag,
)
from giraffesurvival.models import (
    gompertz_model,
    logistic_model,
    von_bertalanffy_model,
    richards_model,
    poly3_model,
    poly4_model,
)


def get_model_function(model_name: str):
    """Return the model function for a given model name."""
    # Strip _free suffix if present
    model_name = model_name.replace('_free', '').replace('_fixed_y0', '')
    
    model_funcs = {
        'gompertz': gompertz_model,
        'logistic': logistic_model,
        'von_bertalanffy': von_bertalanffy_model,
        'richards': richards_model,
        'poly3': poly3_model,
        'poly4': poly4_model,
    }
    return model_funcs.get(model_name)


def plot_all_individual_curves(
    wild: pd.DataFrame,
    individual_fits: pd.DataFrame,
    output_dir: Path,
    min_obs: int = 3,
    sex_filter: str = None,
):
    """
    Create comprehensive plot showing all individual curves.
    
    Args:
        wild: Wild giraffe data
        individual_fits: DataFrame with individual fit results
        output_dir: Directory to save plots
        min_obs: Minimum observations to include
        sex_filter: 'M', 'F', or None for all
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter by sex if requested
    if sex_filter:
        wild_subset = wild[wild['Sex'] == sex_filter]
        fits_subset = individual_fits[individual_fits['Sex'] == sex_filter]
        title_suffix = f" ({sex_filter})"
        file_suffix = f"_{sex_filter}"
    else:
        wild_subset = wild
        fits_subset = individual_fits
        title_suffix = " (All)"
        file_suffix = "_all"
    
    # Filter by minimum observations
    fits_subset = fits_subset[fits_subset['n_obs'] >= min_obs].copy()
    
    if len(fits_subset) == 0:
        print(f"No animals with ≥{min_obs} observations for {sex_filter or 'all'}")
        return
    
    # Calculate population curve from aggregated parameters
    # Get most common model
    best_model = fits_subset['model_y0free'].value_counts().idxmax()
    
    # Extract parameter columns for that model
    if best_model in ['gompertz', 'logistic', 'von_bertalanffy']:
        param_names = ['A', 'k', 't0']
    elif best_model == 'richards':
        param_names = ['A', 'k', 't0', 'nu']
    elif best_model == 'poly3':
        param_names = ['a3', 'a2', 'a1', 'a0']
    elif best_model == 'poly4':
        param_names = ['a4', 'a3', 'a2', 'a1', 'a0']
    else:
        param_names = []
    
    # Aggregate parameters (mean)
    pop_params = []
    for param in param_names:
        col = f"{param}_free"
        if col in fits_subset.columns:
            valid = fits_subset[col].dropna()
            if len(valid) > 0:
                pop_params.append(valid.mean())
            else:
                pop_params.append(0)
        else:
            pop_params.append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    age_range = np.linspace(0, 240, 500)
    
    # Plot each individual's data and fitted curve
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(fits_subset))))
    
    for idx, (_, fit_row) in enumerate(fits_subset.iterrows()):
        aid = fit_row['AID']
        model_name = fit_row['model_y0free']
        
        # Get animal's data
        animal_data = wild_subset[wild_subset['AID'] == aid].copy()
        if len(animal_data) < min_obs:
            continue
        
        # Plot data points
        color = colors[idx % len(colors)]
        ax.scatter(animal_data['age_months'], animal_data['TH'], 
                  s=50, alpha=0.6, color=color, edgecolors='black', linewidth=0.5,
                  zorder=2)
        
        # Get model function
        model_func = get_model_function(model_name)
        if model_func is None:
            continue
        
        # Extract parameters for this individual
        if model_name in ['gompertz', 'logistic', 'von_bertalanffy']:
            param_cols = ['A_free', 'k_free', 't0_free']
        elif model_name == 'richards':
            param_cols = ['A_free', 'k_free', 't0_free', 'nu_free']
        elif model_name == 'poly3':
            param_cols = ['a3_free', 'a2_free', 'a1_free', 'a0_free']
        elif model_name == 'poly4':
            param_cols = ['a4_free', 'a3_free', 'a2_free', 'a1_free', 'a0_free']
        else:
            continue
        
        # Check if all parameters are available
        params = []
        for col in param_cols:
            if col in fit_row.index and not pd.isna(fit_row[col]):
                params.append(fit_row[col])
            else:
                break
        
        if len(params) == len(param_cols):
            # Plot individual fitted curve
            try:
                pred = model_func(age_range, *params)
                # Only plot reasonable values
                pred_clipped = np.clip(pred, 0, 1200)
                ax.plot(age_range, pred_clipped, color=color, alpha=0.4, 
                       linewidth=1.5, zorder=1)
            except:
                pass
    
    # Plot population curve (aggregated parameters)
    if len(pop_params) == len(param_names):
        pop_model_func = get_model_function(best_model)
        if pop_model_func:
            try:
                pop_pred = pop_model_func(age_range, *pop_params)
                pop_pred_clipped = np.clip(pop_pred, 0, 1200)
                ax.plot(age_range, pop_pred_clipped, 'k-', linewidth=4, 
                       label=f'Population Curve\n({best_model}, aggregated params)',
                       zorder=3, alpha=0.8)
            except:
                pass
    
    ax.set_xlabel('Age (months)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Height (cm)', fontsize=14, fontweight='bold')
    ax.set_title(f'Individual Growth Curves{title_suffix}\n'
                f'n={len(fits_subset)} animals with ≥{min_obs} observations',
                fontsize=15, fontweight='bold')
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 1000)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(alpha=0.3)
    
    # Add text box with summary
    textstr = f'Individual curves: {len(fits_subset)}\n'
    textstr += f'Most common model: {best_model}\n'
    textstr += f'Mean observations/animal: {fits_subset["n_obs"].mean():.1f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    fig_path = output_dir / f'individual_curves_min{min_obs}{file_suffix}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved individual curves plot to {fig_path}")
    plt.close()


def plot_individual_curves_grid(
    wild: pd.DataFrame,
    individual_fits: pd.DataFrame,
    output_dir: Path,
    min_obs: int = 3,
):
    """
    Create grid plot with Overall, Males, Females side-by-side.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 7))
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    age_range = np.linspace(0, 240, 500)
    
    for col_idx, (sex_filter, title) in enumerate([
        (None, 'Overall'),
        ('M', 'Males'),
        ('F', 'Females')
    ]):
        ax = fig.add_subplot(gs[0, col_idx])
        
        # Filter data
        if sex_filter:
            wild_subset = wild[wild['Sex'] == sex_filter]
            fits_subset = individual_fits[individual_fits['Sex'] == sex_filter]
        else:
            wild_subset = wild
            fits_subset = individual_fits
        
        # Filter by minimum observations
        fits_subset = fits_subset[fits_subset['n_obs'] >= min_obs].copy()
        
        if len(fits_subset) == 0:
            ax.text(0.5, 0.5, f'No data\n(n=0 with ≥{min_obs} obs)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue
        
        # Calculate population curve
        best_model = fits_subset['model_y0free'].value_counts().idxmax()
        
        if best_model in ['gompertz', 'logistic', 'von_bertalanffy']:
            param_names = ['A', 'k', 't0']
        elif best_model == 'richards':
            param_names = ['A', 'k', 't0', 'nu']
        elif best_model == 'poly3':
            param_names = ['a3', 'a2', 'a1', 'a0']
        elif best_model == 'poly4':
            param_names = ['a4', 'a3', 'a2', 'a1', 'a0']
        else:
            param_names = []
        
        pop_params = []
        for param in param_names:
            col = f"{param}_free"
            if col in fits_subset.columns:
                valid = fits_subset[col].dropna()
                if len(valid) > 0:
                    pop_params.append(valid.mean())
        
        # Plot individuals
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(fits_subset))))
        
        for idx, (_, fit_row) in enumerate(fits_subset.iterrows()):
            aid = fit_row['AID']
            model_name = fit_row['model_y0free']
            
            animal_data = wild_subset[wild_subset['AID'] == aid].copy()
            if len(animal_data) < min_obs:
                continue
            
            color = colors[idx % len(colors)]
            ax.scatter(animal_data['age_months'], animal_data['TH'], 
                      s=40, alpha=0.5, color=color, edgecolors='black', linewidth=0.5,
                      zorder=2)
            
            model_func = get_model_function(model_name)
            if model_func is None:
                continue
            
            # Get params
            if model_name in ['gompertz', 'logistic', 'von_bertalanffy']:
                param_cols = ['A_free', 'k_free', 't0_free']
            elif model_name == 'richards':
                param_cols = ['A_free', 'k_free', 't0_free', 'nu_free']
            elif model_name == 'poly3':
                param_cols = ['a3_free', 'a2_free', 'a1_free', 'a0_free']
            elif model_name == 'poly4':
                param_cols = ['a4_free', 'a3_free', 'a2_free', 'a1_free', 'a0_free']
            else:
                continue
            
            params = []
            for col in param_cols:
                if col in fit_row.index and not pd.isna(fit_row[col]):
                    params.append(fit_row[col])
            
            if len(params) == len(param_cols):
                try:
                    pred = model_func(age_range, *params)
                    pred_clipped = np.clip(pred, 0, 1200)
                    ax.plot(age_range, pred_clipped, color=color, alpha=0.3, 
                           linewidth=1.2, zorder=1)
                except:
                    pass
        
        # Plot population curve
        if len(pop_params) == len(param_names):
            pop_model_func = get_model_function(best_model)
            if pop_model_func:
                try:
                    pop_pred = pop_model_func(age_range, *pop_params)
                    pop_pred_clipped = np.clip(pop_pred, 0, 1200)
                    ax.plot(age_range, pop_pred_clipped, 'k-', linewidth=4, 
                           label=f'Population\n({best_model})',
                           zorder=3, alpha=0.9)
                except:
                    pass
        
        ax.set_xlabel('Age (months)', fontsize=12)
        ax.set_ylabel('Total Height (cm)', fontsize=12)
        ax.set_title(f'{title}\n(n={len(fits_subset)} animals)', 
                    fontsize=13, fontweight='bold')
        ax.set_xlim(0, 240)
        ax.set_ylim(0, 1000)
        if len(fits_subset) > 0:
            ax.legend(fontsize=10, loc='lower right')
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Individual Growth Curves by Sex (≥{min_obs} observations per animal)',
                fontsize=15, fontweight='bold')
    
    fig_path = output_dir / f'individual_curves_grid_min{min_obs}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved individual curves grid to {fig_path}")
    plt.close()


def main():
    # Setup paths
    outputs_dir = Path("Outputs/individual_level_analysis")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PLOTTING INDIVIDUAL GROWTH CURVES")
    print(f"{'='*70}")
    
    # Load data
    print("\nLoading data...")
    wild = load_prepare_wild(Path("Data/wild.csv"))
    wild = add_age_class_midpoints(wild)
    wild = assign_initial_ages_from_classes(wild)
    wild = add_vtb_umb_flag(wild)
    
    # Load individual fits
    individual_fits_path = outputs_dir / "individual_fits_y0free.csv"
    if not individual_fits_path.exists():
        print(f"ERROR: Individual fits not found at {individual_fits_path}")
        print("Please run growth_analyses_individual_level.py first.")
        return
    
    individual_fits = pd.read_csv(individual_fits_path)
    print(f"Loaded {len(individual_fits)} individual fits")
    
    # Create plots for different minimum observation thresholds
    for min_obs in [3, 4, 5]:
        n_animals = (individual_fits['n_obs'] >= min_obs).sum()
        if n_animals > 0:
            print(f"\n--- Plotting animals with ≥{min_obs} observations (n={n_animals}) ---")
            
            # Overall plot
            plot_all_individual_curves(wild, individual_fits, outputs_dir, 
                                      min_obs=min_obs, sex_filter=None)
            
            # Grid plot (Overall, M, F)
            plot_individual_curves_grid(wild, individual_fits, outputs_dir, 
                                       min_obs=min_obs)
    
    print(f"\n{'='*70}")
    print(f"PLOTTING COMPLETE")
    print(f"{'='*70}")
    print(f"Outputs saved to: {outputs_dir}")


if __name__ == "__main__":
    main()
