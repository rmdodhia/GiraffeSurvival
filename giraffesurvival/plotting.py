from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .models import FitResult


def _sanitize_for_filename(label) -> str:
    label = "" if label is None else str(label)
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def _strip_label_prefix(label: str) -> str:
    if label is None:
        return ""
    label_lower = label.lower()
    for prefix in ("params_", "df_", "data_"):
        if label_lower.startswith(prefix):
            return label[len(prefix):]
    return label


# Fixed y-axis ranges per measurement for consistency across plots
_DEFAULT_Y_LIMITS: dict[str, tuple[float, float]] = {
    "TH": (0.0, 1100.0),
    "avg TOO_TOHcm": (0.0, 325.0),
    "avg TOH_NIcm": (0.0, 500.0),
    "avg NI_FBHcm": (0.0, 550.0),
    # Optional: zoo juvenile height if plotted
    "height_cm": (0.0, 500.0),
}


def _apply_default_ylim(ax, value_col: str, y_limits: tuple[float, float] | None = None) -> None:
    if y_limits is not None:
        ax.set_ylim(*y_limits)
        return
    lim = _DEFAULT_Y_LIMITS.get(value_col)
    if lim:
        ax.set_ylim(*lim)


def plot_growth_curve_overall(
    data: pd.DataFrame,
    fits_overall: dict[str, FitResult],
    value_col: str,
    title: str,
    age_max_months: float = 240.0,
    df_label=None,
    y_limits: tuple[float, float] | None = None,
    graph_dir: Path | None = None,
):
    fit = fits_overall.get("overall") if fits_overall else None
    if fit is None or not fit.success:
        print("PLOT CANCELED: overall fit missing or failed.")
        return

    subset = data.dropna(subset=["age_months", value_col]).copy()
    if subset.empty:
        print(f"No data for {value_col}")
        return

    t = subset["age_months"].to_numpy(dtype=float)
    y = subset[value_col].to_numpy(dtype=float)
    y_hat = fit.predict(t)
    residuals = y - y_hat
    dof = max(len(y) - len(fit.params), 1)
    sigma = float(np.sqrt(np.sum(residuals ** 2) / dof))

    t_grid = np.linspace(max(0.1, t.min()), age_max_months, 300)
    curve = fit.predict(t_grid)
    z = 1.96
    lower = curve - z * sigma
    upper = curve + z * sigma

    plt.figure(figsize=(7, 5))
    plt.scatter(t / 12.0, y, alpha=0.3, s=10, label="Data")
    plt.fill_between(t_grid / 12.0, lower, upper, alpha=0.25, color="tab:orange", edgecolor="none", label="95% prediction interval")
    plt.plot(t_grid / 12.0, curve, linewidth=2, color="black", label=f"{fit.model.name} overall")

    plt.xlabel("Age (years)")
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend()
    _apply_default_ylim(plt.gca(), value_col, y_limits)
    plt.tight_layout()

    attrs = getattr(data, "attrs", {})
    if not isinstance(attrs, dict):
        attrs = {}
    inferred_label = _strip_label_prefix(
        df_label or attrs.get("label") or attrs.get("name") or getattr(data, "name", None) or "data"
    )

    label_part = _sanitize_for_filename(inferred_label)
    if not label_part or label_part == "data":
        label_part = _sanitize_for_filename(value_col) or "value"

    output_dir = graph_dir or Path("Graph")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{label_part}_overall.png"
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_growth_curve_by_sex(
    data: pd.DataFrame,
    fits_by_sex: dict[str, FitResult],
    value_col: str,
    title: str,
    age_max_months: float = 240.0,
    df_label=None,
    y_limits: tuple[float, float] | None = None,
    graph_dir: Path | None = None,
):
    if not fits_by_sex:
        print("PLOT CANCELED: No sex-specific fits provided.")
        return

    plt.figure(figsize=(7, 5))
    plotted_any = False

    for sex, color in zip(["M", "F"], ["tab:blue", "tab:orange"]):
        subset = data[data["Sex"] == sex].dropna(subset=["age_months", value_col])
        if not subset.empty:
            plt.scatter(subset["age_months"] / 12.0, subset[value_col], alpha=0.3, s=10, label=f"Data {sex}", color=color)
            plotted_any = True
        fit = fits_by_sex.get(sex)
        if fit and fit.success:
            t_grid = np.linspace(max(0.1, subset["age_months"].min() if not subset.empty else 0.1), age_max_months, 300)
            curve = fit.predict(t_grid)
            plt.plot(t_grid / 12.0, curve, linewidth=2, label=f"{fit.model.name} {sex}", color=color)
            plotted_any = True

    plt.xlabel("Age (years)")
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend()
    _apply_default_ylim(plt.gca(), value_col, y_limits)
    plt.tight_layout()

    if plotted_any:
        attrs = getattr(data, "attrs", {})
        if not isinstance(attrs, dict):
            attrs = {}
        inferred_label = _strip_label_prefix(
            df_label or attrs.get("label") or attrs.get("name") or getattr(data, "name", None) or "data"
        )
        label_part = _sanitize_for_filename(inferred_label)
        if not label_part or label_part == "data":
            label_part = _sanitize_for_filename(value_col) or "value"
        output_dir = graph_dir or Path("Graph")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{label_part}_by_sex.png"
        plt.savefig(filename, dpi=300)
    plt.close()


def plot_growth_curve_by_group(
    data: pd.DataFrame,
    fits_by_group: dict[str, FitResult],
    group_col: str,
    value_col: str,
    title: str,
    age_max_months: float = 240.0,
    df_label=None,
    y_limits: tuple[float, float] | None = None,
    graph_dir: Path | None = None,
):
    if not fits_by_group:
        print(f"PLOT CANCELED: No fits provided for group {group_col}.")
        return

    plt.figure(figsize=(7, 5))
    plotted_any = False
    unique_groups = sorted(fits_by_group.keys())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    for idx, grp in enumerate(unique_groups):
        color = colors[idx % len(colors)]
        subset = data[data[group_col] == grp].dropna(subset=["age_months", value_col])
        if not subset.empty:
            plt.scatter(subset["age_months"] / 12.0, subset[value_col], alpha=0.3, s=10, label=f"Data {grp}", color=color)
            plotted_any = True
        fit = fits_by_group.get(grp)
        if fit and fit.success:
            t_min = max(0.1, subset["age_months"].min() if not subset.empty else 0.1)
            t_grid = np.linspace(t_min, age_max_months, 300)
            curve = fit.predict(t_grid)
            plt.plot(t_grid / 12.0, curve, linewidth=2, label=f"{fit.model.name} {grp}", color=color)
            plotted_any = True

    plt.xlabel("Age (years)")
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend()
    _apply_default_ylim(plt.gca(), value_col, y_limits)
    plt.tight_layout()

    if plotted_any:
        attrs = getattr(data, "attrs", {})
        if not isinstance(attrs, dict):
            attrs = {}
        inferred_label = _strip_label_prefix(
            df_label or attrs.get("label") or attrs.get("name") or getattr(data, "name", None) or "data"
        )
        label_part = _sanitize_for_filename(inferred_label)
        if not label_part or label_part == "data":
            label_part = _sanitize_for_filename(value_col) or "value"
        output_dir = graph_dir or Path("Graph")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{label_part}_by_{_sanitize_for_filename(group_col)}.png"
        plt.savefig(filename, dpi=300)
    plt.close()


def plot_individual_trajectories(
    data: pd.DataFrame,
    group_col: str,
    group_value: str,
    value_col: str,
    overlay_fit: FitResult | None = None,
    out_dir: Path | None = None,
    y_limits: tuple[float, float] | None = None,
    graph_dir: Path | None = None,
):
    """Save per-individual trajectories for a filtered subset.

    Creates one PNG per `AID` for records matching `group_col == group_value`,
    plotting `age_months` vs `value_col` with points + connecting line. Optionally
    overlays the provided group-level `overlay_fit` curve.
    """
    if out_dir is None:
        # Make symbolic operators readable and distinct in folder names
        def _safe_token(s: str) -> str:
            s = s.replace(">", "_gt_").replace("=", "_eq_")
            return _sanitize_for_filename(s)
        root = graph_dir or Path("Graph")
        out_dir = root / f"{_sanitize_for_filename(value_col)}_{_sanitize_for_filename(group_col)}_{_safe_token(group_value)}_individuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    subset = data[(data[group_col] == group_value)].dropna(subset=["age_months", value_col])
    if subset.empty:
        print(f"No data for {value_col} with {group_col}={group_value} to plot individuals.")
        return

    # Precompute overlay curve grid once
    t_grid = None
    overlay_curve = None
    if overlay_fit and overlay_fit.success:
        t_min = max(0.1, subset["age_months"].min())
        t_max = max(subset["age_months"].max(), 240.0)
        t_grid = np.linspace(t_min, t_max, 300)
        overlay_curve = overlay_fit.predict(t_grid)

    for aid, grp in subset.groupby("AID"):
        if grp.empty:
            continue
        plt.figure(figsize=(7, 5))
        # Sort by date to connect points in order
        grp = grp.sort_values("SurveyDate")
        t = grp["age_months"].to_numpy(dtype=float)
        y = grp[value_col].to_numpy(dtype=float)
        plt.plot(t / 12.0, y, marker="o", linestyle="-", linewidth=1.0, markersize=3, alpha=0.8)

        if t_grid is not None and overlay_curve is not None:
            plt.plot(t_grid / 12.0, overlay_curve, color="black", linewidth=1.5, alpha=0.6, label="group fit")
            plt.legend()

        plt.xlabel("Age (years)")
        plt.ylabel(value_col)
        plt.title(f"{value_col} — AID {aid} ({group_col}={group_value})")
        _apply_default_ylim(plt.gca(), value_col, y_limits)
        plt.tight_layout()

        # Prefer integer AID in file names when possible
        aid_token = str(aid)
        try:
            aid_token = str(int(float(aid)))
        except Exception:
            aid_token = str(aid)
        filename = out_dir / f"AID_{_sanitize_for_filename(aid_token)}.png"
        plt.savefig(filename, dpi=200)
        plt.close()
