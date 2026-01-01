import numpy as np
from dataclasses import dataclass
from typing import Callable, Sequence
from scipy.optimize import curve_fit

# Birth height constraint from literature and zoo data
BIRTH_HEIGHT_CM = 180.0  # Mean birth height in cm
BIRTH_HEIGHT_SD = 17.5   # Standard deviation


@dataclass(frozen=True)
class GrowthModel:
    name: str
    func: Callable[..., np.ndarray]
    initial_guess: Callable[[np.ndarray, np.ndarray], Sequence[float]]
    param_names: tuple[str, ...]
    bounds: tuple[np.ndarray, np.ndarray] | None = None


@dataclass
class FitResult:
    model: GrowthModel
    params: np.ndarray
    sse: float
    aic: float
    n_obs: int
    success: bool
    message: str = ""
    covariance: np.ndarray | None = None

    def predict(self, t: np.ndarray) -> np.ndarray:
        return self.model.func(t, *self.params) if self.success else np.full_like(t, np.nan)

    @property
    def param_pairs(self) -> list[tuple[str, float]]:
        return list(zip(self.model.param_names, self.params))


# Curve functions
def gompertz_model(t: np.ndarray, A: float, k: float, t0: float) -> np.ndarray:
    return A * np.exp(-np.exp(-k * (t - t0)))


def logistic_model(t: np.ndarray, A: float, k: float, t0: float) -> np.ndarray:
    return A / (1.0 + np.exp(-k * (t - t0)))


def von_bertalanffy_model(t: np.ndarray, A: float, k: float, t0: float) -> np.ndarray:
    return np.maximum(A * (1.0 - np.exp(-k * np.maximum(t - t0, 0.0))) ** 3, 0.0)


def richards_model(t: np.ndarray, A: float, k: float, t0: float, nu: float) -> np.ndarray:
    return A / (1.0 + nu * np.exp(-k * (t - t0))) ** (1.0 / nu)


def poly3_model(t: np.ndarray, a3: float, a2: float, a1: float, a0: float) -> np.ndarray:
    return ((a3 * t + a2) * t + a1) * t + a0


def poly4_model(t: np.ndarray, a4: float, a3: float, a2: float, a1: float, a0: float) -> np.ndarray:
    return (((a4 * t + a3) * t + a2) * t + a1) * t + a0


# =============================================================================
# BIRTH-HEIGHT CONSTRAINED MODEL FUNCTIONS (Option 1: Reparameterization)
# =============================================================================
# These models have birth height (y0) as an explicit parameter instead of t0.
# The t0 parameter is derived internally from the constraint: model(t=0) = y0
# This forces the curve to pass through the specified birth height at age 0.

def _derive_t0_gompertz(A: float, k: float, y0: float) -> float:
    """Derive t0 from birth height constraint for Gompertz model.
    
    From: y0 = A * exp(-exp(k*t0))
    Solve: t0 = (1/k) * ln(-ln(y0/A))
    """
    if k <= 1e-10 or A <= 0 or y0 <= 0:
        return 0.0
    ratio = np.clip(y0 / A, 1e-10, 1.0 - 1e-10)
    inner = -np.log(ratio)
    if inner <= 0:
        return 0.0
    return np.log(inner) / k


def _derive_t0_logistic(A: float, k: float, y0: float) -> float:
    """Derive t0 from birth height constraint for Logistic model.
    
    From: y0 = A / (1 + exp(k*t0))
    Solve: t0 = (1/k) * ln(A/y0 - 1)
    """
    if k <= 1e-10 or A <= 0 or y0 <= 0:
        return 0.0
    ratio = A / np.clip(y0, 1e-10, A - 1e-10) - 1.0
    if ratio <= 0:
        return 0.0
    return np.log(ratio) / k


def _derive_t0_von_bertalanffy(A: float, k: float, y0: float) -> float:
    """Derive t0 from birth height constraint for Von Bertalanffy model.
    
    From: y0 = A * (1 - exp(k*t0))^3
    Solve: t0 = (1/k) * ln(1 - (y0/A)^(1/3))
    """
    if k <= 1e-10 or A <= 0 or y0 <= 0:
        return 0.0
    ratio = np.clip(y0 / A, 1e-10, 1.0 - 1e-10)
    inner = 1.0 - ratio ** (1.0 / 3.0)
    if inner <= 0:
        return 0.0
    return np.log(inner) / k


def _derive_t0_richards(A: float, k: float, y0: float, nu: float) -> float:
    """Derive t0 from birth height constraint for Richards model.
    
    From: y0 = A / (1 + nu * exp(k*t0))^(1/nu)
    Solve: t0 = (1/k) * ln((1/nu) * ((A/y0)^nu - 1))
    """
    if k <= 1e-10 or A <= 0 or y0 <= 0 or nu <= 0:
        return 0.0
    ratio = np.clip(y0 / A, 1e-10, 1.0 - 1e-10)
    inner = (1.0 / nu) * ((1.0 / ratio) ** nu - 1.0)
    if inner <= 0:
        return 0.0
    return np.log(inner) / k


def gompertz_constrained(t: np.ndarray, A: float, k: float, y0: float) -> np.ndarray:
    """Gompertz model with explicit birth height y0 (replaces t0 parameter)."""
    t0 = _derive_t0_gompertz(A, k, y0)
    return A * np.exp(-np.exp(-k * (t - t0)))


def logistic_constrained(t: np.ndarray, A: float, k: float, y0: float) -> np.ndarray:
    """Logistic model with explicit birth height y0 (replaces t0 parameter)."""
    t0 = _derive_t0_logistic(A, k, y0)
    return A / (1.0 + np.exp(-k * (t - t0)))


def von_bertalanffy_constrained(t: np.ndarray, A: float, k: float, y0: float) -> np.ndarray:
    """Von Bertalanffy model with explicit birth height y0 (replaces t0 parameter)."""
    t0 = _derive_t0_von_bertalanffy(A, k, y0)
    return np.maximum(A * (1.0 - np.exp(-k * np.maximum(t - t0, 0.0))) ** 3, 0.0)


def richards_constrained(t: np.ndarray, A: float, k: float, y0: float, nu: float) -> np.ndarray:
    """Richards model with explicit birth height y0 (replaces t0 parameter)."""
    t0 = _derive_t0_richards(A, k, y0, nu)
    return A / (1.0 + nu * np.exp(-k * (t - t0))) ** (1.0 / nu)


def poly3_constrained(t: np.ndarray, a3: float, a2: float, a1: float, y0: float) -> np.ndarray:
    """Cubic polynomial with birth height y0 as the intercept (a0 = y0)."""
    return ((a3 * t + a2) * t + a1) * t + y0


def poly4_constrained(t: np.ndarray, a4: float, a3: float, a2: float, a1: float, y0: float) -> np.ndarray:
    """Quartic polynomial with birth height y0 as the intercept (a0 = y0)."""
    return (((a4 * t + a3) * t + a2) * t + a1) * t + y0


# Initial guesses for constrained models (y0 fixed or bounded around BIRTH_HEIGHT_CM)
def _gompertz_constrained_initial_guess(t: np.ndarray, y: np.ndarray):
    return (_max_or_one(y) * 1.05, 0.05, BIRTH_HEIGHT_CM)


def _logistic_constrained_initial_guess(t: np.ndarray, y: np.ndarray):
    return (_max_or_one(y) * 1.05, 0.05, BIRTH_HEIGHT_CM)


def _bertalanffy_constrained_initial_guess(t: np.ndarray, y: np.ndarray):
    return (_max_or_one(y) * 1.05, 0.03, BIRTH_HEIGHT_CM)


def _richards_constrained_initial_guess(t: np.ndarray, y: np.ndarray):
    return (_max_or_one(y) * 1.05, 0.05, BIRTH_HEIGHT_CM, 1.0)


def _poly3_constrained_initial_guess(t: np.ndarray, y: np.ndarray):
    if len(t) >= 4 and not np.isnan(y).all():
        # Fit with constraint that intercept = BIRTH_HEIGHT_CM
        # Use regular polyfit then adjust
        coeffs = np.polyfit(t, y, 3)
        return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), BIRTH_HEIGHT_CM)
    return (0.0, 0.0, 0.0, BIRTH_HEIGHT_CM)


def _poly4_constrained_initial_guess(t: np.ndarray, y: np.ndarray):
    if len(t) >= 5 and not np.isnan(y).all():
        coeffs = np.polyfit(t, y, 4)
        return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3]), BIRTH_HEIGHT_CM)
    return (0.0, 0.0, 0.0, 0.0, BIRTH_HEIGHT_CM)


def _max_or_one(values: np.ndarray) -> float:
    return float(np.nanmax(values)) if len(values) else 1.0


def _gompertz_initial_guess(t: np.ndarray, y: np.ndarray):
    return (_max_or_one(y) * 1.05, 0.05, float(np.nanmedian(t)) if len(t) else 0.0)


def _logistic_initial_guess(t: np.ndarray, y: np.ndarray):
    return (_max_or_one(y) * 1.05, 0.05, float(np.nanmedian(t)) if len(t) else 0.0)


def _bertalanffy_initial_guess(t: np.ndarray, y: np.ndarray):
    return (_max_or_one(y) * 1.05, 0.03, max(float(np.nanmedian(t)), 0.1) if len(t) else 0.1)


def _richards_initial_guess(t: np.ndarray, y: np.ndarray):
    return (_max_or_one(y) * 1.05, 0.05, float(np.nanmedian(t)) if len(t) else 0.0, 1.0)


def _poly3_initial_guess(t: np.ndarray, y: np.ndarray):
    if len(t) >= 4 and not np.isnan(y).all():
        coeffs = np.polyfit(t, y, 3)
        return tuple(map(float, coeffs))
    mean_y = float(np.nanmean(y)) if len(y) else 0.0
    return (0.0, 0.0, 0.0, mean_y)


def _poly4_initial_guess(t: np.ndarray, y: np.ndarray):
    if len(t) >= 5 and not np.isnan(y).all():
        coeffs = np.polyfit(t, y, 4)
        return tuple(map(float, coeffs))
    mean_y = float(np.nanmean(y)) if len(y) else 0.0
    return (0.0, 0.0, 0.0, 0.0, mean_y)


AVAILABLE_MODELS: dict[str, GrowthModel] = {
    "gompertz": GrowthModel(
        "gompertz",
        gompertz_model,
        _gompertz_initial_guess,
        ("A", "k", "t0"),
        bounds=(np.array([0.0, 0.0, -np.inf]), np.array([np.inf, np.inf, np.inf])),
    ),
    "logistic": GrowthModel(
        "logistic",
        logistic_model,
        _logistic_initial_guess,
        ("A", "k", "t0"),
        bounds=(np.array([0.0, 0.0, -np.inf]), np.array([np.inf, np.inf, np.inf])),
    ),
    "von_bertalanffy": GrowthModel(
        "von_bertalanffy",
        von_bertalanffy_model,
        _bertalanffy_initial_guess,
        ("A", "k", "t0"),
        bounds=(np.array([0.0, 0.0, -np.inf]), np.array([np.inf, np.inf, np.inf])),
    ),
    "richards": GrowthModel(
        "richards",
        richards_model,
        _richards_initial_guess,
        ("A", "k", "t0", "nu"),
        bounds=(
            np.array([0.0, 0.0, -np.inf, 0.01]),
            np.array([np.inf, np.inf, np.inf, 100.0]),
        ),
    ),
    "poly3": GrowthModel(
        "poly3",
        poly3_model,
        _poly3_initial_guess,
        ("a3", "a2", "a1", "a0"),
    ),
    "poly4": GrowthModel(
        "poly4",
        poly4_model,
        _poly4_initial_guess,
        ("a4", "a3", "a2", "a1", "a0"),
    ),
}


DEFAULT_MODEL_SEQUENCE: tuple[str, ...] = (
    "gompertz",
    "logistic",
    "von_bertalanffy",
    "richards",
    "poly3",
    "poly4",
)


JUVENILE_MODEL_SEQUENCE: tuple[str, ...] = (
    "gompertz",
    "logistic",
    "von_bertalanffy",
    "richards",
    "poly3",
    "poly4",
)


# =============================================================================
# BIRTH-HEIGHT CONSTRAINED MODELS DICTIONARY
# =============================================================================
# Use these models when you want to force the growth curve to pass through
# a specific birth height (default: 180 cm from literature/zoo data).
# The y0 parameter can be fixed exactly at BIRTH_HEIGHT_CM or bounded.

# Bounds for y0: allow fitting within ±2 SD of mean birth height
_Y0_LOWER = BIRTH_HEIGHT_CM - 2 * BIRTH_HEIGHT_SD  # 145 cm
_Y0_UPPER = BIRTH_HEIGHT_CM + 2 * BIRTH_HEIGHT_SD  # 215 cm

CONSTRAINED_MODELS: dict[str, GrowthModel] = {
    "gompertz_constrained": GrowthModel(
        "gompertz_constrained",
        gompertz_constrained,
        _gompertz_constrained_initial_guess,
        ("A", "k", "y0"),
        bounds=(np.array([0.0, 0.0, _Y0_LOWER]), np.array([np.inf, np.inf, _Y0_UPPER])),
    ),
    "logistic_constrained": GrowthModel(
        "logistic_constrained",
        logistic_constrained,
        _logistic_constrained_initial_guess,
        ("A", "k", "y0"),
        bounds=(np.array([0.0, 0.0, _Y0_LOWER]), np.array([np.inf, np.inf, _Y0_UPPER])),
    ),
    "von_bertalanffy_constrained": GrowthModel(
        "von_bertalanffy_constrained",
        von_bertalanffy_constrained,
        _bertalanffy_constrained_initial_guess,
        ("A", "k", "y0"),
        bounds=(np.array([0.0, 0.0, _Y0_LOWER]), np.array([np.inf, np.inf, _Y0_UPPER])),
    ),
    "richards_constrained": GrowthModel(
        "richards_constrained",
        richards_constrained,
        _richards_constrained_initial_guess,
        ("A", "k", "y0", "nu"),
        bounds=(
            np.array([0.0, 0.0, _Y0_LOWER, 0.01]),
            np.array([np.inf, np.inf, _Y0_UPPER, 100.0]),
        ),
    ),
    "poly3_constrained": GrowthModel(
        "poly3_constrained",
        poly3_constrained,
        _poly3_constrained_initial_guess,
        ("a3", "a2", "a1", "y0"),
        bounds=(
            np.array([-np.inf, -np.inf, -np.inf, _Y0_LOWER]),
            np.array([np.inf, np.inf, np.inf, _Y0_UPPER]),
        ),
    ),
    "poly4_constrained": GrowthModel(
        "poly4_constrained",
        poly4_constrained,
        _poly4_constrained_initial_guess,
        ("a4", "a3", "a2", "a1", "y0"),
        bounds=(
            np.array([-np.inf, -np.inf, -np.inf, -np.inf, _Y0_LOWER]),
            np.array([np.inf, np.inf, np.inf, np.inf, _Y0_UPPER]),
        ),
    ),
}

# Models with y0 fixed exactly at BIRTH_HEIGHT_CM (no flexibility)
FIXED_BIRTH_HEIGHT_MODELS: dict[str, GrowthModel] = {
    "gompertz_fixed_y0": GrowthModel(
        "gompertz_fixed_y0",
        gompertz_constrained,
        _gompertz_constrained_initial_guess,
        ("A", "k", "y0"),
        bounds=(np.array([0.0, 0.0, BIRTH_HEIGHT_CM - 0.01]), np.array([np.inf, np.inf, BIRTH_HEIGHT_CM + 0.01])),
    ),
    "logistic_fixed_y0": GrowthModel(
        "logistic_fixed_y0",
        logistic_constrained,
        _logistic_constrained_initial_guess,
        ("A", "k", "y0"),
        bounds=(np.array([0.0, 0.0, BIRTH_HEIGHT_CM - 0.01]), np.array([np.inf, np.inf, BIRTH_HEIGHT_CM + 0.01])),
    ),
    "von_bertalanffy_fixed_y0": GrowthModel(
        "von_bertalanffy_fixed_y0",
        von_bertalanffy_constrained,
        _bertalanffy_constrained_initial_guess,
        ("A", "k", "y0"),
        bounds=(np.array([0.0, 0.0, BIRTH_HEIGHT_CM - 0.01]), np.array([np.inf, np.inf, BIRTH_HEIGHT_CM + 0.01])),
    ),
    "richards_fixed_y0": GrowthModel(
        "richards_fixed_y0",
        richards_constrained,
        _richards_constrained_initial_guess,
        ("A", "k", "y0", "nu"),
        bounds=(
            np.array([0.0, 0.0, BIRTH_HEIGHT_CM - 0.01, 0.01]),
            np.array([np.inf, np.inf, BIRTH_HEIGHT_CM + 0.01, 100.0]),
        ),
    ),
    "poly3_fixed_y0": GrowthModel(
        "poly3_fixed_y0",
        poly3_constrained,
        _poly3_constrained_initial_guess,
        ("a3", "a2", "a1", "y0"),
        bounds=(
            np.array([-np.inf, -np.inf, -np.inf, BIRTH_HEIGHT_CM - 0.01]),
            np.array([np.inf, np.inf, np.inf, BIRTH_HEIGHT_CM + 0.01]),
        ),
    ),
    "poly4_fixed_y0": GrowthModel(
        "poly4_fixed_y0",
        poly4_constrained,
        _poly4_constrained_initial_guess,
        ("a4", "a3", "a2", "a1", "y0"),
        bounds=(
            np.array([-np.inf, -np.inf, -np.inf, -np.inf, BIRTH_HEIGHT_CM - 0.01]),
            np.array([np.inf, np.inf, np.inf, np.inf, BIRTH_HEIGHT_CM + 0.01]),
        ),
    ),
}

# Model sequences for constrained fitting
CONSTRAINED_MODEL_SEQUENCE: tuple[str, ...] = (
    "gompertz_constrained",
    "logistic_constrained",
    "von_bertalanffy_constrained",
    "richards_constrained",
    "poly3_constrained",
    "poly4_constrained",
)

FIXED_Y0_MODEL_SEQUENCE: tuple[str, ...] = (
    "gompertz_fixed_y0",
    "logistic_fixed_y0",
    "von_bertalanffy_fixed_y0",
    "richards_fixed_y0",
    "poly3_fixed_y0",
    "poly4_fixed_y0",
)

# Combined dictionary of all available models
ALL_MODELS: dict[str, GrowthModel] = {
    **AVAILABLE_MODELS,
    **CONSTRAINED_MODELS,
    **FIXED_BIRTH_HEIGHT_MODELS,
}


def fit_single_model(model: GrowthModel, t: np.ndarray, y: np.ndarray) -> FitResult:
    # Handle polynomial models with direct polyfit (faster and more stable)
    if model.name == "poly3":
        try:
            if len(t) < 4:
                raise ValueError("Need at least 4 points to fit poly3")
            coeffs = np.polyfit(t, y, 3)
            popt = np.asarray(coeffs, dtype=float)
            residuals = y - poly3_model(t, *popt)
            sse = float(np.sum(residuals ** 2))
            n = len(y)
            k = len(popt)
            sse_safe = max(sse, 1e-9)
            aic = n * np.log(sse_safe / n) + 2 * k if n > k else np.inf
            return FitResult(model, popt, sse, aic, n, True, "", None)
        except Exception as exc:
            return FitResult(model, np.array([]), np.inf, np.inf, len(y), False, str(exc))

    if model.name == "poly4":
        try:
            if len(t) < 5:
                raise ValueError("Need at least 5 points to fit poly4")
            coeffs = np.polyfit(t, y, 4)
            popt = np.asarray(coeffs, dtype=float)
            residuals = y - poly4_model(t, *popt)
            sse = float(np.sum(residuals ** 2))
            n = len(y)
            k = len(popt)
            sse_safe = max(sse, 1e-9)
            aic = n * np.log(sse_safe / n) + 2 * k if n > k else np.inf
            return FitResult(model, popt, sse, aic, n, True, "", None)
        except Exception as exc:
            return FitResult(model, np.array([]), np.inf, np.inf, len(y), False, str(exc))

    try:
        p0 = model.initial_guess(t, y)
        bounds = model.bounds if model.bounds is not None else (-np.inf, np.inf)
        popt, pcov = curve_fit(
            model.func,
            t,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=50000,
        )
        residuals = y - model.func(t, *popt)
        sse = float(np.sum(residuals ** 2))
        n = len(y)
        k = len(popt)
        sse_safe = max(sse, 1e-9)
        aic = n * np.log(sse_safe / n) + 2 * k if n > k else np.inf
        return FitResult(model, np.asarray(popt, dtype=float), sse, aic, n, True, "", pcov)
    except Exception as exc:
        return FitResult(model, np.array([]), np.inf, np.inf, len(y), False, str(exc))


def select_best_model(
    t: np.ndarray,
    y: np.ndarray,
    model_names: Sequence[str] = DEFAULT_MODEL_SEQUENCE,
) -> tuple[FitResult, list[FitResult]]:
    # Look up models from ALL_MODELS (includes unconstrained, constrained, and fixed y0)
    candidates = [ALL_MODELS[name] for name in model_names if name in ALL_MODELS]
    if not candidates:
        raise ValueError("No valid model names supplied for fitting")

    results = [fit_single_model(model, t, y) for model in candidates]
    successes = [res for res in results if res.success]

    if successes:
        successes.sort(key=lambda res: res.aic)
        best = successes[0]

        gompertz_fit = next((res for res in successes if res.model.name == "gompertz" and np.isfinite(res.aic)), None)
        if gompertz_fit is not None and best.model.name != "gompertz" and np.isfinite(best.aic):
            baseline = abs(gompertz_fit.aic)
            tolerance = 0.01 * baseline if baseline > 1e-12 else 0.01
            diff = gompertz_fit.aic - best.aic
            if 0.0 <= diff <= tolerance:
                best = gompertz_fit
    else:
        results.sort(key=lambda res: res.sse)
        best = results[0]

    return best, results
