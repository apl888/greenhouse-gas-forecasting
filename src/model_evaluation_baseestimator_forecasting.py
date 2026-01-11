from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

# === 1. Base forecaster class ===

class BaseForecaster(ABC):
    """Abstract base class for all forecasters"""
    
    def __init__(self, name: str):
        self.name = name
        self.fitted_model = None
        self.train_data = None
        
    @abstractmethod
    def fit(self, train_data: pd.Series, **kwargs):
        """Fit the model to training data"""
        pass
    
    @abstractmethod
    def predict(self, steps: int, **kwargs) -> pd.Series:
        """Generate point forecasts"""
        pass
    
    @abstractmethod
    def predict_interval(self, steps: int, coverage: float = 0.95, **kwargs) -> pd.DataFrame:
        """Generate prediction intervals"""
        pass
    
    @abstractmethod
    def get_residuals(self) -> pd.Series:
        """Get in-sample residuals"""
        pass
    
    def get_model_summary(self) -> dict:
        """Return model parameters and diagnostics"""
        return {"name": self.name}
    
# === 2. SARIMAX forecaster ===
    
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import BaseForecaster

class SARIMAForecaster(BaseForecaster):
    def __init__(self, order, seasonal_order, **kwargs):
        super().__init__(name=f"SARIMA{order}{seasonal_order}")
        self.order = order
        self.seasonal_order = seasonal_order
        self.kwargs = kwargs
        
    def fit(self, train_data, **fit_kwargs):
        self.train_data = train_data
        self.fitted_model = SARIMAX(
            train_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            **self.kwargs
        ).fit(**fit_kwargs)
        return self
    
    def predict(self, steps, **kwargs):
        forecast = self.fitted_model.get_forecast(steps=steps, **kwargs)
        return forecast.predicted_mean
    
    def predict_interval(self, steps, coverage=0.95, **kwargs):
        forecast = self.fitted_model.get_forecast(steps=steps, **kwargs)
        alpha = 1 - coverage
        return forecast.conf_int(alpha=alpha)
    
    def get_residuals(self):
        return self.fitted_model.resid
    
    def get_model_summary(self):
        summary = super().get_model_summary()
        summary.update({
            "aic": self.fitted_model.aic,
            "bic": self.fitted_model.bic,
            "params": self.fitted_model.params,
            "converged": self.fitted_model.mle_retvals.get("converged", False)
        })
        return summary
# === 3. ETS Forecaster ===

# === 4. TBATS Forecaster ===
from sktime.forecasting.tbats import TBATS
from .base import BaseForecaster

class TBATSForecaster(BaseForecaster):
    def __init__(self, use_box_cox=True, use_trend=True, use_damped_trend=False,
                 sp=[52.178], use_arma_errors=True, n_jobs=-1, **kwargs):
        super().__init__(name="TBATS")
        self.model_params = {
            "use_box_cox": use_box_cox,
            "use_trend": use_trend,
            "use_damped_trend": use_damped_trend,
            "sp": sp,
            "use_arma_errors": use_arma_errors,
            "n_jobs": n_jobs,
            **kwargs
        }
        
    def fit(self, train_data, **fit_kwargs):
        self.train_data = train_data
        self.model = TBATS(**self.model_params)
        self.fitted_model = self.model.fit(train_data, **fit_kwargs)
        return self
    
    def predict(self, steps, **kwargs):
        horizon = np.arange(1, steps + 1)
        return self.fitted_model.predict(horizon, **kwargs)
    
    def predict_interval(self, steps, coverage=0.95, **kwargs):
        horizon = np.arange(1, steps + 1)
        return self.fitted_model.predict_interval(horizon, coverage=coverage, **kwargs)
    
    def get_residuals(self):
        # TBATS doesn't store residuals directly - compute from fitted values
        fitted_values = self.fitted_model.predict(np.arange(1, len(self.train_data) + 1))
        return self.train_data - fitted_values
    
    def get_model_summary(self):
        summary = super().get_model_summary()
        if hasattr(self.fitted_model, '_forecaster'):
            tbats_model = self.fitted_model._forecaster
            summary.update({
                "aic": tbats_model.aic,
                "parameters": {
                    "box_cox_lambda": tbats_model.params.box_cox.lambda_,
                    "alpha": tbats_model.params.alpha,
                    "beta": tbats_model.params.beta,
                    "seasonal_harmonics": tbats_model.params.components.seasonal_harmonics,
                    "arma_order": tbats_model.params.arma.order
                }
            })
        return summary
    
# === 5. Unified evaluation framework ===
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import warnings

class ModelEvaluator:
    """Unified evaluator for multiple forecasting models"""
    
    def __init__(self, seasonal_period: int = 52):
        self.seasonal_period = seasonal_period
        self.results = {}
        
    def evaluate_single_model(
        self,
        forecaster: BaseForecaster,
        train_data: pd.Series,
        test_data: pd.Series,
        model_name: Optional[str] = None
    ) -> Dict:
        """Evaluate a single model on train/test split"""
        
        if model_name is None:
            model_name = forecaster.name
            
        # Fit model
        forecaster.fit(train_data)
        
        # Generate predictions
        steps = len(test_data)
        point_forecast = forecaster.predict(steps)
        
        # Generate intervals if available
        try:
            intervals = forecaster.predict_interval(steps, coverage=0.95)
            coverage = self._calculate_coverage(test_data, intervals)
        except:
            intervals = None
            coverage = None
            
        # Calculate metrics
        metrics = self._calculate_metrics(
            test_data, point_forecast, train_data, intervals
        )
        
        # Store results
        self.results[model_name] = {
            "forecaster": forecaster,
            "point_forecast": point_forecast,
            "intervals": intervals,
            "metrics": metrics,
            "model_summary": forecaster.get_model_summary()
        }
        
        return self.results[model_name]
    
    def cross_validate(
        self,
        forecasters: List[BaseForecaster],
        data: pd.Series,
        n_splits: int = 5,
        test_size: int = 52,
        gap: int = 13
    ) -> pd.DataFrame:
        """Cross-validate multiple models"""
        
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        cv_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(data)):
            train_fold = data.iloc[train_idx]
            val_fold = data.iloc[val_idx]
            
            for forecaster in tqdm(forecasters, desc=f"Fold {fold_idx+1}"):
                try:
                    result = self.evaluate_single_model(
                        forecaster, train_fold, val_fold
                    )
                    
                    cv_results.append({
                        "model": forecaster.name,
                        "fold": fold_idx + 1,
                        "rmse": result["metrics"]["rmse"],
                        "skill": result["metrics"]["skill_vs_naive"],
                        "nse": result["metrics"]["nse"],
                        "coverage": result["metrics"].get("coverage", None)
                    })
                    
                except Exception as e:
                    warnings.warn(f"Model {forecaster.name} failed on fold {fold_idx+1}: {str(e)}")
        
        return pd.DataFrame(cv_results)
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all evaluated models"""
        
        comparison = []
        for model_name, result in self.results.items():
            metrics = result["metrics"]
            comparison.append({
                "Model": model_name,
                "RMSE": metrics["rmse"],
                "Skill vs Naive": metrics["skill_vs_naive"],
                "NSE": metrics["nse"],
                "Coverage": metrics.get("coverage", "N/A"),
                "AIC": result["model_summary"].get("aic", "N/A")
            })
        
        return pd.DataFrame(comparison)
    
    def _calculate_metrics(
        self,
        actual: pd.Series,
        predicted: pd.Series,
        train_data: pd.Series,
        intervals: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Calculate all performance metrics"""
        
        from ..utils.metrics import (
            calculate_skill_score,
            nash_sutcliffe_efficiency,
            calculate_mase,
            calculate_smape
        )
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        
        # Skill score
        naive_forecast = self._seasonal_naive_forecast(train_data, len(actual))
        naive_rmse = np.sqrt(mean_squared_error(actual, naive_forecast))
        skill = (1 - rmse / naive_rmse) * 100
        
        # Other metrics
        nse = nash_sutcliffe_efficiency(actual, predicted)
        mase = calculate_mase(actual, predicted, train_data, self.seasonal_period)
        smape = calculate_smape(actual, predicted)
        
        # Coverage
        coverage = None
        if intervals is not None:
            coverage = self._calculate_coverage(actual, intervals)
        
        return {
            "rmse": rmse,
            "mae": mae,
            "skill_vs_naive": skill,
            "nse": nse,
            "mase": mase,
            "smape": smape,
            "coverage": coverage
        }
    
    def _seasonal_naive_forecast(self, train_data: pd.Series, steps: int) -> pd.Series:
        """Generate seasonal naive forecast"""
        seasonal_naive = train_data.iloc[-self.seasonal_period:].values
        seasonal_naive = np.tile(
            seasonal_naive,
            int(np.ceil(steps / self.seasonal_period))
        )[:steps]
        return pd.Series(seasonal_naive)
    
    def _calculate_coverage(self, actual: pd.Series, intervals: pd.DataFrame) -> float:
        """Calculate prediction interval coverage"""
        return np.mean(
            (actual >= intervals.iloc[:, 0]) & 
            (actual <= intervals.iloc[:, 1])
        )