# src/preprocessing.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from scipy import stats

class GasPreprocessor:
    '''
    Preprocess a greenhouse gas time series by handling outliers, resampling, and interpolating missing values.  
    
    This class also supports optional exploratory data analysis (EDA), including:
    - Stationarity tests (ADF and KPSS)
    - STL decomposition of the signal into trend, seasonal, and residual components
    - Autocorrelation and partial autocorrelation plots of residuals

     Parameters:
        gas_name (str): Name of the gas column in the input DataFrame.
        seasonal_period (int): Seasonality period for STL decomposition (e.g., 52 for weekly data with yearly seasonality).
        window (int): Window size for median smoothing.
        iqr_factor (float): Multiplier for the IQR to detect outliers.
        interpolate_method (str): Interpolation method for filling missing values (e.g., 'linear').
        resample_freq (str): Frequency string for resampling (e.g., 'W' for weekly).
        lags (int): Number of lags to show in ACF and PACF plots.
        do_eda (bool): If True, performs exploratory data analysis with plots and statistical tests.
        transformation (str, optional): Type of transformation to apply. Options: None, 'log', 'boxcox'.
    
    Attributes:
        trained_ (bool): Indicates whether the model has been fitted.
        stl_result_ (STL): STL decomposition result after fitting.
        start_date_ (pd.Timestamp): First valid timestamp after outlier removal.
        fitted_lambda_ (float, optional): Lambda parameter fitted for Box-Cox transformation.
        outlier_mask_ (pd.Series): Boolean series indicating outliers.
            
    Methods:
        fit(df): Fits the preprocessor on the input DataFrame. Performs optional EDA and STL decomposition.
        transform(df): Transforms a new DataFrame using the same preprocessing steps as in `fit`.
        fit_transform(df): Combines `fit` and `transform` for convenience.
        inverse_transform(series): Inverts the transformation to return to original scale.
        difference(series, order): Applies differencing to a time series.
        test_heteroscedasticity(residuals): Tests for heteroscedasticity in residuals. data for comparison.
    '''
    
# Section 1: Public interface methods
    
    def __init__(self, 
                 gas_name, 
                 seasonal_period=52, 
                 resample_freq=None,
                 window=7, 
                 iqr_factor=1.5, 
                 interpolate_method='linear', 
                 lags=52, 
                 do_eda=False, 
                 transformation=None, 
                 bc_lambda=None):
        self.gas_name = gas_name
        self.seasonal_period = seasonal_period
        self.window = window
        self.iqr_factor = iqr_factor
        self.interpolate_method = interpolate_method
        self.resample_freq = resample_freq
        self.lags = lags
        self.do_eda = do_eda
        self.transformation = transformation
        self.bc_lambda = bc_lambda
        self.stl_result_ = None
        self.start_date_ = None
        self.trained_ = False       
        self.fitted_lambda_ = None
        self.outlier_mask_ = None
        
    '''
    If resample_freq=None, the input data is assumed to already have a regular frequency.
    No resampling is applied internally.
    '''

    def fit(self, df, custom_title=None):           
        print(f'\n[INFO] Fitting preprocessing for {self.gas_name}')
        print(f"[INFO] Starting fit() for {self.gas_name} | Resample freq = {self.resample_freq or 'None (using existing index)'}")
        df = df.copy()
        
        # Allow either 'date' column or DatetimeIndex
        if 'date' in df.columns:
            df = df.set_index('date')
        elif isinstance(df.index, pd.DatetimeIndex):
            pass
        else:
            raise ValueError("Input must have a 'date' column or DatetimeIndex.")
        
        # Optional resampling
        if self.resample_freq:
            df = df.resample(self.resample_freq).mean()
            print(f"[INFO] Data resampled to {self.resample_freq} frequency.")
        else:
            inferred = pd.infer_freq(df.index)
            if inferred is None:
                print(f"[Warning] {self.gas_name}: input index has irregular frequency — no resampling applied.")

        raw_series = df[self.gas_name]
        
        # Trim both leading and trailing NaNs
        trimmed_series = self._trim_leading_nans(raw_series)
        trimmed_series = self._trim_trailing_nans(trimmed_series)
        
        # Store the actual data boundaries for reference
        self.data_start_date_ = trimmed_series.first_valid_index()
        self.data_end_date_ = trimmed_series.last_valid_index()    
        
        # Set the start date
        self.start_date_ = self.data_start_date_  
        
        # Debug input data
        print(f'Raw data: {len(raw_series)} points, {raw_series.isna().sum()} NaNs')
        print(f'Trimmed data: {len(trimmed_series)} points, {trimmed_series.isna().sum()} NaNs')
        print(f'Data range after trimming: {self.data_start_date_} to {self.data_end_date_}')
        
        # Ensure that negative values have been converted to NaN
        negative_mask = trimmed_series < 0
        if negative_mask.any():
            print(f'Warning: {negative_mask.sum()} negative values found in {self.gas_name} series. Converting to NaN.')
            trimmed_series = trimmed_series.where(trimmed_series >= 0, np.nan)
                    
        # Apply transformation (on the trimmed series with only positive values)
        if self.transformation:
            print(f'[INFO] Applying {self.transformation} transformation.')
            working_series = self._apply_transform(trimmed_series)
        else:
            working_series = trimmed_series

        # Handle missing dates and get a uniform frequency by resampling.
        if self.resample_freq:
            resampled_series = working_series.resample(self.resample_freq).mean()
        else:
            resampled_series = working_series
        
        # Debug after resampling
        print(f'After resampling: {len(resampled_series)} points, {resampled_series.isna().sum()} NaNs')
        
        # Make a working copy
        series_to_clean = resampled_series.copy()
        
        # Robust STL for outlier detection
        stl_robust = STL(series_to_clean.dropna(), # STL can't handle NaNs internally
                        period=self.seasonal_period,
                        robust=True)
        stl_result = stl_robust.fit()
        
        # Find outliers based on the residuals of the robust decomposition
        resid = stl_result.resid
        q1 = resid.quantile(0.25)
        q3 = resid.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_factor * iqr
        upper_bound = q3 + self.iqr_factor * iqr
        is_outlier = (resid < lower_bound) | (resid > upper_bound)

        # Get the dates of the outliers and mask them in the original resampled series
        outlier_dates = resid[is_outlier].index
        print(f"[INFO] Found {len(outlier_dates)} potential outliers using robust STL residuals.")
        series_to_clean.loc[outlier_dates] = np.nan # Mask outliers in the series
        
        # Debug after outlier removal
        print(f'After outlier removal: {len(series_to_clean)} points, {series_to_clean.isna().sum()} NaNs')

        # Store the outlier mask for use in transform()
        # Create a boolean series aligned with resampled_series where True=outlier
        self.outlier_mask_ = pd.Series(False, index=resampled_series.index)
        self.outlier_mask_.loc[outlier_dates] = True

        # Smooth and interpolate the cleaned series
        # Smoothing helps with small fluctuations, interpolation handles NaNs (both original and outlier-induced)
        smoothed = self._smooth_series(series_to_clean)
        
        # debug after smoothing
        print(f'After smoothing: {len(smoothed)} points, {smoothed.isna().sum()} NaNs')
        
        interpolated = self._interpolate_series(smoothed)
        
        # debug after interpolation
        print(f'After interpolation: {len(interpolated)} points, {interpolated.isna().sum()} NaNs')

        if self.do_eda:
                # Plot: Raw Resampled vs. Cleaned & Processed
                if custom_title is None:
                    custom_title = f'{self.gas_name}: Raw Resampled vs. Cleaned & Processed'
                self._plot_smoothed_interpolated_data(resampled_series, interpolated, custom_title=custom_title)

                # Run stationarity tests on the final processed series
                print('[INFO] EDA stationarity tests on processed data:')
                self._run_stationarity_tests(interpolated, 'Processed Data')

                # Perform final STL decomposition on the clean, processed data for analysis
                self.stl_result_ = STL(interpolated, period=self.seasonal_period, robust=True).fit()
                self._plot_decomposition(self.stl_result_)
                self.test_heteroscedasticity(self.stl_result_.resid, label='Heteroscedasticity Tests of Residuals')

        else:
            # If not doing EDA, we still need to fit the STL model for potential later use
            self.stl_result_ = STL(interpolated, period=self.seasonal_period, robust=True).fit()

        self.trained_ = True
        self.cleaned_series_ = interpolated
        
        # Force a fixed weekly frequency (if the data is roughly weekly) to lock alignment
        if self.resample_freq is None:
            inferred_freq = pd.infer_freq(self.cleaned_series_.index)
            if inferred_freq:
                self.cleaned_series_ = self.cleaned_series_.asfreq(inferred_freq)
        
        return self

    def transform(self, df, custom_title=None):
        '''
        Transforms a new DataFrame using the same preprocessing steps as in `fit`.
    
        Parameters:
        df (pd.DataFrame): New DataFrame with a 'date' column and the gas column to process.
    
        Returns:
        pd.Series: Transformed gas time series (smoothed, resampled, and interpolated).
        '''
        # Force test data to use same frequency as training
        if not self.resample_freq and hasattr(self, "cleaned_series_") and self.cleaned_series_.index.freq is not None:
            df = df.asfreq(self.cleaned_series_.index.freq)
            
        if not self.trained_:
            raise ValueError('You must call .fit() before .transform().')

        df = df.copy()
        if 'date' in df.columns:
            df = df.set_index('date')
        elif isinstance(df.index, pd.DatetimeIndex):
            pass
        else:
            raise ValueError("Input must have a 'date' column or DatetimeIndex.")
        
        # store the original test set data range
        original_test_start = df.index.min()
        original_test_end = df.index.max()

        # Use the new data's own time range (e.g., validation and test sets)
        new_series = df[self.gas_name]
        
        # Trim both leading and trailing NaNs
        trimmed_series = self._trim_leading_nans(new_series)
        trimmed_series = self._trim_trailing_nans(trimmed_series)
        
        # Handle negative values 
        negative_mask = trimmed_series < 0
        if negative_mask.any():
            print(f'Warning: {negative_mask.sum()} negative values found in new {self.gas_name} data.  Converting to NaN.')
            trimmed_series = trimmed_series.where(trimmed_series >= 0, np.nan)
            
        # Apply transformation (using parameters learned during fit)
        if self.transformation:
            working_series = self._apply_transform(trimmed_series)
        else:
            working_series = trimmed_series
                    
        # Resample only if frequency was specified
        if self.resample_freq:
            new_resampled = working_series.resample(self.resample_freq).mean()
        else:
            new_resampled = working_series
        
        # Ensure the resampled test set starts where the training ended
        # calculate exptected start date (train end + 1 week)
        if not hasattr(self, "data_end_date_"):
            raise ValueError("You must fit the preprocessor before using transform().")
        if self.resample_freq:
            expected_start = self.data_end_date_ + pd.tseries.frequencies.to_offset(self.resample_freq)
        else:
            expected_start = self.data_end_date_
    
        # If there is a gap, adjust the test set dates to be contiguous
        if new_resampled.index[0] > expected_start:
            print(f'Adjusting test set dates to be contiguous with train set')
            # Create new index starting where the train set ended
            new_index = pd.date_range(
                start=expected_start,
                periods=len(new_resampled),
                freq=self.resample_freq
            )
            new_resampled.index = new_index

        # Apply the smoothing and interpolation
        smoothed = self._smooth_series(new_resampled)
        # Enforce same frequency as training before interpolation
        if self.cleaned_series_.index.freq is not None:
            smoothed = smoothed.asfreq(self.cleaned_series_.index.freq)
            
        interpolated = self._interpolate_series(smoothed)
        
        # Check if we're transforming the training data (same as fitted data)
        is_training_data = (df.index.equals(self.cleaned_series_.index)) if hasattr(self, 'cleaned_series_') else False
    
        # Only remove overlap for test data, not training data
        if hasattr(self, "data_end_date_") and not is_training_data:
            if interpolated.index[0] <= self.data_end_date_:
                print(f"[INFO] Dropping overlapping test point at {interpolated.index[0]} (train end = {self.data_end_date_})")
                interpolated = interpolated.loc[interpolated.index > self.data_end_date_]

        return interpolated

    def fit_transform(self, df, custom_title=None):
        '''
        Fits the preprocessor on the input DataFrame and returns the transformed series.

        Parameters:
        df (pd.DataFrame): Input DataFrame with a 'date' column and the gas column.

        Returns:
        pd.Series: Transformed gas time series.
        '''
        self.fit(df, custom_title=custom_title)  
        return self.transform(df, custom_title=custom_title)
    
    def inverse_transform(self, series):
        '''
        Inverts the transformation applied during fit/transform.
        Use this on forecasts or predictions to bring them back to original scale.
        
        Parameters:
        series (pd.Series): Series in transformed space to convert back to original units.
        
        Returns:
        pd.Series: Series in original units.
        '''
        if self.transformation: 
            return self._apply_transform(series, inverse=True)
        else:
            return series
        
# Section 2: Core preprocessing methods 

    def _trim_trailing_nans(self, series):
        '''
        Trim trailing NaN values from a series
        '''
        last_valid_idx = series.last_valid_index()
        if last_valid_idx is not None and last_valid_idx < series.index[-1]:
            print(f'\nTrimming {len(series.loc[last_valid_idx:]) - 1} trailing NaN values')
            return series.loc[:last_valid_idx]
        return series
    
    def _trim_leading_nans(self, series):
        '''
        Trim leading NaN values from a series
        '''
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is not None and first_valid_idx > series.index[0]:
            print(f'\nTrimming {len(series.loc[:first_valid_idx]) - 1} leading NaN values')
            return series.loc[first_valid_idx:]
        return series
                    
    def _apply_transform(self, series, inverse=False):
        '''
        Applies or inverts the specified transformation to a series
        '''
        if self.transformation is None:
            return series
        
        if not inverse:
            # apply forward transformation
            if self.transformation == 'log':
                # ensure positive values for log transform
                positive_series = series.where(series > 0, np.nan)
                return np.log(positive_series)
            elif self.transformation == 'boxcox':
                # Handle NaN values first
                non_na_mask = series.notna()
                positive_series = series[non_na_mask]
                
                # Ensure all values are positive
                if (positive_series <= 0).any():
                    print(f'Warning: Non-positive values found in Box-Cox transformation. Clipping to small positive value.')
                    positive_series = positive_series.clip(lower=1e-10)
                
                print(f'Box_Cox input stats: min={positive_series.min()}, max={positive_series.max()}, mean={positive_series.mean()}')
                
                # Check if we're using a fixed lambda
                if self.bc_lambda is not None:
                    # Use the fixed lambda value
                    self.fitted_lambda_ = self.bc_lambda
                    print(f'Using fixed lambda: {self.fitted_lambda_}')
                    
                    # handle lambda = 0 case (log transformation)
                    if self.fitted_lambda_ == 0:
                        transformed_data = np.log(positive_series)
                    else:
                        # Apply Box-Cox transformation with fixed lambda
                        transformed_data = stats.boxcox(positive_series, lmbda=self.fitted_lambda_)
                else:
                    # check if in fit (need to compute lambda) or transform (use stored lambda)
                    if not hasattr(self, 'fitted_lambda_') or self.fitted_lambda_ is None:
                        # this should happen only during .fit()
                        # boxcox requires positive data, which is ensured by prior step
                        transformed_data, fitted_lambda = stats.boxcox(positive_series)
                        print(f'Calculated lambda: {fitted_lambda}')
                        self.fitted_lambda_ = fitted_lambda
                    else:
                        # this is .transform(), use the stored lambda
                        if self.fitted_lambda_ == 0:
                            transformed_data = np.log(positive_series)
                        else:
                            transformed_data = stats.boxcox(positive_series, lmbda=self.fitted_lambda_)
                
                # Create a new series with transformed values, preserving the index
                transformed_series = pd.Series(transformed_data, index=positive_series.index)
                
                # Reindex to original index, NaNs will remain NaN
                return transformed_series.reindex(series.index)
        else:
            # apply inverse transformation
            if self.transformation == 'log':
                return np.exp(series)
            elif self.transformation == 'boxcox':
                if self.fitted_lambda_ == 0:
                    return np.exp(series) 
                else: 
                    # Correct inverse Box-Cox transformation
                    return np.power(series * self.fitted_lambda_ + 1, 1 / self.fitted_lambda_)


    def _smooth_series(self, series):
        return series.rolling(window=self.window, center=True, min_periods=1).median()

    def _interpolate_series(self, series):
        # Ensure start_date_ is set
        if self.start_date_ is None:
            raise ValueError('start_date_ must be set before calling _interpolate_series')
        # Ensure we're not trying to interpolate before the first valid date
        if series.index.min() < self.start_date_:
            print(f"Warning: Series contains dates before {self.start_date_}. Trimming to valid range.")
            series = series.loc[self.start_date_:]
            
        # interpolate within the valid data range
        interpolated = series.interpolate(method=self.interpolate_method, limit_direction='both', limit_area='inside')       
            
        return interpolated
    
    def difference(self, series, order=1):
        '''
        Apply difference of specified order to a time series.

        Parameters:
            series (pd.Series): a single gas time series
            order (int): order of differencing

        Returns:
            pd.Series: differenced series
        '''
        differenced = series.diff(order).dropna()
        return differenced

# Section 3: EDA/Visualization methods 

    def _plot_smoothed_interpolated_data(self, raw_series, processed_series, figsize=(10,4), 
                                         title_fontsize=16, custom_title=None):
        plt.close('all') # close any open figures to avoid ghost plots
    
        plt.figure(figsize=figsize)
        plt.plot(raw_series, label='Raw Data', marker='.', markersize=3, color='#0072B2', linewidth=0.75, alpha=0.7)
        plt.plot(processed_series, label='Smoothed, Resampled, & Interpolated', color="#F0950D", alpha=0.9)
        
        title = custom_title if custom_title else f'{self.gas_name} Time Series'
        plt.title(title, fontsize=title_fontsize)
        
        plt.ylabel('Concentration', fontsize=18)
        plt.xlabel('Time', fontsize=18)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()

    def _run_stationarity_tests(self, series, label):
        try:
            adf_result = adfuller(series.dropna())
        except ValueError as e:
            print(f'Error in ADF test: {e}')
        
        try:
            kpss_result = kpss(series.dropna(), regression='c')
        except ValueError as e:
            print(f'Error in KPSS test: {e}')

        print(f'ADF and KPSS tests for {label}:')
        print(f'ADF statistic {adf_result[0]:.4f}')
        print(f'ADF p-value {adf_result[1]:.4f}')
        print(f'ADF critical values: {adf_result[4]}\n')
    
        print(f'KPSS statistic {kpss_result[0]:.4f}')
        print(f'KPSS p-value {kpss_result[1]:.4f}')
        print(f'KPSS critical values: {kpss_result[3]}\n')
    
        if adf_result[1] < 0.05 and kpss_result[1] > 0.05:
            print(f'the {label} time series is likely stationary according to ADF and KPSS tests.\n')
        elif adf_result[1] > 0.05 and kpss_result[1] < 0.05:
            print(f'the {label} time series is non-stationary according to ADF and KPSS tests.\n')
        elif adf_result[1] > 0.05 and kpss_result[1] > 0.05:
            print(f'the {label} time series may be trend-stationary according to ADF and KPSS tests.\n')
        else: 
            print(f'the {label} time series may be difference-stationary according to ADF and KPSS tests.\n')

    def _plot_decomposition(self, stl_result):
        # check if residuals have enough data
        residuals_clean = stl_result.resid.dropna()
        if len(residuals_clean) < 2:
            print('Warning: Not enough data in residuals for decomposition plots')
            return
        
        # create the decomposition plot
        fig = stl_result.plot()
        plt.suptitle('STL Decomposition', fontsize=16)      
        
        # Access the axes of the plot
        axes = fig.axes
        
        # Define font sizes
        label_fontsize = 12
        tick_fontsize = 12
        
        # customize subplots
        for i, ax in enumerate(axes):
            # set the y-axis label for each subplot
            if i == 0:
                y_label = 'Observed'
            elif i == 1:
                y_label = 'Trend'
            elif i == 2:
                y_label = 'Seasonal'
            elif i == 3:
                y_label = 'Residual'
            else:
                y_label = ax.get_ylabel() 
            
            # set y-axis position, label, and tick font sizes
            ax.yaxis.set_label_position('right')
            ax.set_ylabel(y_label, fontsize=label_fontsize)
            ax.tick_params(axis='y', labelsize=tick_fontsize)
            
            # set x-axis tick fontsize
            ax.tick_params(axis='x', labelsize=tick_fontsize)
            
            # add x-axis label to the bottom subplot only
            if i == len(axes) - 1: # last subplot
                ax.set_xlabel('Year', fontsize=label_fontsize)
        
        # Customize the residual plot (4th subplot)
        if len(axes) >= 4:
            resid_ax = axes[3]  # Residuals are typically the 4th subplot
            # Change residual plot to use lines with small markers
            lines = resid_ax.get_lines()
            if lines:
                # Modify the existing line (default is markers only for residuals)
                line = lines[0]
                line.set_linestyle('-')  # Add line connecting points
                line.set_marker('o')     # Ensure markers are circles
                line.set_markersize(2)   # Reduce marker size
                line.set_alpha(0.6)      # Optional: adjust transparency
            
        plt.tight_layout()
        plt.show()
    
        # ACF and PACF of residuals
        plt.figure(figsize=(12,4))
        
        # calculate safe number of lags (must be less than the length of the series)
        safe_lags = min(self.lags, len(residuals_clean) -1)
        
        plt.subplot(1,2,1)
        plot_acf(stl_result.resid.dropna(), ax=plt.gca(), lags=safe_lags)
        plt.title('ACF of Residuals', fontsize=16)
        plt.ylabel('Autocorrelation Coef', fontsize=16)
        plt.xlabel('Lag', fontsize=16)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
    
        plt.subplot(1,2,2)
        plot_pacf(stl_result.resid.dropna(), ax=plt.gca(), lags=safe_lags)
        plt.title('PACF of Residuals', fontsize=16)
        plt.ylabel('Partial Autocorrelation Coef', fontsize=16)
        plt.xlabel('Lag', fontsize=16)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
    
        plt.tight_layout()
        plt.show()
        
    def test_heteroscedasticity(self, residuals, label='Heteroscedasticity Tests'):
        '''
        Test for heteroscedasticity in the residuals using the Breusch-Pagan and White tests.

        Parameters:
            residuals (pd.Series): residuals from the fitted model

        Returns:
            Breusch-Pagan p-value: p < 0.05 indicates heteroscedasticity
            White Test p-value: p < 0.05 indicates heteroscedasticity 
        '''
        X = sm.add_constant(np.arange(len(residuals)))
        bp_test = het_breuschpagan(residuals, X)
        white_test = het_white(residuals, X)
        
        print(f'\n{label}')
        print(f'Breusch-Pagan p-value: {bp_test[1]:.4f}')
        if bp_test[1] < 0.05:
            print('Heteroscedasticity detected (Breusch-Pagan test)')
        else:
            print('No heteroscedasticity detected (Breusch-Pagan test)')
            
        print(f'\nWhite Test p-value: {white_test[1]:.4f}')
        if white_test[1] < 0.05:
            print('Heteroscedasticity detected (White test)')
        else:
            print('No heteroscedasticity detected (White test)')

        return {'bp_pvalue': bp_test[1], 'white_pvalue': white_test[1]} 

    def _run_stationarity_and_ac_analysis(self, series, label='Differenced Series'):
        '''
        Run stationarity tests and plot ACF/PACF plots on the provided series.

        Parameters:
            series (pd.Series): the time series to analyze, which is intended for the differenced series
            label (str): label for the output and plot titles
        '''
        print(f'\n[INFO] Stationarity and Autocorrelation Analysis for {label}')
        self._run_stationarity_tests(series, label)

        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        plot_acf(series.dropna(), ax=plt.gca(), lags=self.lags)
        plt.title(f'ACF of {label}')

        plt.subplot(1,2,2)
        plot_pacf(series.dropna(), ax=plt.gca(), lags=self.lags)
        plt.title(f'PACF of {label}')

        plt.tight_layout()
        plt.show()
        
    def plot_acf_pacf(self, series, title_suffix=''):
        '''
        Plot ACF and PACF plots for any time series
        '''
        # make sure there is enough data
        series_clean = series.dropna()
        if len(series_clean) < 2:
            print('Warning: Not enough data for ACF and PACF plots')
            return
        
        # calculate safe number of lags
        safe_lags = min(self.lags, len(series_clean) - 1)
        
        # plot
        plt.figure(figsize=(12,4))
        
        plt.subplot(1,2,1)
        plot_acf(series_clean, ax=plt.gca(), lags=safe_lags)
        plt.title(f'ACF Plot of {title_suffix}', fontsize=16, y=1.10)
        plt.ylabel('Autocorrelation Coef', fontsize=16)
        plt.xlabel('Lag', fontsize=16)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        plt.subplot(1,2,2)
        plot_pacf(series_clean, ax=plt.gca(), lags=safe_lags)
        plt.title(f'PACF Plot of {title_suffix}', fontsize=16, y=1.10)
        plt.ylabel('Partial Autocorrelation Coef', fontsize=16)
        plt.xlabel('Lag', fontsize=16)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        