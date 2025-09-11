# src/preprocessing.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

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
    
    Attributes:
        trained_ (bool): Indicates whether the model has been fitted.
        stl_result_ (STL): STL decomposition result after fitting.
        start_date_ (pd.Timestamp): First valid timestamp after outlier removal.
    
    Methods:
        fit(df): Fits the preprocessor on the input DataFrame. Performs optional EDA and STL decomposition.
        transform(df): Transforms a new DataFrame using the same preprocessing steps as in `fit`.
        fit_transform(df): Combines `fit` and `transform` for convenience.
        plot_smoothed_interpolated_data(raw_series, processed_series): Plots original and processed data for comparison.
    '''
    def __init__(self, gas_name, seasonal_period=52, window=7, iqr_factor=1.5, interpolate_method='linear', 
                 resample_freq='W', lags=52, do_eda=True):
        self.gas_name = gas_name
        self.seasonal_period = seasonal_period
        self.window = window
        self.iqr_factor = iqr_factor
        self.interpolate_method = interpolate_method
        self.resample_freq = resample_freq
        self.lags = lags
        self.do_eda = do_eda

        self.stl_result_ = None
        self.start_date_ = None
        self.trained_ = False

    def _smooth_series(self, series):
        return series.rolling(window=self.window, center=True, min_periods=1).median()

    def _interpolate_series(self, series):
        return series.interpolate(method=self.interpolate_method)

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

    def fit(self, df, custom_title=None):           
        print(f'\n[INFO] Fitting preprocessing for {self.gas_name}')
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        raw_series = df[self.gas_name]
        
        # Find the data collection start date for the gas 
        self.start_date_ = raw_series.first_valid_index()
        print(f'Data collection start date for {self.gas_name}: {self.start_date_}')
        
        # ensure that negative values have been converted to NaN
        negative_mask = raw_series < 0
        if negative_mask.any():
            print(f'Warning: {negative_mask.sum()} negative values found in {self.gas_name} series. Converting to NaN.')
            raw_series = raw_series.where(raw_series >= 0, np.nan)
            
        # Trim the series to start from the first valid measurement (data collection start date)
        trimmed_series = raw_series.loc[self.start_date_:]
            
        # Debug input data
        print(f'Raw data: {len(raw_series)} points, {raw_series.isna().sum()} NaNs')
        print(f'Trimmed data: {len(trimmed_series)} points, {trimmed_series.isna().sum()} NaNs')

        # Handle missing dates and get a uniform frequency by resampling.
        # This creates a 'preliminary' series for the decomposition step.
        resampled_series = trimmed_series.resample(self.resample_freq).mean()
        
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
        # This ensures that the influence of the outlier on interpolation is removed.
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
        
        self.start_date_ = interpolated.first_valid_index()

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
        return self

    def transform(self, df, custom_title=None):
        '''
        Transforms a new DataFrame using the same preprocessing steps as in `fit`.
    
        Parameters:
        df (pd.DataFrame): New DataFrame with a 'date' column and the gas column to process.
    
        Returns:
        pd.Series: Transformed gas time series (smoothed, resampled, and interpolated).
        '''
        if not self.trained_:
            raise ValueError('You must call .fit() before .transform().')
            # Check if fit() stored the outlier mask
        if not hasattr(self, 'outlier_mask_'):
            raise ValueError('Model has not been fit with the new outlier detection logic. Call fit() first.')

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # trim the new data to start from the same date as in fit()
        new_series = df[self.gas_name].loc[self.start_date_:]
        
        # Resample the new data to the same frequency used in fit()
        new_resampled = new_series.resample(self.resample_freq).mean()

        # Align the new data with the stored outlier mask.
        # Create a series for the new data dates, default to NaN for outliers
        # The .reindex aligns the new data with the index of the mask from fit()
        new_series_to_clean = new_resampled.reindex(self.outlier_mask_.index)
        # Apply the mask: where outlier_mask_ is True, keep as NaN. Otherwise, use the new value.
        new_series_to_clean.loc[self.outlier_mask_] = np.nan

        # Apply the smoothing and interpolation
        smoothed = self._smooth_series(new_series_to_clean)
        interpolated = self._interpolate_series(smoothed)

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