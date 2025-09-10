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

    def _mask_outliers(self, series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_factor * iqr
        upper = q3 + self.iqr_factor * iqr
        return series.where((series >= lower) & (series <= upper), np.nan)

    def _smooth_series(self, series):
        return series.rolling(window=self.window, center=True, min_periods=1).median()

    def _interpolate_series(self, series):
        return series.interpolate(method=self.interpolate_method)

    def _plot_smoothed_interpolated_data(self, raw_series, processed_series, figsize=(10,4), 
                                         title_fontsize=16, custom_title=None):
        plt.close('all') # close any open figures to avoid ghost plots
    
        plt.figure(figsize=figsize)
        plt.plot(raw_series, label='Raw Data', marker='.', markersize=3, linewidth=0.75, alpha=0.5)
        plt.plot(processed_series, label='Smoothed, Resampled, & Interpolated', alpha=0.7)
        
        title = custom_title if custom_title else f'{self.gas_name} Time Series'
        plt.title(title, fontsize=title_fontsize)
        
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
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
        stl_result.plot()
        plt.suptitle('STL Decomposition', fontsize=16)
        plt.tight_layout()
        plt.show()
    
        # ACF and PACF of residuals
        plt.figure(figsize=(12,4))
        
        plt.subplot(1,2,1)
        plot_acf(stl_result.resid.dropna(), ax=plt.gca(), lags=self.lags)
        plt.title('ACF of Residuals')
    
        plt.subplot(1,2,2)
        plot_pacf(stl_result.resid.dropna(), ax=plt.gca(), lags=self.lags)
        plt.title('PACF of Residuals')
    
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

        series = df[self.gas_name]
        masked = self._mask_outliers(series)
        self.start_date_ = masked.first_valid_index()

        smoothed = self._smooth_series(masked).loc[self.start_date_:]
        resampled = smoothed.resample(self.resample_freq).mean()
        interpolated = self._interpolate_series(resampled)

        if self.do_eda:
            if custom_title is None:
                custom_title = f'{self.gas_name} Time Series: Raw Data vs. Smoothed, Resampled, & Interpolated'
            self._plot_smoothed_interpolated_data(series, interpolated, custom_title=custom_title)
            print('[INFO] EDA stationarity tests:')
            self._run_stationarity_tests(series, 'Raw Data')
            self._run_stationarity_tests(smoothed, 'Smoothed Data')
            self._run_stationarity_tests(interpolated, 'Resampled & Interpolated Data')

        stl = STL(interpolated, period=self.seasonal_period, robust=True)
        self.stl_result_ = stl.fit()

        if self.do_eda:
            self._plot_decomposition(self.stl_result_)
            self._run_stationarity_tests(self.stl_result_.trend, 'Trend')
            self._run_stationarity_tests(self.stl_result_.seasonal, 'Seasonal')
            self._run_stationarity_tests(self.stl_result_.resid, 'Residuals')
            self.test_heteroscedasticity(self.stl_result_.resid, label='Heteroscedasticity Tests of Residuals')
            
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

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        series = df[self.gas_name]
        masked = self._mask_outliers(series)
        smoothed = self._smooth_series(masked).loc[self.start_date_:]
        resampled = smoothed.resample(self.resample_freq).mean()
        interpolated = self._interpolate_series(resampled)

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