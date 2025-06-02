# greenhouse-gas-forecasting

## Data Citation
This project uses data from the [NOAA Global Monitoring Laboratory](https://gml.noaa.gov/). If you use this repository, please cite teh original data source:

> Lan, X., J.W. Mund, A.M. Crotwell, K.W. Thoning, E. Moglia, M. Madronich, K. Baugh, G. Petron, M.J. Crotwell, D. Neff, S. Wolter, T. Mefford and S. DeVogel (2025), Atmospheric Methane Dry Air Mole Fractions from the NOAA GML Carbon Cycle Cooperative Global Air Sampling Network, 1983-2024,  Version: 2025-04-26, https://doi.org/10.15138/VNCZ-M766

## Model Overview
SARIMA-based forecasting of Mauna Loa methane gas concentration data from NOAA

This project forecasts weekly atmospheric concentrations of greenhouse gases using time series modeling techniques. Although six gases were loaded into the dataset, this analysis focuses on methane (CH4) due to its significant role in climate change.

The core objective is to preprocess CH4 time series data and produce a forecast using a SARIMA model, aided by auto-ARIMA selection and manual refinement.

Project Overview
- Load and preprocess flask collected CH4 conectration data from NOAA Mauna Loa Observatory
- Smooth and interpolate the time series data
- split into train and test sets
- employ Auto-ARIMA for parameter suggestion
- fit SARIMA model based on trial-and-error and statistical insight
- forecast CH4 concentrations
- visualize predictions with confidence intervals

Method and Workflow
1. Data loading
   - greenhouse gas data for six gases was loaded into a pandas DataFrame
   - focused analysis of CH4
     
2. Preprocessing
   - implemented via custom GasPreprocessor class
   - masks extreme outliers based on IQR
   - applies rolling median smoothing
   - resamples data for consistent weekly frequency
   - linearly interpolates missing data
   - performs
     - ADF and KPSS tests for stationarity
     - STL decomposition to visually assess signal components
     - Breusch-Pagan and White tests for heterscedasticity of the residuals
     - Autocorrelation and Partial autocorrelation analysis
   - provides a fit() and transform() interface

3.  Modeling and Forecasting
      - split the preprocessed CH4 series into training and test sets
      - applied log transform to stabilize variance
      - used Atuo-ARIMA to identify strarting SARIMA parameters
      - final SARIMA model chosen based on both Auto-ARIMA suggestions and empirical tuning
      - forecasted 13 weeks ahead with 95% confidence intervals

4. Forecast Evaluation
   - custom forecast_metrics() function computes forecast performance
   - visualizations include:
     - training data
     - test data
     - forecast mean
     - confidence intervals
   
