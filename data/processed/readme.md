# Processed Data Documentation  
**Location:** `../data/processed/`  

This directory contains cleaned and transformed greenhouse gas (GHG) datasets, primarily focused on methane (CH₄), prepared for time series forecasting.  

---

## File Descriptions  

### 1. **Multi-Gas Datasets**  
#### `all_ghg_aligned.csv`  
- **Source:** NOAA flask measurements at Mauna Loa Observatory, Hawaii  
- **Variables:**  
  - `datetime`: Timestamp (weekly frequency)  
  - `value`: Atmospheric concentrations (ppb/ppm) for six gases:  
    - CH₄ (methane), CO (carbon monoxide), CO₂ (carbon dioxide)  
    - H₂ (hydrogen), N₂O (nitrous oxide), SF₆ (sulfur hexafluoride)  
- **Created in:** `1_data_loading.ipynb`  
- **Used in:** `2_all_gas_eda.ipynb`  

#### `all_ghg_aligned_nan.csv`  
- **Modifications:**  
  - Negative concentrations → `NaN` (physically implausible values)  
- **Purpose:** Sanitized baseline for CH₄-focused analysis  
- **Created in:** `2_all_gas_eda.ipynb`  
- **Used in:** `3_ch4_preprocessing.ipynb`, `4_ch4_modeling.ipynb`  

---

### 2. **CH₄-Specific Datasets**  
#### `ch4_preprocessed.csv`  
- **Processing Steps:**  
  1. Isolated CH₄ from `all_ghg_aligned_nan.csv`  
  2. Handled `NaN`s/outliers via linear interpolation  
  3. Resampled to consistent weekly frequency (`W-SUN`)  
- **Structure:**  
  - `date`: DateTime index  
  - `value`: CH₄ concentration (ppb)  
- **Created in:** `3_ch4_preprocessing.ipynb`  
- **Used in:** `6a_ch4_static_forecast.ipynb`, `6b_ch4_rolling_forecast.ipynb`  

#### `ch4_preprocessed_logged.csv`  
- **Transformation:** Natural log of `ch4_preprocessed.csv` values  
- **Purpose:** Stabilize variance for SARIMA modeling  
- **Created in:** `5_ch4_forecasting.ipynb`  
- **Used in:** `6a_ch4_static_forecast.ipynb`, `6b_ch4_rolling_forecast.ipynb`  

---

### 3. **Train/Valid/Test Splits**  
#### `ch4_train_preprocessed.csv`, `ch4_valid_preprocessed.csv`, & `ch4_test_preprocessed.csv`  
- **Source:** A rigorous temporal train-validation-test split of the `all_ghg_aligned_nan.csv` CH₄ data designed to prevent data leakage during model evaluation.
- **Split Strategy:**
  - Test Set: The most recent 52 weeks (1 year) of data. Used for the final, unbiased evaluation of the chosen model.
  - Validation Set: The 52 weeks of data immediately preceding the test set. Used for hyperparameter tuning and model selection during development.
  - Training Set: All remaining data prior to the validation set. Used for model fitting.
- **Preprocessing:**  
  - `GasPreprocessor` pipeline (outlier removal, imputation)  
  - Weekly resampling (`W-SUN`)  
- **Created in:** `4_ch4_modeling.ipynb`  
- **Used in:** `5_ch4_forecasting.ipynb`  

#### `ch4_train_logged.csv` & `ch4_test_logged.csv`  
- **Transformation:** Log-transform of train/test sets  
- **Purpose:** Model volatility reduction  
- **Created in:** `4_ch4_modeling.ipynb`  
- **Used in:** `5_ch4_forecasting.ipynb`  

---

## Key Notes  
- **Units:** All concentrations in parts per billion (ppb) except CO2 (ppm) and SF6 (ppt) 
- **Frequency:** The original data is weekly, but with inconsistent frequency.  Resampled data use weekly frequency ending Sundays (`W-SUN`)  
- **NaN Handling:**  
  - Initial negatives → `NaN` in `all_ghg_aligned_nan.csv`  
  - Interpolated in CH4 files via `GasPreprocessor` 