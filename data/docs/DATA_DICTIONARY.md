# NOAA Mauna Loa Surface-Flask Greenhouse Gas Data Dictionary 

### **Original Dataset Columns**  
| Column | Description | Unit | Kept? | Notes |  
|--------|-------------|------|-------|-------|  
| `site_code` | Measurement site ID | text | dropped | Always "MLO" (Mauna Loa) |  
| `year`, `month`, `day`, `hour`, `minute`, `second` | Timestamp components | - | dropped | Merged into `datetime` |  
| `datetime` | Full timestamp (UTC) | ISO 8601 | kept | Used as primary time index |  
| `time_decimal` | Fractional year | decimal | dropped | Alternative time format |  
| `value` | Gas concentration | varies | kept | Primary target variable |  
| `value_unc` | Measurement uncertainty | varies | dropped | Dropped (not used in forecasting) |  
| `latitude`, `longitude`, `altitude` | Location metadata | degrees/meters | dropped | Fixed for Mauna Loa |  
| ... | *(other columns)* | ... | dropped | *(see NOAA's documentation)* |  

### **Processing Steps**  
1. **Column Selection**: Only `datetime` and `value` are retained.  
2. **Unit Conversion**: All gases use their native units (e.g., CH4, N2O, CO, H2 in ppb, CO2 in ppm, SF6 in ppt).  
3. **Missing Data**: Rows with `value = -999.99` (NOAA's missing value flag) are converted to NaN.  
4. **Resampling**: Data is resampled to weekly frequency (mean aggregation). 
