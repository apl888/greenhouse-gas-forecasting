# Data

## Attribution
This project uses publicly available greenhouse gas concentration data from NOAA's Mauna Loa Observatory.

The data is dynamically loaded from NOAA's servers. **Always cite the original dataset**—see the [main citation](../README.md#data-citation) for details.  

## Datasets
To inspect or modify the data sources, see the dataset dictionary in the code:

```python
datasets = {
    'CH4': 'https://gml.noaa.gov/aftp/data/trace_gases/ch4/flask/surface/txt/ch4_mlo_surface-flask_1_ccgg_event.txt',
    'N2O': 'https://gml.noaa.gov/aftp/data/trace_gases/n2o/flask/surface/txt/n2o_mlo_surface-flask_1_ccgg_event.txt',
    'SF6': 'https://gml.noaa.gov/aftp/data/trace_gases/sf6/flask/surface/txt/sf6_mlo_surface-flask_1_ccgg_event.txt',
    'CO2': 'https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/txt/co2_mlo_surface-flask_1_ccgg_event.txt',
    'CO': 'https://gml.noaa.gov/aftp/data/trace_gases/co/flask/surface/txt/co_mlo_surface-flask_1_ccgg_event.txt',
    'H2': 'https://gml.noaa.gov/aftp/data/trace_gases/h2/flask/surface/txt/h2_mlo_surface-flask_1_ccgg_event.txt'
}
```

This project uses a simplified version of NOAA's dataset, keeping only:  
- `datetime`: Timestamp of measurement  
- `value`: Gas concentration (units vary by gas)  

For the full dataset schema, see [DATA_DICTIONARY.md](DATA_DICTIONARY.md).  
