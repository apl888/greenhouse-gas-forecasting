# Data

This project uses publicly available greenhouse gas concentration data from NOAA's Mauna Loa Observatory.

The data is **not stored** in this repository. Instead, it is loaded dynamically from NOAA's website using the script in the analysis notebook or code.

To inspect or modify the data sources, see the dataset dictionary in the code:

datasets = {
    'CH4': 'https://gml.noaa.gov/aftp/data/trace_gases/ch4/flask/surface/txt/ch4_mlo_surface-flask_1_ccgg_event.txt',
    'N2O': 'https://gml.noaa.gov/aftp/data/trace_gases/n2o/flask/surface/txt/n2o_mlo_surface-flask_1_ccgg_event.txt',
    'SF6': 'https://gml.noaa.gov/aftp/data/trace_gases/sf6/flask/surface/txt/sf6_mlo_surface-flask_1_ccgg_event.txt',
    'CO2': 'https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/txt/co2_mlo_surface-flask_1_ccgg_event.txt',
    'CO': 'https://gml.noaa.gov/aftp/data/trace_gases/co/flask/surface/txt/co_mlo_surface-flask_1_ccgg_event.txt',
    'H2': 'https://gml.noaa.gov/aftp/data/trace_gases/h2/flask/surface/txt/h2_mlo_surface-flask_1_ccgg_event.txt'
}
