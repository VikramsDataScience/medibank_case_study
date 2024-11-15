import pandas as pd
from ydata_profiling import ProfileReport

# Relative imports
from . import charts_reports_path, data_path
from .preprocessing import hospital_data_maps

for hospital_name in hospital_data_maps:
    df = pd.read_csv(data_path / f"preprocessed_{hospital_name}_data.csv").fillna(0)

    # Generate ydata profile report and export to storage as an HTML document
    profile_report = ProfileReport(df,
                                title=f"{hospital_name} Hospital EDA Report".replace('_', ' '),
                                tsmode=True, # Since the data is a Time Series activate the "tsmode"
                                explorative=True)

    profile_report.to_file(charts_reports_path / f"{hospital_name}_EDA_Profile_Report.html")
