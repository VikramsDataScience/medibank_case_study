# Medibank Case Study

## Data Dictionary (sourced from the data.gov, but provided below for ease of reference)
- Attendances: the number of patients recorded as arriving at a public emergency department 
- Admissions: the number of patients who are subsequently admitted to the hospital for care and/or treatment
- Triage categories are allocated to each patient based on an assessment of their presenting conditions, generally by the triage nurse, with triage 1 being the most urgent and triage 5 being the least urgent. 
    - Triage 1: Resuscitation- immediate, within seconds; 
    - Triage 2: Emergency- within 10 minutes; 
    - Triage 3: Urgent- within 30 minutes; 
    - Triage 4: Semi-urgent- within 60 minutes; 
    - Triage 5: Non-urgent - within 120 minutes. 
- N/A - Values is less than 3 and has been suppressed.

## Preprocessing
I had initially intended to use Pandas' MultiIndexing capability, specifically `pd.MultiIndex.from_tuples()` to create a multi level index to access the tuple pairs in the dataset, given that this data set has two levels (Hospital, and the associated ED Metrics such as the Attendance, Admissions, etc.). However, the dataset has been set up in a slightly odd fashion, whereby, the following code I had initially written:
```
df = pd.read_csv(data_path / "govhack3.csv", header=[0, 1])
original_tuples = df.columns.to_list()
df.columns = pd.MultiIndex.from_tuples(original_tuples, names=['Hospital', 'Metric'])
```
Was only able to access the 'Attendance' metric:
```
Metric  Attendance
0              235
1              209
2              204
3              199
4              193
..             ...
360            222
361            224
362            239
363            218
364            234
```

Despite quite a few efforts to debug and repair this issue, I was sadly unable to arrive at a good outcome. So, I had to opt for the more manual approach that can be found in the `src.preprocessing` module to be able to access a given hospital's metrics.

I apologise for not being able to get this working in the clean and robust way that we would all like! But in the spirit of not wasting too much time, I opted for the aforementioned approach which did yield the correct data.

## Exploratory Data Analysis (EDA)
Most of Triage categories across all the hospitals had at least some missing data. Given that this data is a Time Series with a fixed daily frequency, we cannot have any missing data. So I opted to perform some simple static imputation using the calculated Median values by column.

### STL (Seasonality-Trend decomposition using LOESS) model for Royal Perth Hospital (Triage Category 1)
For the code and the plots generated please refer to the `src.decomposition_models` module and the `/charts_reports` directory the plots themselves.

### MSTL (Multiple Seasonality-Trend decomposition using LOESS) for Outlier Detection at Royal Perth Hospital (Triage Category 1)
- According to the researcher, Rob Hyndman, MSTL can be used for outlier detection if the Seasonality and Trend components are ignored (https://robjhyndman.com/hyndsight/tsoutliers/). 
- In accordance with this methodology, I've come up with the implementation that can found in the `src.decomposition_models` module of his proposal. 
- By plotting and examining the residuals (i.e. ignoring the Seasonality and Trend components, but maintaining them in the plots for comparison) that are 3 standard deviations ($\sigma$) away the model's residuals, we can regard them as outliers, rather than traditional noise that's contained in $R_t$ component of MSTL equation:
<!-- Centered equation -->
$$y_t = T_t + S_t + R_t$$
<!-- Centered equation -->
- Anything that is $<3\sigma$ in the residuals can be regarded as noise in the data for which the model cannot account and is stored in the $R_t$ component. The decision for using $3\sigma$ came from some quick research conducted, and was validated by a lesson found published by <a href='https://online.stat.psu.edu/stat501/lesson/11/11.3#:~:text=The%20good%20thing%20about%20internally,is%20generally%20deemed%20an%20outlier.'>Penn State University</a>.
