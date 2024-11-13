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

## Part 2: Exploratory Data Analysis (EDA)
Most of Triage categories across all the hospitals had at least some missing data. Given that this data is a Time Series with a fixed daily frequency, we cannot have any missing data. So I opted to perform some simple static imputation using the calculated Median values by column.

With regards to the below, for the code and the plots generated, please refer to the `src.decomposition_models` module and the `/charts_reports` directory the plots themselves.

### STL (Seasonality-Trend decomposition using LOESS) model for Royal Perth Hospital (Triage Category 1)
STL LOESS/LOWESS stands for 'Seasonal and Trend decomposition using LOcally Estimated Scatterplot Smoothing'/'Locally Weighted Scatterplot Smoothing'. This method is a more robust version of the Classical Seasonal Decomposition method since its LOESS capability allows it to uncover non-linear relationships.
- Conceptual reference: https://otexts.com/fpp2/stl.html (more basic) and https://www.wessa.net/download/stl.pdf (very good research paper with more concise Mathematical definitions of the algo).

&nbsp; When using STL Decomposition to smooth over some of the variable spikes in patient activity over the length of the time series, we can see that there is a relatively flat trend that is quite stable. In saying that, there is stronger seasonality present over November 2013, and again in June 2014. There's a good chance that this is attributable to holiday periods. 
&nbsp; With respect to spikes in activity, August 31 2013 & June 29 2014 saw 'Tri_1' activity spike to 15 for both those days. As can be the case with ED, this could be the result of a nearby accident or emergency event that resulted in a spike for urgent medical attention.

### MSTL (Multiple Seasonality-Trend decomposition using LOESS) for Outlier Detection at Royal Perth Hospital (Triage Category 1)
- According to the researcher, Rob Hyndman, MSTL can be used for outlier detection if the Seasonality and Trend components are ignored (https://robjhyndman.com/hyndsight/tsoutliers/). 
- In accordance with this methodology, I've come up with the implementation that can found in the `src.decomposition_models` module of his proposal. 
- By plotting and examining the residuals (i.e. ignoring the Seasonality and Trend components, but maintaining them in the plots for comparison) that are 3 standard deviations ($\sigma$) away the model's residuals (or remainders), we can regard them as outliers, rather than traditional noise that's contained in $R_t$ component of the MSTL equation:
<!-- Centered equation -->
$$y_t = T_t + S_t + R_t$$
<!-- Centered equation -->
- Anything that is $<3\sigma$ in the residuals can be regarded as noise in the data for which the model cannot account and is stored in the $R_t$ component. The decision for using $3\sigma$ came from some quick research conducted, and was validated by a lesson found published by <a href='https://online.stat.psu.edu/stat501/lesson/11/11.3#:~:text=The%20good%20thing%20about%20internally,is%20generally%20deemed%20an%20outlier.'>Penn State University</a>.

&nbsp; On outlier detection, there are 4 outliers detected by the MSTL algorithm - 31 August, 8 December 2013, 14 & 15 June 2014. This is good news for the upcoming forecast model, as there are only 4 outliers that can cause the accuracy of any forecast for this time series to be adversely affected. Of course, this does depend on the length of the forecast horizon. Since EDs are driven by emergencies, it's generally better to only have short window forecasts (for instance, my 2nd project at Healthscope was a 3-day forecast of Presentations for Healthscope's 8 EDs). Longer horizon forecasts for data that can change rapidly would inevitably lead to a poor forecast.

## Part 3: Differences between the hospitals based on the given data
An important lesson I learned at Healthscope was that each hospital is a microcosm of uniqueness when regarding patient activity! The assumptions that apply to one hospital don't tend to apply to others.

## Part 4: Forecast model for Triage category 1 at Royal Perth Hospital
### Model's Drivers
- 1 year of training data
- The trends and cyclical components over that 1 year

### Model's Limitations
- **Limited training data**: This is 1 year of training data that has been sufficient to capture the trends and cyclical effects over the course of that year only - which included the 4 outliers that the MSTL decomposition model uncovered. But beyond this 1 year of training data we don't have any other trends and cyclicalities that govern the activity of this ED. For instance, what if this 2013-2014 period was a year where activity levels were either lower or higher than in previous years? The model wouldn't have sufficient context for predicting the next forecast horizon, and would likely fail when generating the Out-Of-Sample forecasts, but would do well in the Out-Of-Sample Cross Validated _Predictions_, since it's testing it's own predictions against a limited training set!
- **No Exogenous features**: Under the hood, the STLForecast is basically an ARIMA model that predicts trends and cyclicalities that are infered from the training data. However, forecasting is a very complex and highly sensitive body of work that requires exogenous features to enrich the $\hat{y}$ predictor! For instance, at Healthscope when developing the 3-day ED Presentations forecast model, after extensive experimentation and exploration of the data, I was finally able to find that transfer_reason, type, arrival, triage_category, and value were the most effective 5 exogenous categorical features that would enrich the $\hat{y}$ predictor (Presentations) and reduce the MAE and MAPE measures of error during Cross Validation. Something of that nature would be needed here, as well.
- **Seasonality not included**: Since this is quite a simple ARIMA model, Seasonality is not included. However, the STL Decomposition EDA does detect the presence of seasonality in the data that varies over the time series.
- **No holiday effects**: Much like seasonality, holidays have a substantial effect on ED patient activity. During the development of the Healthscope ED Presentations model, it would found that the Christmas/NY period was the time when the most number of accidents tend to occur in the home that generate visits to the ED - especially falls from ladders. Those were quite common as more people were using their holidays to perform DIY repairs around the home, resulting in more accidents.
- **Too long a forecast horizon**: 
