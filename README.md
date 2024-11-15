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
I had initially intended to use Pandas' MultiIndexing capability, specifically `pd.MultiIndex.from_tuples()` (since the indexes they've created are tuple pairs) to create a multi level index to access the tuple pairs in the dataset, given that this data set has two levels (Hospital, and the associated ED Metrics such as the Attendance, Admissions, etc.). However, the dataset has been set up in a slightly odd fashion, whereby, the following code I had initially written:
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

I apologise for not being able to get this working in the clean and robust way that we would all like! But in the spirit of not wasting too much time, I opted for the aforementioned approach which appeared to yield the correct data.

## CI Workflow and Docker Container
For the python modules that have been developed in this case study, you'll see that I've built a Docker Container with an accompanying CI Workflow (YAML file in the `.github/workflows` directory) that automatically triggers the container whenever there's a push to any branch. I know it wasn't part of the requirement for this case study, but I wanted to go the extra mile to demonstrate skill in Machine Learning Engineering, as well, since possessing ML Ops skill is a critical part of modern Data Science! I also just wanted to do it!

## Part 2: Exploratory Data Analysis (EDA)
Most of Triage categories across all the hospitals had at least some missing data, since the activity levels that are <3 are suppressed. Given that this data is a Time Series with a fixed daily frequency, we cannot have any missing data, as this breaks the Time Series and won't allow us to build Decomposition or Forecast models (please read on for the code and the analysis for these two important steps!). So I opted to impute 0s into the columns.

With regards to the below, for the code and the plots generated, please refer to the `src.decomposition_models` module and the `/charts_reports` directory for the plots.

### STL (Seasonality-Trend decomposition using LOESS) model for Royal Perth Hospital (Triage Category 1)
STL LOESS/LOWESS stands for 'Seasonal and Trend decomposition using LOcally Estimated Scatterplot Smoothing'/'Locally Weighted Scatterplot Smoothing'. This method is a more robust version of the Classical Seasonal Decomposition method since its LOESS capability allows it to uncover non-linear relationships.
- Conceptual reference: https://otexts.com/fpp2/stl.html (more basic) and https://www.wessa.net/download/stl.pdf (very good research paper with more concise Mathematical definitions of the algo).

&nbsp; When using STL Decomposition to smooth over some of the variable spikes in patient activity over the length of the time series, we can see that there is a relatively stable trend over the time series. In saying that the following two things did occur: 
- There is stronger seasonality present over November 2013, and again in June 2014. There's a good chance that this is attributable to holiday periods.
- There was also a significant dip and then return to normal trend over October 2013 and again over June 2014. There are school holidays during both those periods. It's quite possible those could be a cause for the change in activity levels.

&nbsp; With respect to spikes in activity, August 31 2013 & June 29 2014 saw 'Tri_1' activity spike to 15 for both those days. As can be the case with EDs, this could be the result of a nearby accident or emergency event that resulted in a spike for urgent medical attention.

### MSTL (Multiple Seasonality-Trend decomposition using LOESS) for Outlier Detection at Royal Perth Hospital (Triage Category 1)
- According to the researcher, Rob Hyndman, MSTL can be used for outlier detection if the Seasonality and Trend components are ignored (https://robjhyndman.com/hyndsight/tsoutliers/). 
- In accordance with this methodology, I've come up with the implementation that can found in the `MSTL_Outlier_Detection()` function of the `src.decomposition_models` module of his proposal. 
- By plotting and examining the residuals (i.e. ignoring the Seasonality and Trend components, but maintaining them in the plots for comparison) that are 3 standard deviations ($\sigma$) away from the model's residuals (or remainders), we can regard them as outliers, rather than traditional noise that's contained in $R_t$ (Remainder) component of the MSTL equation:
<!-- Centered equation -->
$$y_t = T_t + S_t + R_t$$
<!-- Centered equation -->
- Anything that is $<3\sigma$ in the residuals can be regarded as noise in the data for which the model cannot account and is stored in the $R_t$ component. The decision for using $3\sigma$ came from some quick research conducted, and was validated by a lesson found published by <a href='https://online.stat.psu.edu/stat501/lesson/11/11.3#:~:text=The%20good%20thing%20about%20internally,is%20generally%20deemed%20an%20outlier.'>Penn State University</a>.

&nbsp; On outlier detection, there are only 2 outliers detected by the MSTL algorithm in this Time Series - one on 31 August 2013 and the other on 8 December 2013. This is certainly less than I expected but represents good news for the upcoming forecast model, as there are only 2 outliers that can cause the accuracy of any forecast for this time series to be adversely affected. Of course, this does depend on the length of the forecast horizon (more on that below). Since EDs are driven by emergencies, it's generally better to only have short window forecasts (for instance, my 2nd project at Healthscope was a 3-day forecast of Presentations for Healthscope's 8 EDs). Longer horizon forecasts for data that can change rapidly would inevitably lead to a poor forecast.

## Part 3: Differences between the hospitals based on the given data
An important lesson I learned at Healthscope was that each hospital is a microcosm of uniqueness when assessing patient activity - the assumptions that apply to one hospital don't tend to apply to others! On that note, using the `y-data` profiling reports found in the `/charts_reports` directory, these are some of the differences I can note:
- **Attendance/Admissions ratios**: When considering EDs, Attendance (or Presentations as we called them at Healthscope) to Admissions ratios are an important consideration, as it's quite routine practice to turn away patients for a variety of reasons (for instance, the patient's symptoms may not actually be urgent, or that the ED may already be full and may place themselves into a Bypass mode so that ambulances don't bring patients to that ED, etc.). For the hospitals listed in this dataset these were the ratios by site. As we can below, the ratios vary dramatically across each hospital!:
    - Royal Perth: 74.36%
    - Fremantle: 61.76%
    - Princess Margaret: 33.64%
    - King Edward: 23.91%
    - Sir Charles: 93.42%
    - Armadale: 33.73%
    - Swan District: 57.81%
    - Rockingham: 42.86%
    - Joondalup: 43.62%

There are also significant differences in the number of Triage 1 category admissions between sites:
<ul>
  <li>Royal Perth: 1438</li>
    <li>Fremantle: 268</li>
    <li>Princess Margaret: 80</li>
    <li>King Edward: 0</li>
    <li>Sir Charles: 783</li>
    <li>Armadale: 69</li>
    <li>Swan District: 109</li>
    <li>Rockingham: 32</li>
    <li>Joondalup: 152</li>
</ul>

From the above Triage 1 categories, I can see why a forecast model for the Royal Perth ED would be desirable. There are substantially more patients admitted there!

## Part 4: Forecast model for Triage Category 1 at Royal Perth Hospital
### Model's Drivers
- 1 year of training data for Triage Category 1 activity at Royal Perth Hospital over 2013-2014
- The trends and cyclical components over that 1 year

### Model's Limitations
- **Limited training data**: This is 1 year of training data that has been sufficient to capture the trends and cyclical effects over the course of that year only - which included the 2 outliers that the MSTL decomposition model uncovered. But beyond this 1 year of training data we don't have any other trends and cyclicalities that govern the activity of this ED. For instance, what if this 2013-2014 period was a year where activity levels were either lower or higher than in previous years? The model wouldn't have sufficient context for predicting the next forecast horizon, and would likely fail when generating the Out-Of-Sample forecasts, but would do well in the Out-Of-Sample Cross Validated _Predictions_, since it's testing it's own predictions against a limited training set!
- **No Exogenous features**: Under the hood, the STLForecast is basically an ARIMA model that predicts trends and cyclicalities that are infered from the training data. However, forecasting is a very complex and highly sensitive body of work that requires exogenous features to enrich the $\hat{y}$ predictor! For instance, at Healthscope when developing the 3-day ED Presentations forecast model, after extensive experimentation and exploration of the data, I was finally able to find that transfer_reason, type, arrival, triage_category, and value were the most effective 5 exogenous categorical features that would enrich the $\hat{y}$ predictor (Presentations) and reduce the MAE and MAPE measures of error during Cross Validation. Something of that nature would be needed here, as well.
- **Seasonality not included**: Since this is quite a simple ARIMA model, Seasonality is not included. However, the STL Decomposition EDA does detect the presence of seasonality in the data that varies over the time series.
- **No holiday effects**: Much like seasonality, holidays have a substantial effect on ED patient activity. During the development of the Healthscope ED Presentations model, it was found that the Christmas/NY period was the time when the most number of accidents tend to occur in the home that generate visits to the ED - especially falls from ladders. Those were quite common as more people were using their holidays to perform DIY repairs around the home, resulting in more accidents.
- **No parameter optimisation/differencing performed**: To determine the p,d,q parameters for any ARIMA, SARIMA/SARIMAX models, we normally need to run a full parameter optimization routine using the `pmdarima.arima.auto_arima` capability to determine the correct `order()` and, if applicable, `seasonal_order()` (P,D,Q,s) to use to fit the model. As part of this optimisation routine, we would also need to apply differencing such that y-axis values are 'differenced', so to speak, and we don't allow those values to negatively impact the forecast. Since I wanted to keep things simple, I didn't apply that parameter optimisation/differencing to this model, as it tends to be quite computationally expensive (since it aims to find the lowest AIC or BIC values using a stepwise parameter search or random walk), and can take some time to run, even for small datasets.
- **Too long a forecast horizon**: The ask in the case study was to predict 2015 activity based on 365 days of 2013-2014 data. This implies a forecast horizon of about 1 year. With the nature of Cross Validation to calculate the OOS prediction error, I was able to generate an Out-Of-Sample Forecast of a maximum of 358 days. Anything beyond this would raise errors, since there wasn't sufficient training data to calculate CV for a full 365 days. So, we can say it's just shy of 1 year! 
<br> &nbsp; But, as was discussed in the Outlier detection section, especially for EDs, it's best to only develop short window forecasts. This is especially so, given the very limited training data. However, in saying that, the Cross Validated accuracy for the simple 358 day forecast was:
```
Cross-validated Mean Absolute Error (MAE): 2.85
Cross-validated Mean Absolute Percentage Error (MAPE): 52.47%
```
&nbsp; This was also surprising for me, as I expected the error to be much larger given the lengthy forecast horizon!

**N.B.** If you're interested, all the output referenced in this README file can also be found in the following GitHub runner in the <a href="https://github.com/VikramsDataScience/medibank_case_study/actions/runs/11829105284/job/32960349828">Run Docker Container</a> section of the CI Workflow. This way you can verify that the output is real :)!
<br> &nbsp; For other wards, longer window forecasts will still have lower accuracy, but are still considered very achievable. For instance, my first project in Healthscope required developing 10-week forecast models that could predict patient activity for every time shift (AM, PM, Night Duty) for most of the wards (some wards, such as pediatrics, and a few others were deemed out-of-scope by the business) across Healthscope's 38 hospitals. 

## Part 5: Doctors required for Royal Perth ED on specified dates
Please refer to `src.doctors_required_royal_perth` for the code developed to answer this question.

**N.B.** The questions ask for the number of the doctors needed for the 1/1/14 and the 24/7/14. However, the latest date in the dataset is 30/06/2014 (output below is from the code in the `src.doctors_required_royal_perth`):
```
Max date in the dataset:  2014-06-30
```
Due to this limitation I've instead run the prediction for the 1/1/14 and the 30/06/14. I hope that's okay?

The code developed applied the following reasoning:
- A doctor can work up to 10 hours and treat 1 patient every 30 minutes
- The most important labour constraint is that the maximum number of patients a doctor can treat within a 10-hour shift would be:
<!-- Centered equation -->
$$\frac{10 \times 60}{30} = 20$$
<!-- Centered equation -->
- In my python implementation of the above calculation, I chose to use floor division to round down the constraint to the nearest integer so as to create a more conservative calculation, since a fatigued doctor is more likely to make critical mistakes.
- From here, I've constructed a for loop to iterate through the 2 required dates and generate the following output:
```
Attendance for dates of interest:
           Date  Attendance
184 2014-01-01         222
364 2014-06-30         234
No. of doctors required on 2014-01-01:  11
No. of doctors required on 2014-06-30:  12
```
