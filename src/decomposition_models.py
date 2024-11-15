from typing import TextIO
import pandas as pd
from statsmodels.tsa.seasonal import STL, MSTL
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Relative imports
from . import data_path, charts_reports_path

name = 'Royal Perth Triage Category 1'
preprocessed_df = pd.read_csv(data_path / "preprocessed_royal_perth_data.csv", 
                              parse_dates=['Date']).fillna(0)

min_date = preprocessed_df['Date'].min()
max_date = preprocessed_df['Date'].max()
preprocessed_df.index = pd.date_range(start=min_date, 
                          end=max_date, 
                          freq='D')
preprocessed_df = preprocessed_df.set_index('Date', verify_integrity=True)


######## Fit STL Decomposition model, generate plots, and save plots to storage ########
def STL_Decomposition(preprocessed_df) -> TextIO:
    print('Training STL Decomposition model...')
    # Fit STL Decomposition model by column names
    model_robust = STL(preprocessed_df['Tri_1'],
                        seasonal=7,
                        period=7,
                        robust=True)
    stl_robust = model_robust.fit()

    # Plot the results and export them to storage as HTML files
    fig = make_subplots(rows=1,
                        cols=1)

    fig.append_trace(
        go.Scatter(x=stl_robust.observed.index,
                y=stl_robust.observed.values,
                mode='lines',
                name='Observed'),
    row=1,
    col=1)

    fig.append_trace(
        go.Scatter(x=stl_robust.trend.index,
                y=stl_robust.trend.values,
                mode='lines',
                name='Trend'),
    row=1,
    col=1)

    fig.append_trace(
        go.Scatter(x=stl_robust.seasonal.index,
                y=stl_robust.seasonal.values,
                mode='lines',
                name='Seasonality'),
    row=1,
    col=1)

    fig.append_trace(
        go.Scatter(x=stl_robust.resid.index,
                y=stl_robust.resid.values,
                mode='markers',
                name='Residuals'),
    row=1,
    col=1)

    fig.update_layout(height=500,
                    title=f'STL Decomposition Model for {name}',
                    margin={'t':150},
                    title_x=0.5,
                    showlegend=True)

    # Export plots to storage as HTML files and replace spaces in col names with underscores in the file path so as not to break Pathing
    with open(charts_reports_path / f'STL_Plot_for_{name}.html'.replace(' ', '_'), 'w') as plot:
        plot.write(fig.to_html(include_plotlyjs='cdn'))


######## Fit MSTL Decomposition model to determine if there are outliers ########
def MSTL_Outlier_Detection(preprocessed_df) -> TextIO:
    """Outlier detection for MSTL has been defined as 3 Standard Deviations
    away from the mean in the Residuals component of the MSTL equation.
    If there are no dots on the resulting plot, we can infer no perceivable 
    outliers detected. For background and research, please refer to the README file.
    """

    print('Training MSTL Outlier Detection model...')

    mstl_robust = MSTL(preprocessed_df['Tri_1'],
                        periods=7)
    mstl = mstl_robust.fit()

    residuals = mstl.resid
    # Set outlier threshold to be 3 * Standard Deviation of the model's residuals
    threshold = 3 * np.std(residuals)
    outliers = np.abs(residuals) > threshold

    # Plot the results and export them to storage as HTML files
    fig = make_subplots(rows=1,
                        cols=1)

    fig.append_trace(
        go.Scatter(x=mstl.observed.index,
                y=mstl.observed.values,
                mode='lines',
                name='Observed'),
    row=1,
    col=1)

    fig.append_trace(
        go.Scatter(x=mstl.trend.index,
                y=mstl.trend.values,
                mode='lines',
                name='Trend'),
    row=1,
    col=1)

    fig.append_trace(
        go.Scatter(x=mstl.seasonal.index,
                y=mstl.seasonal.values,
                mode='lines',
                name='Seasonality'),
    row=1,
    col=1)

    fig.append_trace(
        go.Scatter(x=residuals[outliers].index,
                y=residuals[outliers],
                mode='markers',
                name='Outliers'),
    row=1,
    col=1)

    fig.update_layout(height=500,
                    title=f'MSTL Decomposition Model for {name} (Outlier Detection)',
                    margin={'t':150},
                    title_x=0.5,
                    showlegend=True)

    # Export plots to storage as HTML files and replace spaces in col names with underscores in the file path so as not to break Pathing
    with open(charts_reports_path / f'MSTL_Plot_for_{name}_(Outlier_Detection).html'.replace(' ', '_'), 'w') as plot:
        plot.write(fig.to_html(include_plotlyjs='cdn'))


STL_Decomposition(preprocessed_df=preprocessed_df)
MSTL_Outlier_Detection(preprocessed_df=preprocessed_df)