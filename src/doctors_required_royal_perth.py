import pandas as pd

# Relative imports
from . import charts_reports_path

df = pd.read_csv(charts_reports_path / "preprocessed_royal_perth_data.csv", 
                            parse_dates=['Date']).fillna(0)

# Convert 'Date' to a proper datetime format for filtering
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
print("Latest date available in the dataset: ", df['Date'].dt.date.max())

# Filter data for the specified dates
dates_of_interest = ['2014-01-01', '2014-06-30']
filtered_data = df[df['Date'].isin(pd.to_datetime(dates_of_interest))]

# Display the filtered data for those specific dates
print("\nAttendance for dates of interest: \n", filtered_data[['Date', 'Attendance']])

# Labour constraints for calculations using floor division (round down to nearest integer)
patients_per_doctor_per_day = (10 * 60) // 30

# Calculate the number of doctors needed for dates_of_interest
for date in dates_of_interest:
    attendance = filtered_data.loc[filtered_data['Date'] == pd.Timestamp(date), 'Attendance'].values[0]
    
    # Calculate labour requirements and round values (either up or down) to the nearest integer
    min_doctors_required = round((attendance / patients_per_doctor_per_day),
                                 ndigits=None)

    print(f"No. of doctors required on {date}: ", min_doctors_required)
