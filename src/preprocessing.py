import pandas as pd

# Relative imports
from . import data_path, charts_reports_path

df = pd.read_csv(data_path / "govhack3.csv", header=[0, 1])

# Preprocess data to separate DataFrames by hospital
royal_perth_data = {'Date': df[('Unnamed: 0_level_0', 'Date')],
        'Attendance': df[('Royal Perth Hospital', 'Attendance')],
        'Admissions': df[('Unnamed: 2_level_0', 'Admissions')],
        'Tri_1': df[('Unnamed: 3_level_0', 'Tri_1')],
        'Tri_2': df[('Unnamed: 4_level_0', 'Tri_2')],
        'Tri_3': df[('Unnamed: 5_level_0', 'Tri_3')],
        'Tri_4': df[('Unnamed: 27_level_0', 'Tri_4')],
        'Tri_5': df[('Unnamed: 28_level_0', 'Tri_5')]
}

fremantle_data = {
                'Date': df[('Unnamed: 0_level_0', 'Date')],
        'Attendance': df[('Fremantle Hospital', 'Attendance')],
        'Admissions': df[('Unnamed: 9_level_0', 'Admissions')],
        'Tri_1': df[('Unnamed: 10_level_0', 'Tri_1')],
        'Tri_2': df[('Unnamed: 11_level_0', 'Tri_2')],
        'Tri_3': df[('Unnamed: 12_level_0', 'Tri_3')],
        'Tri_4': df[('Unnamed: 13_level_0', 'Tri_4')],
        'Tri_5': df[('Unnamed: 14_level_0', 'Tri_5')]
}

princess_margaret_data = {
                'Date': df[('Unnamed: 0_level_0', 'Date')],
        'Attendance': df[('Princess Margaret Hospital For Children', 'Attendance')],
        'Admissions': df[('Unnamed: 16_level_0', 'Admissions')],
        'Tri_1': df[('Unnamed: 17_level_0', 'Tri_1')],
        'Tri_2': df[('Unnamed: 18_level_0', 'Tri_2')],
        'Tri_3': df[('Unnamed: 19_level_0', 'Tri_3')],
        'Tri_4': df[('Unnamed: 20_level_0', 'Tri_4')],
        'Tri_5': df[('Unnamed: 21_level_0', 'Tri_5')]
}

king_edward_data = {
                'Date': df[('Unnamed: 0_level_0', 'Date')],
        'Attendance': df[('King Edward Memorial Hospital For Women', 'Attendance')],
        'Admissions': df[('Unnamed: 23_level_0', 'Admissions')],
        'Tri_1': df[('Unnamed: 24_level_0', 'Tri_1')],
        'Tri_2': df[('Unnamed: 25_level_0', 'Tri_2')],
        'Tri_3': df[('Unnamed: 26_level_0', 'Tri_3')],
        'Tri_4': df[('Unnamed: 27_level_0', 'Tri_4')],
        'Tri_5': df[('Unnamed: 28_level_0', 'Tri_5')]
}

sir_charles_data = {
                'Date': df[('Unnamed: 0_level_0', 'Date')],
        'Attendance': df[('Sir Charles Gairdner Hospital', 'Attendance')],
        'Admissions': df[('Unnamed: 30_level_0', 'Admissions')],
        'Tri_1': df[('Unnamed: 31_level_0', 'Tri_1')],
        'Tri_2': df[('Unnamed: 32_level_0', 'Tri_2')],
        'Tri_3': df[('Unnamed: 33_level_0', 'Tri_3')],
        'Tri_4': df[('Unnamed: 34_level_0', 'Tri_4')],
        'Tri_5': df[('Unnamed: 35_level_0', 'Tri_5')]
}

armadale_data = {
                'Date': df[('Unnamed: 0_level_0', 'Date')],
        'Attendance': df[('Armadale/Kelmscott District Memorial Hospital', 'Attendance')],
        'Admissions': df[('Unnamed: 37_level_0', 'Admissions')],
        'Tri_1': df[('Unnamed: 38_level_0', 'Tri_1')],
        'Tri_2': df[('Unnamed: 39_level_0', 'Tri_2')],
        'Tri_3': df[('Unnamed: 40_level_0', 'Tri_3')],
        'Tri_4': df[('Unnamed: 41_level_0', 'Tri_4')],
        'Tri_5': df[('Unnamed: 42_level_0', 'Tri_5')]
}

swan_district_data = {
                'Date': df[('Unnamed: 0_level_0', 'Date')],
        'Attendance': df[('Swan District Hospital', 'Attendance')],
        'Admissions': df[('Unnamed: 44_level_0', 'Admissions')],
        'Tri_1': df[('Unnamed: 45_level_0', 'Tri_1')],
        'Tri_2': df[('Unnamed: 46_level_0', 'Tri_2')],
        'Tri_3': df[('Unnamed: 47_level_0', 'Tri_3')],
        'Tri_4': df[('Unnamed: 48_level_0', 'Tri_4')],
        'Tri_5': df[('Unnamed: 49_level_0', 'Tri_5')]
}

rockingham_data = {
                'Date': df[('Unnamed: 0_level_0', 'Date')],
        'Attendance': df[('Rockingham General Hospital', 'Attendance')],
        'Admissions': df[('Unnamed: 51_level_0', 'Admissions')],
        'Tri_1': df[('Unnamed: 52_level_0', 'Tri_1')],
        'Tri_2': df[('Unnamed: 53_level_0', 'Tri_2')],
        'Tri_3': df[('Unnamed: 54_level_0', 'Tri_3')],
        'Tri_4': df[('Unnamed: 55_level_0', 'Tri_4')],
        'Tri_5': df[('Unnamed: 56_level_0', 'Tri_5')]
}

joondalup_data = {
                'Date': df[('Unnamed: 0_level_0', 'Date')],
        'Attendance': df[('Joondalup Health Campus', 'Attendance')],
        'Admissions': df[('Unnamed: 58_level_0', 'Admissions')],
        'Tri_1': df[('Unnamed: 59_level_0', 'Tri_1')],
        'Tri_2': df[('Unnamed: 60_level_0', 'Tri_2')],
        'Tri_3': df[('Unnamed: 61_level_0', 'Tri_3')],
        'Tri_4': df[('Unnamed: 62_level_0', 'Tri_4')],
        'Tri_5': df[('Unnamed: 63_level_0', 'Tri_5')]
}

# Create JSON dicts to map hospital data
hospital_dfs = {}
hospital_data_maps = {
    'royal_perth': royal_perth_data,
    'fremantle': fremantle_data,
    'princess_margaret': princess_margaret_data,
    'king_edward': king_edward_data,
    'sir_charles': sir_charles_data,
    'armadale': armadale_data,
    'swan_district': swan_district_data,
    'rockingham': rockingham_data,
    'joondalup': joondalup_data
}

# Convert to DF and save to storage for downstream modules
for hospital_name, data in hospital_data_maps.items():
        df = (pd.DataFrame(data)
                .set_index('Date', verify_integrity=True)
                .to_csv(data_path / f"preprocessed_{hospital_name}_data.csv"))
        