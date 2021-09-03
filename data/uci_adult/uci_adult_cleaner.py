"""
UCI Adult data set
Turns categorical data into continuous data
"""

import csv
from os.path import join as pjoin

DATA_DIR = './'
COLUMNS = [ "Age",
            ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"],
            "Fnlwgt",
            ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
            "Education-num",
            ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
            ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
            ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
            ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
            ["Female", "Male"],
            "Capital-gain",
            "Capital-loss",
            "Hours-per-week",
            ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
            ["<=50K", ">50K"]
        ]


HEADER = ["age", "education", "ethnicity (white::other)", "gender (female::male)", "hours-per-week", "income (<=50k::>50k)"]

def categorical_to_cont(row):
    row_cont = []
    enabled = [0, 4, 8, 9, 12, 14]
    for el in enabled:
        if not isinstance(COLUMNS[el], list):
            row_cont.append(row[el])
        else:
            row_cont.append(1) if row[el] == COLUMNS[el][0] else row_cont.append(0)
    return row_cont


if __name__ == '__main__':
    raw_data = pjoin(DATA_DIR, 'adult.csv')
    cleaned_data = []
    print("Cleaning data")

    with open(raw_data) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skips the first row, which is the header
        for row in reader:
            cleaned_data.append(categorical_to_cont(row))

    with open('adult_clean.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(HEADER)
        for row in cleaned_data:
            writer.writerow(row)

    print("Done")





