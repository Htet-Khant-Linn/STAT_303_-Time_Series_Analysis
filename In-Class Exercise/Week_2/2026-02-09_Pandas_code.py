# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 13:45:25 2026

@author: Si Thu Aung
"""

import pandas as pd

"""
Series

"""

Age = pd.Series([10, 20, 30, 40], index=['age1', 'age2', 'age3', 'age4'])

Age.age3

Age.age1

# Filtering values of the Series
Filtered_Age = Age[Age>10]

Filtered_Age2 = Age[(Age>10) & (Age<40)]

Filtered_Age3 = Age[Age.between(20, 30)]

# Calling values of the Series
Age.values

# Calling indexes of the Series
Age.index


"""
DataFrame

"""
import numpy as np

# Creating DataFrame

DF = np.array([[20, 10, 8], [25, 8, 10], [27, 5, 3], [30, 9, 7]])

Data_set = pd.DataFrame(DF)

Data_set = pd.DataFrame(DF, index=['S1', 'S2', 'S3', 'S4'])

Data_set = pd.DataFrame(DF, index=['S1', 'S2', 'S3', 'S4'], columns=['Age', 'Grade1', 'Grade2'])

Data_set['Grade3'] = [9, 6, 7, 10]

# Indexing the DataFrame
# .loc and iloc

# .loc is label-based indexing and include the end
# .iloc is interger-based indexing and exclude the end

Data_set.loc['S1']

Data_set.loc['S1':'S2']

Data_set.iloc[1, 2]

Data_set.iloc[:, 0]

Data_set.iloc[:, 3]

Data_set.iloc[:, 1]

Data_set.iloc[:, 1:3]

Data_set.iloc[:, :3]

Data_set.iloc[1:2, :]

Data_set.iloc[:3, :]

Data_set.iloc[3, :]

Data_set.iloc[:, :-1]

Filtered_Data_set = Data_set.iloc[:, 1:3]

# Drop and replace

Data_set_drop = Data_set.drop('Grade1', axis=1)

Data_set_change1 = Data_set.replace(10, 12)

Data_set_change2 = Data_set.replace({30:10, 9:30})

Data_set.head()

Data_set.head(2)

Data_set.tail(2)

# Sorting values of the DataFrame

Data_set.sort_values(by=['Grade1'], ascending=True)

Data_set.sort_values(by=['Age'], ascending=False)

Data_set.sort_index(axis=0, ascending=True)

Data_set.sort_index(axis=0, ascending=False)
# rearranges data based on its index labels rather than values

# Importing Dataset

DF = pd.read_csv("household_electricity_consumption.csv")


























