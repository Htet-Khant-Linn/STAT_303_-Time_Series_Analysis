"""
Date: 2026-02-04
Owner: Htet Khant Linn
Week: 1    

"""
#%%
import numpy as np


#%% Multiplication
first_array = np.array([[1,2,3],[4,5,6]])

A = np.array([[1,2], [3,4]])
B = np.array([[3, 4], [5,7]])

C = A @ B

C_element = A * B

D = np.dot(A, B)

E = np.multiply(A, B)

#%% Adding and substraction

sum1 = A + B

sub1 = A - B

sub2 = np.subtract(A, B)

sum2 = np.sum(A)

Borad_num = A + 3

board_matrix = np.array([[3,3], [3,3]])

#%% Dividing

D = np.divide([12, 14,15], 5)

D_floor = np.floor_divide([12, 14, 15], 5)

sq = np.math.sqrt(10)


#%% Distribution

normal_d = np.random.standard_normal((3,4))

uniform_d = np.random.uniform(1, 12, (3, 4))

# General float No.

np.random.rand()

# Generate IntegerNo.

Random_Int = np.random.randint(1, 50, (2, 5))


zero_matrix = np.zeros((3,4))

one_matrix = np.ones((3,4))

Filter_Ar = np.logical_and(Random_Int>30, Random_Int<50)

Filtered_int = Random_Int[Filter_Ar]


#%%

Data_N = np.array([1,3, 4, 5, 7, 9])

Mean_N = np.mean(Data_N)

Median_N = np.median(Data_N)

Var_N = np.var(Data_N)

SD_N = np.std(Data_N)

