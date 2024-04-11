import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

df = pd.read_csv('DataSets/Bike_AdventureWorkDW.csv',header=None)

# print(df.to_string())
# print(df.values.tolist())
# print(len(df))

transactions = []
for i in range(0,len(df)):
    transactions.append([str(df.values[i,j]) for j in range(0,len(df.values[i]))])

# print(transactions)

min_n_products = 3
# print(40*7 / 13007)
# for example: 10 percent of daily transactions
min_support = 0.003
# we have no limit on the size of association rules
# so we set it to the longest transaction
# max_length = max([len(x) for x in transactions])
# print(max_length)
max_length = 3

from pyECLAT import ECLAT
# # # # create an instance of eclat
my_eclat = ECLAT(data = df, verbose=True)
# print(my_eclat)
# # # fit the algorithm
rule_indices, rule_supports = my_eclat.fit(min_support=min_support,
                                           min_combination=min_n_products,
                                         max_combination=max_length)
# # #
# print(rule_supports)
serie_res = pd.Series(data = rule_supports, index = rule_supports.keys())
# print(serie_res)
# #
df_supp = pd.DataFrame(serie_res)
df_supp = df_supp.reset_index()
df_supp.columns = ['Products','Support']
FinalOutput = df_supp.nlargest(n=2000, columns='Support')
print(FinalOutput.to_string())

