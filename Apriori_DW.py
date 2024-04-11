import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

df = pd.read_csv('DataSets/Bike_AdventureWorkDW.csv', header=None)
transactions = []

for i in range(0, len(df)):
    transactions.append([str(df.values[i, j]) for j in range(0, len(df.values[i]))])

rules = apriori(transactions=transactions, min_support=0.002, min_confidence=0.4, min_lift=0.15, min_length=3, max_length=3)
results = list(rules)

def inspect(results):
    Products = [', '.join(result.items) for result in results]
    supports = [result.support for result in results]
    confidences = [result.ordered_statistics[0].confidence for result in results]
    lifts = [result.ordered_statistics[0].lift for result in results]
    return list(zip(Products, supports, confidences, lifts))


result_df = pd.DataFrame(inspect(results), columns=['Products', 'Support', 'Confidence', 'Lift'])
output = result_df.nlargest(n=2000, columns='Lift')
print(output.to_string())

