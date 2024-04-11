import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# KNN Algorithm
df = pd.read_csv('DataSets\gene_expression.csv')
# print(df.head().to_string())
# print(df.info)
sns.scatterplot(x='Gene_One',y='Gene_Two',data=df)
# plt.show()
sns.scatterplot(x='Gene_One',y='Gene_Two',hue='Cancer_Present',data=df)
# plt.show()

plt.figure(dpi=150)
# alpha is transparency
sns.scatterplot(x='Gene_One',y='Gene_Two',hue='Cancer_Present',data=df,alpha=0.5)
# plt.show()

# when we want to select a section from chart
sns.scatterplot(x='Gene_One',y='Gene_Two',hue='Cancer_Present',data=df,alpha=0.5)
plt.xlim(2,6)
plt.ylim(3,10)
plt.legend(loc=(1.1,0.5))
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Cancer_Present',axis=1)
y = df['Cancer_Present']
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=45)

# rescale is necessary for KNN
scalar = StandardScaler()
scaled_x_train = scalar.fit_transform(x_train)
scaled_x_test = scalar.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(scaled_x_train,y_train)

full_test = pd.concat([x_test,y_test],axis=1)
len(full_test)
sns.scatterplot(x='Gene_One',y='Gene_Two',hue='Cancer_Present',data=full_test,alpha=0.7)
# plt.show()

# Model Evaluation
y_pred =knn_model.predict(scaled_x_test)
#
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
#
test_error_rates = []
for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(x_train.values,y_train.values)

    y_pred_test = knn_model.predict(scaled_x_test)
    test_error = 1 - knn_model.score(y_test,y_pred_test)
    print(test_error)
    test_error_rates.append(test_error)
