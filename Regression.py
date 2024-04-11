import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import messagebox as msg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


houseDataEntryForm = Tk()
houseDataEntryForm.title('House Price Prediction Form')
houseDataEntryForm.geometry('550x600')
right = int(houseDataEntryForm.winfo_screenwidth()/2-550/2)
down = int(houseDataEntryForm.winfo_screenheight()/2-600/2)
houseDataEntryForm.geometry('+{}+{}'.format(right,down))
houseDataEntryForm.iconbitmap('images/house.ico')

housePredictFrame = tkinter.Frame(houseDataEntryForm)
housePredictFrame.pack()

housePredictInfo = tkinter.LabelFrame(housePredictFrame, text='House Information')
housePredictInfo.grid(row=0, column=0, sticky='news', padx=20, pady=10)

lblAddress= Label(housePredictInfo, text='Address')
lblAddress.grid(row=1, column=0, padx=10, pady=5,sticky='w')
txtCityCode = IntVar()
entAddress = ttk.Entry(housePredictInfo, width=10, textvariable=txtCityCode)
entAddress.grid(row=1, column=1, padx=10, pady=5)

lblElevator = Label(housePredictInfo, text='Elevator:')
lblElevator.grid(row=2, column=0, padx=10, pady=5,sticky='w')
intElevator = IntVar()
entWithElevator = ttk.Radiobutton(housePredictInfo, width=20, text='Yes', variable=intElevator, value=True)
entWithElevator.grid(row=2, column=1)
entNoElevator = ttk.Radiobutton(housePredictInfo, width=20, text='No', variable=intElevator, value=False)
entNoElevator.grid(row=2, column=2)

lblFloor = Label(housePredictInfo, text='Floor:(-1 to 30)')
lblFloor.grid(row=3, column=0, padx=10, pady=5,sticky='w')
IntFloor =IntVar()
entFloor = ttk.Entry(housePredictInfo, width=10, textvariable=IntFloor)
entFloor.grid(row=3, column=1, padx=10, pady=5)

lblArea = Label(housePredictInfo, text='Area:(Square Feet)')
lblArea.grid(row=4, column=0, padx=10, pady=5,sticky='w')
IntArea = IntVar()
entArea = ttk.Entry(housePredictInfo, width=10, textvariable=IntArea)
entArea.grid(row=4, column=1, padx=10, pady=5)

lblParking = Label(housePredictInfo, text='Parking:')
lblParking.grid(row=5, column=0, padx=10, pady=5,sticky='w')
intParking = IntVar()
entWithParking = ttk.Radiobutton(housePredictInfo, width=20, text='Yes', variable=intParking, value=True)
entWithParking.grid(row=5, column=1)
entNoParking = ttk.Radiobutton(housePredictInfo, width=20, text='No', variable=intParking, value=False)
entNoParking.grid(row=5, column=2)

lblRoom = Label(housePredictInfo, text='Number Of Room')
lblRoom.grid(row=6, column=0, padx=10, pady=5,sticky='w')
intRoom = IntVar()
entRoom = ttk.Combobox(housePredictInfo, width=10, textvariable=intRoom)
entRoom["values"] = ["0", "1", "2", "3","4"]
entRoom.grid(row=6, column=1, padx=10, pady=5)

lblWarehouse = Label(housePredictInfo, text='Warehouse:')
lblWarehouse.grid(row=7, column=0, padx=10, pady=5,sticky='w')
intWarehouse= IntVar()
entWithWarehouse = ttk.Radiobutton(housePredictInfo, width=20, text='Yes', variable=intWarehouse, value=True)
entWithWarehouse.grid(row=7, column=1)
entNoWarehouse = ttk.Radiobutton(housePredictInfo, width=20, text='No', variable=intWarehouse, value=False)
entNoWarehouse.grid(row=7, column=2)

lblYearOfConstruction = Label(housePredictInfo, text='Year Of Construction')
lblYearOfConstruction.grid(row=8, column=0, padx=10, pady=5,sticky='w')
intYearOfConstruction = IntVar()
entYearOfConstruction = ttk.Entry(housePredictInfo, width=10, textvariable=intYearOfConstruction)
entYearOfConstruction.grid(row=8, column=1, padx=10, pady=5)



# entAddress["values"] = ["317-دروس"]
# entAddress.grid(row=7, column=3, padx=10, pady=5)

def predFunction_NB():
    df = pd.read_csv("DataSets/HouseNew.csv")
    df['Address'].fillna('نامشخص', inplace=True)
    df['Floor'].fillna(100, inplace=True)
    # print(df.to_string())

    df1 = df['Price_Category'] = pd.cut(df['Price'], bins=[0, 3000000000, 10000000000, np.inf],
                                        labels=['Low', 'Medium', 'High'])

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df['City_Code'] = label_encoder.fit_transform(df['Address'])
    df['City_Code'] = df['City_Code'].astype('category')

    X = df.drop('Price_Category', axis=1)
    y = df['Price_Category']
    # print(X)

    X1 = X.drop('Price', axis=1)
    y1 = df['Price_Category']
    # print(X1.to_string())

    X2 = X1.drop('Address', axis=1)
    # print(X2.to_string())
    y2 = df['Price_Category']
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.25, random_state=35)

    from sklearn.naive_bayes import GaussianNB
    Classifier = GaussianNB()
    Classifier.fit(X_train.values, y_train.values)
    GaussianNB()

    elevator = bool(intElevator.get())
    floor = float(IntFloor.get())
    area = int(IntArea.get())
    parking = bool(intParking.get())
    room = int(intRoom.get())
    warehouse = bool(intWarehouse.get())
    yearOfConstruction = int(intYearOfConstruction.get())
    Address =txtCityCode.get()
    # floor_str = txtFloor.get()
    # if floor_str.strip():  # Check if the input is not empty
    #     floor = int(floor_str)
    # else:
    #     pass
    # floor_str = txtFloor.get()
    # if floor_str.isdigit():  # Check if the input is composed of digits
    #     floor = int(floor_str)
    # else:
    #   pass
        # Handle the case when the input is empty (e.g., show a message to the user)

    #
    # prediction = Classifier.predict([[True,1,311,True,4,True,1396]])
    prediction = Classifier.predict([[elevator,floor,area,parking,room,warehouse,yearOfConstruction,Address]])
    # print(prediction)
    prediction_prob = Classifier.predict_proba([[elevator,floor,area,parking,room,warehouse,yearOfConstruction,Address]])
    # prediction_prob_rounded = [round(prob * 100, 2) for prob in prediction_prob[0]]
    # print(prediction_prob)
# # return prediction,prediction_prob
# #
    housePredictResult = tkinter.LabelFrame(housePredictFrame, text='NB Prediction Result')
    housePredictResult.grid(row=12, column=0,sticky='news',padx=5, pady=5)
# # #
    lblResult = Label(housePredictResult)
    combined_output = 'House Price Prediction: ' + str(prediction) + '       ' + 'Percentage: ' + str(prediction_prob)
    lblResult.config(text=combined_output,background='gray',highlightcolor='blue')
    lblResult.grid(row=13, column=0, padx=5, pady=10)

def PredFunction_KNN():
    df = pd.read_csv("DataSets/HouseNew.csv")
    df['Address'].fillna('نامشخص', inplace=True)
    df['Floor'].fillna(100, inplace=True)
    # print(df.to_string())

    df1 = df['Price_Category'] = pd.cut(df['Price'], bins=[0, 3000000000, 10000000000, np.inf],
                                        labels=['Low', 'Medium', 'High'])

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df['City_Code'] = label_encoder.fit_transform(df['Address'])
    df['City_Code'] = df['City_Code'].astype('category')

    X = df.drop('Price_Category', axis=1)
    y = df['Price_Category']
    # print(X)

    X1 = X.drop('Price', axis=1)
    y1 = df['Price_Category']
    # print(X1.to_string())

    X2 = X1.drop('Address', axis=1)
    # print(X2.to_string())
    y2 = df['Price_Category']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.25, random_state=32)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_x_train = scaler.fit_transform(X_train)
    scaled_x_test = scaler.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    #
    knn_model = KNeighborsClassifier(n_neighbors=7)
    knn_model.fit(scaled_x_train,y_train)
    # print(knn_model.score(scaled_x_test, y_test))


    elevator = bool(intElevator.get())
    floor = float(IntFloor.get())
    area = int(IntArea.get())
    parking = bool(intParking.get())
    room = int(intRoom.get())
    warehouse = bool(intWarehouse.get())
    yearOfConstruction = int(intYearOfConstruction.get())
    Address =txtCityCode.get()

    prediction = knn_model.predict([[elevator,floor,area,parking,room,warehouse,yearOfConstruction,Address]])
    # print(prediction)
    prediction_prob = knn_model.predict_proba([[elevator,floor,area,parking,room,warehouse,yearOfConstruction,Address]])
    # prediction_prob_rounded = [round(prob * 100, 2) for prob in prediction_prob[0]]
    # print(prediction_prob_rounded)

    housePredictResult = tkinter.LabelFrame(housePredictFrame, text='KNN Prediction Result')
    housePredictResult.grid(row=14, column=0, sticky='news', padx=5, pady=5)
    # # #
    lblResult = Label(housePredictResult)
    combined_output = 'House Price Prediction: ' + str(prediction) + '       ' + 'Percentage: ' + str(
        prediction_prob)
    lblResult.config(text=combined_output, background='gray', highlightcolor='blue')
    lblResult.grid(row=15, column=0, padx=5, pady=10)
def PredFunction_Regression():
    df = pd.read_csv("DataSets/HouseNew.csv")
    df['Address'].fillna('نامشخص', inplace=True)
    df['Floor'].fillna(100, inplace=True)
    # print(df.dtypes)
    # print(df.isna().sum())


    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df['City_Code'] = label_encoder.fit_transform(df['Address'])
    df['City_Code'] = df['City_Code'].astype('category')

    X = df.drop('Price', axis=1)
    y = df['Price']
    # print(X)

    X1 = X.drop('Address', axis=1)
    y1 = df['Price']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=32)
    # print(X_train)
    # # #
    from sklearn.linear_model import LinearRegression
    # # #
    # # # # print(help(LinearRegression))
    # # #
    model = LinearRegression()
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    # print(test_predictions)
    final_model = LinearRegression()
    final_model.fit(X1.values, y1.values)
    # y_hat = final_model.predict(X.values)
    # print(y_hat)
    # coeff_df = pd.DataFrame(final_model.coef_, X.columns, columns=['Coefficient'])
    # print(coeff_df)
    # # # #
    elevator = bool(intElevator.get())
    floor = float(IntFloor.get())
    area = int(IntArea.get())
    parking = bool(intParking.get())
    room = int(intRoom.get())
    warehouse = bool(intWarehouse.get())
    yearOfConstruction = int(intYearOfConstruction.get())
    Address =txtCityCode.get()

    newdata = [[elevator,floor,area,parking,room,warehouse,yearOfConstruction,Address]]
    test = final_model.predict(newdata)
    # print(test)

    housePredictResult = tkinter.LabelFrame(housePredictFrame, text='Regression Prediction Result')
    housePredictResult.grid(row=15, column=0, sticky='news', padx=5, pady=5)
    # # #
    lblResult = Label(housePredictResult)
    lblResult.config(text='House Price Prediction: ' + str(test), background='gray', highlightcolor='blue')
    lblResult.grid(row=16, column=0, padx=5, pady=10)

housePredictCheckList = tkinter.LabelFrame(housePredictFrame, text='Select Your Desired Algorithm!')
housePredictCheckList.grid(row=9, column=0, sticky='news', padx=20, pady=10)

var_NB = BooleanVar()
checkNB = Checkbutton(housePredictCheckList, text='Naive Bayes', variable=var_NB)
checkNB.grid(row=10, column=0, padx=10, pady=2)

var_KNN = BooleanVar()
checkKNN = Checkbutton(housePredictCheckList, text='KNN', variable=var_KNN)
checkKNN.grid(row=10, column=1,padx=10, pady=2)

var_Regression = BooleanVar()
checkRegression = Checkbutton(housePredictCheckList, text='Regression', variable=var_Regression)
checkRegression.grid(row=10, column=2,padx=10, pady=2)

def checklistFunction():
    if var_NB.get():
       predFunction_NB()
    if var_KNN.get():
       PredFunction_KNN()
    if var_Regression.get():
       PredFunction_Regression()

btnPredict = ttk.Button(housePredictFrame, text='Predict', width=10, command=checklistFunction, style="Custom.TButton")
btnPredict.grid(row=11, column=0, padx=10, pady=2,sticky='news')

houseDataEntryForm.mainloop()
