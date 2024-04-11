import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import messagebox as msg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

hepatitDataEntryForm = Tk()
hepatitDataEntryForm.title('Hepatit C Prediction Form')
hepatitDataEntryForm.geometry('700x700')
right = int(hepatitDataEntryForm.winfo_screenwidth()/2-700/2)
down = int(hepatitDataEntryForm.winfo_screenheight()/2-700/2)
hepatitDataEntryForm.geometry('+{}+{}'.format(right,down))
hepatitDataEntryForm.iconbitmap('images/health.ico')

hepatitPredictFrame = tkinter.Frame(hepatitDataEntryForm)
hepatitPredictFrame.pack()

# hepatitPredictDesc = tkinter.LabelFrame(hepatitPredictFrame, text='Welcome to Hepatit Predition Page!')
# hepatitPredictDesc.grid(row=0, column=0, sticky='news', padx=20, pady=5)
#
# lblTitle = Label(hepatitPredictDesc, text='Please enter all the information correctly and press the predict button to see the result!')
# lblTitle.grid(row=1, column=0, padx=10, pady=10)

healthCarePredictInfo = tkinter.LabelFrame(hepatitPredictFrame, text='Patient Information')
healthCarePredictInfo.grid(row=2, column=0, sticky='news', padx=20, pady=10)

lblAge = Label(healthCarePredictInfo, text='Age:')
lblAge.grid(row=3, column=0, padx=10, pady=5,sticky='w')
txtAge =StringVar()
entAge = ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtAge)
entAge.grid(row=3, column=1, padx=10, pady=5)

lblSex = Label(healthCarePredictInfo, text='Sex:')
lblSex.grid(row=4, column=0, padx=10, pady=5,sticky='w')
intSex = IntVar()
entSexFemale = ttk.Radiobutton(healthCarePredictInfo, width=20, text='Female', variable=intSex, value=1)
entSexFemale.grid(row=4, column=1)
entSexMale = ttk.Radiobutton(healthCarePredictInfo, width=20, text='Male', variable=intSex, value=2)
entSexMale.grid(row=4, column=2)

lblALB = Label(healthCarePredictInfo, text='Albumin: Normal Range(34-54)')
lblALB.grid(row=5, column=0, padx=10, pady=5,sticky='w')
txtALB = StringVar()
entALB = ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtALB)
entALB.grid(row=5, column=1, padx=10, pady=5)

lblALP = Label(healthCarePredictInfo, text='Alkaline Phosphatase: Normal Range(44-147)')
lblALP.grid(row=6, column=0, padx=10, pady=5,sticky='w')
txtALP = StringVar()
entALP = ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtALP)
entALP.grid(row=6, column=1, padx=10, pady=5)

lblALT = Label(healthCarePredictInfo, text='Alanine Aminotransferase: Normal Range(7-56)')
lblALT.grid(row=7, column=0, padx=10, pady=5,sticky='w')
txtALT = StringVar()
entALT = ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtALT)
entALT.grid(row=7, column=1, padx=10, pady=5)

lblAST = Label(healthCarePredictInfo, text='Aspartate Aminotransferase: Normal Range(7-56)')
lblAST.grid(row=8, column=0, padx=10, pady=5,sticky='w')
txtAST = StringVar()
entAST = ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtAST)
entAST.grid(row=8, column=1, padx=10, pady=5)

lblBIL = Label(healthCarePredictInfo, text='Bilirubin: Normal Range(5-20)')
lblBIL.grid(row=9, column=0, padx=10, pady=5,sticky='w')
txtBIL = StringVar()
entBIL = ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtBIL)
entBIL.grid(row=9, column=1, padx=10, pady=5)

lblCHE = Label(healthCarePredictInfo, text='Serum Cholinesterase: Normal Range(8-18)')
lblCHE.grid(row=10, column=0, padx=10, pady=5,sticky='w')
txtCHE = StringVar()
entCHE = ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtCHE)
entCHE.grid(row=10, column=1, padx=10, pady=5)

lblCHOL = Label(healthCarePredictInfo, text='Cholesterol : Normal Range(125-200)')
lblCHOL.grid(row=11, column=0, padx=10, pady=5,sticky='w')
txtCHOL = StringVar()
entCHOL= ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtCHOL)
entCHOL.grid(row=11, column=1, padx=10, pady=5)

lblCREA = Label(healthCarePredictInfo, text='Creatinine  : Normal Range(59-135)')
lblCREA.grid(row=12, column=0, padx=10, pady=5,sticky='w')
txtCREA = StringVar()
entCREA= ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtCREA)
entCREA.grid(row=12, column=1, padx=10, pady=5)

lblGGT = Label(healthCarePredictInfo, text='γ-glutamyl Transferase: Normal Range(5-40)')
lblGGT.grid(row=13, column=0, padx=10, pady=5,sticky='w')
txtGGT = StringVar()
entGGT= ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtGGT)
entGGT.grid(row=13, column=1, padx=10, pady=5)

lblPROT = Label(healthCarePredictInfo, text='protein: Normal Range(6-8.3)')
lblPROT.grid(row=14, column=0, padx=10, pady=5,sticky='w')
txtPROT = StringVar()
entPROT= ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtPROT)
entPROT.grid(row=14, column=1, padx=10, pady=5)

def predFunction_NB():
    df = pd.read_csv('DataSets/hcv-2020.csv')
    df.fillna(df.mean(numeric_only=True), inplace=True)

    X = df.iloc[:, 0:12]
    y = df.iloc[:, 12]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    from sklearn.naive_bayes import GaussianNB
    Classifier = GaussianNB()
    Classifier.fit(X_train.values, y_train.values)
    GaussianNB()

    age = int(txtAge.get())
    sex = intSex.get()
    ALB = float(txtALB.get())
    ALP = float(txtALP.get())
    ALT = float(txtALT.get())
    AST = float(txtAST.get())
    BIL = float(txtBIL.get())
    CHE = float(txtCHE.get())
    CHOL = float(txtCHOL.get())
    CREA = float(txtCREA.get())
    GGT = float(txtGGT.get())
    PROT = float(txtPROT.get())

    prediction = Classifier.predict([[age,sex,ALB,ALP,ALT,AST,BIL,CHE,CHOL,CREA,GGT,PROT]])
    # print(prediction)
    prediction_prob = Classifier.predict_proba([[age,sex,ALB,ALP,ALT,AST,BIL,CHE,CHOL,CREA,GGT,PROT]])
    prediction_prob_rounded = [round(prob * 100, 2) for prob in prediction_prob[0]]
    # print(prediction_prob)
    # return prediction,prediction_prob

    if prediction == 0:
        # prediction_prob_rounded = [round(prob * 100, 2) for prob in prediction_prob[0]]
        # combined_output = f'Viable blood donors: {prediction}\n' + \
        #                   f'Percentage: {prediction_prob_rounded[0]}%'
        combined_output = 'Viable blood donors: ' + str(prediction) + '       ' + 'Percentage: ' + str(prediction_prob_rounded)
        # msg.showinfo("HCV Result", combined_output)
    elif prediction == 1:
        combined_output = 'Suspect Blood Donor: ' + str(prediction) + '       ' +'Percentage: ' + str(prediction_prob_rounded)
        # msg.showinfo("HCV Result", combined_output)
    elif prediction == 2:
        combined_output = 'Hepatitis: ' + str(prediction) + '       ' +'Percentage: ' + str(prediction_prob_rounded)
        # msg.showinfo("HCV Result", combined_output)
    elif prediction == 3:
        combined_output = 'Fibrosis: ' + str(prediction) + '       ' +'Percentage: ' + str(prediction_prob_rounded)
        # msg.showinfo("HCV Result", combined_output)
    else:
        combined_output = 'Cirrhosis: ' + str(prediction) + '       ' + 'Percentage: ' + str(prediction_prob_rounded)
        # msg.showinfo("HCV Result", combined_output)
    healthCarePredictResult = tkinter.LabelFrame(hepatitPredictFrame, text='NB Prediction Result')
    healthCarePredictResult.grid(row=20, column=0,sticky='news',padx=5, pady=5)

    lblResult = Label(healthCarePredictResult)
    lblResult.config(text=combined_output,background='gray',highlightcolor='blue')
    lblResult.grid(row=21, column=0, padx=5, pady=10)

def resetFormFunction():
    txtAge.set("")
    txtALT.set("")
    txtALP.set("")
    txtALB.set("")
    txtBIL.set("")
    txtAST.set("")
    txtCHE.set("")
    txtCHOL.set("")
    txtCREA.set("")
    txtGGT.set("")
    txtPROT.set("")
def PredFunction_KNN():
    df = pd.read_csv('DataSets/hcv-2020.csv')

    # print(df.shape)
    # print(df.head().to_string())
    # print(df.info)
    # print(df.isna().sum())
    df.fillna(df.mean(numeric_only=True),inplace=True)
    # print(df.isna().sum())

    x = df.iloc[:,0:12]
    y = df.iloc[:,12]

    # print(y.value_counts())
    # print(x['Sex'].value_counts())
    # print(x['ALB'].value_counts())
    # print(x['ALP'].value_counts())
    # print(x['ALT'].value_counts())
    # print(x['AST'].value_counts())
    # print(x['BIL'].value_counts())
    # print(x['CHE'].value_counts())
    # print(x['CHOL'].value_counts())
    # print(x['CREA'].value_counts())
    # print(x['GGT'].value_counts())
    # print(x['PROT'].value_counts())

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
    # print(f"x_train:{x_train.shape}")
    # print(f"y_train:{y_train.shape}")
    # print(f"x_test:{x_test.shape}")
    # print(f"y_test:{y_test.shape}")



    # classifier = KNeighborsClassifier(n_neighbors=13)
    # classifier.fit(x_train,y_train)
    # print(classifier.score(x_test,y_test))
    # Normalization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_x_train = scaler.fit_transform(x_train)
    scaled_x_test = scaler.transform(x_test)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    #
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(scaled_x_train,y_train)
    # print(KNeighborsClassifier.score(scaled_x_test,y_test))


    age = int(txtAge.get())
    sex = intSex.get()
    ALB = float(txtALB.get())
    ALP = float(txtALP.get())
    ALT = float(txtALT.get())
    AST = float(txtAST.get())
    BIL = float(txtBIL.get())
    CHE = float(txtCHE.get())
    CHOL = float(txtCHOL.get())
    CREA = float(txtCREA.get())
    GGT = float(txtGGT.get())
    PROT = float(txtPROT.get())

    prediction = knn_model.predict([[age,sex,ALB,ALP,ALT,AST,BIL,CHE,CHOL,CREA,GGT,PROT]])
    # print(prediction)
    prediction_prob = knn_model.predict_proba([[age,sex,ALB,ALP,ALT,AST,BIL,CHE,CHOL,CREA,GGT,PROT]])
    prediction_prob_rounded = [round(prob * 100, 2) for prob in prediction_prob[0]]

    # print(prediction_prob)
    # return prediction,prediction_prob
    if prediction == 0:
        combined_output = 'Viable blood donors: ' + str(prediction) + "       " + 'Percentage: ' + str(prediction_prob_rounded)
        # msg.showinfo("HCV Result", combined_output)
    elif prediction == 1:
        combined_output = 'Suspect Blood Donor: ' + str(prediction) + "       " + 'Percentage: ' + str(prediction_prob_rounded)
        # msg.showinfo("HCV Result", combined_output)
    elif prediction == 2:
        combined_output = 'Hepatitis: ' + str(prediction) + "       " + 'Percentage: ' + str(prediction_prob_rounded)
        # msg.showinfo("HCV Result", combined_output)
    elif prediction == 3:
        combined_output = 'Fibrosis: ' + str(prediction) + "       " + 'Percentage: ' + str(prediction_prob_rounded)
        # msg.showinfo("HCV Result", combined_output)
    else:
        combined_output = 'Cirrhosis: ' + str(prediction) + "       " + 'Percentage: ' + str(prediction_prob_rounded)
        # msg.showinfo("HCV Result", combined_output)
    healthCarePredictResult = tkinter.LabelFrame(hepatitPredictFrame, text='KKN Prediction Result')
    healthCarePredictResult.grid(row=21, column=0,sticky='news',padx=5, pady=5)

    lblResult = Label(healthCarePredictResult)
    lblResult.config(text=combined_output,background='gray',highlightcolor='blue')
    lblResult.grid(row=22, column=0, padx=5, pady=10)

healthCarePredictCheckList = tkinter.LabelFrame(hepatitPredictFrame, text='Select Your Desired Algorithm!')
healthCarePredictCheckList.grid(row=16, column=0, sticky='news', padx=20, pady=10)

var_NB = BooleanVar()
checkNB = Checkbutton(healthCarePredictCheckList, text='Naive Bayes', variable=var_NB)
checkNB.grid(row=17, column=0, padx=10, pady=2)

var_KNN = BooleanVar()
checkKNN = Checkbutton(healthCarePredictCheckList, text='KNN', variable=var_KNN)
checkKNN.grid(row=17, column=1,padx=10, pady=2)
def checklistFunction():
    if var_NB.get():
       predFunction_NB()
    if var_KNN.get():
       PredFunction_KNN()

btnPredict = ttk.Button(hepatitPredictFrame, text='Predict', width=10, command=checklistFunction, style="Custom.TButton")
btnPredict.grid(row=18, column=0, padx=10, pady=2,sticky='news')

btnPredict = ttk.Button(hepatitPredictFrame, text='Reset Form', width=10, command=resetFormFunction, style="Custom.TButton")
btnPredict.grid(row=19, column=0, padx=10, pady=2,sticky='news')

# btnPredict = ttk.Button(hepatitPredictFrame, text='KNN predict', width=10, command=PredFunction_KNN, style="Custom.TButton")
# btnPredict.grid(row=18, column=0, padx=10, pady=2,sticky='news')

hepatitDataEntryForm.mainloop()

# for i in range(1, 30):
#     Classifier = KNeighborsClassifier(n_neighbors = i)
#     Classifier.fit(x_train, y_train)
#     Result = Classifier.score(x_test, y_test)
#     print(f"n_neighbors : {i} , Score : {Result}")
# -------------------------------------------------------------
# #Determining the best value for k
# test_error_rates = []
# for k in range(1, 30):
#     knn_model = KNeighborsClassifier(n_neighbors=k)
#     knn_model.fit(scaled_x_train, y_train)
#
#     y_pred_test = knn_model.predict(scaled_x_test)
#
#     test_error = 1 - accuracy_score(y_test, y_pred_test)
#     test_error_rates.append(test_error)
#
# plt.figure(figsize=(10,6),dpi=200)
# plt.plot(range(1,30),test_error_rates,label='Test Error')
# plt.legend()
# plt.ylabel('Error Rate')
# plt.xlabel("K Value")
# plt.show()
