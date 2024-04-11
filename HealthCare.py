import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import messagebox as msg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB



HealthDataEntryForm = Tk()
HealthDataEntryForm.title('Health Care Prediction Form')
HealthDataEntryForm.geometry('700x450')
right = int(HealthDataEntryForm.winfo_screenwidth()/2-700/2)
down = int(HealthDataEntryForm.winfo_screenheight()/2-450/2)
HealthDataEntryForm.geometry('+{}+{}'.format(right,down))
HealthDataEntryForm.iconbitmap('images/health.ico')

healthCarePredictFrame = tkinter.Frame(HealthDataEntryForm)
healthCarePredictFrame.pack()

healthCarePredictDesc = tkinter.LabelFrame(healthCarePredictFrame, text='Welcome to Healthcare Predition Page!')
healthCarePredictDesc.grid(row=0, column=0, sticky='news', padx=20, pady=5)

lblTitle = Label(healthCarePredictDesc, text='Please enter all the information correctly and press the predict button to see the result!')
lblTitle.grid(row=1, column=0, padx=10, pady=10)

healthCarePredictInfo = tkinter.LabelFrame(healthCarePredictFrame, text='Patient Information')
healthCarePredictInfo.grid(row=2, column=0, sticky='news', padx=20, pady=10)

lblGender = Label(healthCarePredictInfo, text='Gender:')
lblGender.grid(row=3, column=0,sticky='w')
intGender = IntVar()
entGenderFemale = ttk.Radiobutton(healthCarePredictInfo, width=10, text='Female', variable=intGender, value=1)
entGenderFemale.grid(row=3, column=1)
entGenderMale = ttk.Radiobutton(healthCarePredictInfo, width=10, text='Male', variable=intGender, value=2)
entGenderMale.grid(row=3, column=2)
entGenderUnknown = ttk.Radiobutton(healthCarePredictInfo, width=10, text='Unknown', variable=intGender, value=3)
entGenderUnknown.grid(row=3, column=3)

lblAge = Label(healthCarePredictInfo, text='Age:')
lblAge.grid(row=4, column=0, padx=10, pady=10,sticky='w')
txtAge = StringVar()
entAge = ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtAge)
entAge.grid(row=4, column=1, padx=10, pady=10)

lblHypertension = Label(healthCarePredictInfo, text='Hypertension:')
lblHypertension.grid(row=5, column=0,sticky='w')
intHypertension = IntVar()
entHypertensionPositive = ttk.Radiobutton(healthCarePredictInfo, width=10, text='Yes', variable=intHypertension, value=1)
entHypertensionPositive.grid(row=5, column=1)
entHypertensionNegative = ttk.Radiobutton(healthCarePredictInfo, width=10, text='No', variable=intHypertension, value=0)
entHypertensionNegative.grid(row=5, column=2)

lblHeartDisease = Label(healthCarePredictInfo, text='Heart Disease:')
lblHeartDisease.grid(row=6, column=0,sticky='w')
intHeartDisease = IntVar()
entHeartDiseasePositive = ttk.Radiobutton(healthCarePredictInfo, width=10, text='Yes', variable=intHeartDisease, value=1)
entHeartDiseasePositive.grid(row=6, column=1)
entHeartDiseaseNegative = ttk.Radiobutton(healthCarePredictInfo, width=10, text='No', variable=intHeartDisease, value=0)
entHeartDiseaseNegative.grid(row=6, column=2)

lblEverMarried = Label(healthCarePredictInfo, text='Ever Married:')
lblEverMarried.grid(row=7, column=0,sticky='w')
intEverMarried = IntVar()
entEverMarriedPositive = ttk.Radiobutton(healthCarePredictInfo, width=10, text='Yes', variable=intEverMarried, value=1)
entEverMarriedPositive.grid(row=7, column=1)
entEverMarriedNegative = ttk.Radiobutton(healthCarePredictInfo, width=10, text='No', variable=intEverMarried, value=0)
entEverMarriedNegative.grid(row=7, column=2)

lblWorkType = Label(healthCarePredictInfo, text='Work Type:')
lblWorkType.grid(row=8, column=0,sticky='w')
intWorkType = IntVar()
entWorkTypePrivate = ttk.Radiobutton(healthCarePredictInfo, width=10, text='Private', variable=intWorkType, value=1)
entWorkTypePrivate.grid(row=8, column=1)
entWorkTypeSelfEmployed = ttk.Radiobutton(healthCarePredictInfo, width=15, text='Self-employed', variable=intWorkType,value=2)
entWorkTypeSelfEmployed.grid(row=8, column=2)
entWorkTypeChildren = ttk.Radiobutton(healthCarePredictInfo, width=10, text='Children', variable=intWorkType, value=3)
entWorkTypeChildren.grid(row=8, column=3)
entWorkTypeGovtJob = ttk.Radiobutton(healthCarePredictInfo, width=10, text='Govt Job', variable=intWorkType, value=4)
entWorkTypeGovtJob.grid(row=8, column=4)
entWorkTypeNeverWorked = ttk.Radiobutton(healthCarePredictInfo, width=13, text='Never Worked', variable=intWorkType, value=5)
entWorkTypeNeverWorked.grid(row=8, column=5)

lblResidenceType = Label(healthCarePredictInfo, text='Residence Type:')
lblResidenceType.grid(row=9, column=0,sticky='w')
intResidenceType = IntVar()
entResidenceTypeUrban = ttk.Radiobutton(healthCarePredictInfo, width=10, text='Urban', variable=intResidenceType, value=1)
entResidenceTypeUrban.grid(row=9, column=1)
entResidenceTypeRural = ttk.Radiobutton(healthCarePredictInfo, width=10, text='Rural', variable=intResidenceType, value=2)
entResidenceTypeRural.grid(row=9, column=2)

lblAvgGluose = Label(healthCarePredictInfo, text='AVG Glucose:')
lblAvgGluose.grid(row=10, column=0, padx=10, pady=10,sticky='w')
txtAvgGluose = StringVar()
entAvgGluose = ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtAvgGluose)
entAvgGluose.grid(row=10, column=1, padx=10, pady=10)

lblBMI = Label(healthCarePredictInfo, text='BMI:')
lblBMI.grid(row=11, column=0, padx=10, pady=10,sticky='w')
txtBMI = StringVar()
entBMI = ttk.Entry(healthCarePredictInfo, width=10, textvariable=txtBMI)
entBMI.grid(row=11, column=1, padx=10, pady=10)

lblSmokingStatus = Label(healthCarePredictInfo, text='Smoking Status:')
lblSmokingStatus.grid(row=12, column=0,sticky='w')
intSmokingStatus = IntVar()
entSmokingStatusNever = ttk.Radiobutton(healthCarePredictInfo, width=10, text='never smoked', variable=intSmokingStatus,value=1)
entSmokingStatusNever.grid(row=12, column=1)
entSmokingStatusUnknown = ttk.Radiobutton(healthCarePredictInfo, width=10, text='Unknown', variable=intSmokingStatus, value=2)
entSmokingStatusUnknown.grid(row=12, column=2)
entSmokingStatusFormerly = ttk.Radiobutton(healthCarePredictInfo, width=15, text='formerly smoked', variable=intSmokingStatus,value=3)
entSmokingStatusFormerly.grid(row=12, column=3)
entSmokingStatusSmokes = ttk.Radiobutton(healthCarePredictInfo, width=10, text='smokes', variable=intSmokingStatus, value=4)
entSmokingStatusSmokes.grid(row=12, column=4)

# Naive Bayes Algorithm
def predictionFunction():
      df = pd.read_csv('DataSets\healthcare-dataset-stroke-data.csv')
      df.fillna(df.mean(numeric_only=True), inplace=True)
      df = df.drop(['id'], axis=1)
      X = df.iloc[:, 0:10]
      y = df.iloc[:, 10]

      X['gender'] = X['gender'].replace('Female', 1)
      X['gender'] = X['gender'].replace('Male', 2)
      X['gender'] = X['gender'].replace('Other', 3)
      #
      X['ever_married'] = X['ever_married'].replace('Yes', 1)
      X['ever_married'] = X['ever_married'].replace('No', 0)

      X['work_type'] = X['work_type'].replace('Private', 1)
      X['work_type'] = X['work_type'].replace('Self-employed', 2)
      X['work_type'] = X['work_type'].replace('children', 3)
      X['work_type'] = X['work_type'].replace('Govt_job', 4)
      X['work_type'] = X['work_type'].replace('Never_worked', 5)
      #
      X['Residence_type'] = X['Residence_type'].replace('Urban', 1)
      X['Residence_type'] = X['Residence_type'].replace('Rural', 2)
      #
      X['smoking_status'] = X['smoking_status'].replace('never smoked', 1)
      X['smoking_status'] = X['smoking_status'].replace('Unknown', 2)
      X['smoking_status'] = X['smoking_status'].replace('formerly smoked', 3)
      X['smoking_status'] = X['smoking_status'].replace('smokes', 4)
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

      Classifier = GaussianNB()
      Classifier.fit(X_train.values, y_train.values)
      GaussianNB()

      gender = intGender.get()
      age = int(txtAge.get())
      hypertension = intHypertension.get()
      heartDisease = intHeartDisease.get()
      everMarried = intEverMarried.get()
      workType = intWorkType.get()
      residenceType = intResidenceType.get()
      avgGlucose = int(txtAvgGluose.get())
      bmi = float(txtBMI.get())
      smokingStatus = intSmokingStatus.get()


      diagnosisOutput = Classifier.predict([[gender, age, hypertension, heartDisease, everMarried, workType, residenceType,
                                 avgGlucose, bmi, smokingStatus]])
      diagnosticOutputPercentage = Classifier.predict_proba([[gender, age, hypertension, heartDisease, everMarried, workType, residenceType,
                                       avgGlucose, bmi, smokingStatus]])

      if diagnosisOutput==1:
            FinalOutput = 'Positive Result:'+ str(diagnosisOutput) + '\n' + 'Percentage:' + str(diagnosticOutputPercentage)
            msg.showinfo("Healthcare Result", FinalOutput)
      else:
            FinalOutput = 'Negative Result:'+ str(diagnosisOutput) + '\n' + 'Percentage:' + str(diagnosticOutputPercentage)
            msg.showinfo('Healthcare Result',FinalOutput)

for widget in healthCarePredictInfo.winfo_children():
    widget.grid_configure(padx=5, pady=2)

btnPredict = ttk.Button(healthCarePredictFrame, text='Predict', width=10, command=predictionFunction)
btnPredict.grid(row=13, column=0,padx=10, pady=20, sticky='news')


healthCarePredictInfo.mainloop()
