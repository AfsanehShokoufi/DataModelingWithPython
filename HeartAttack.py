import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import messagebox as msg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


dataEntryForm = Tk()
dataEntryForm.title('Heart Attack Prediction Form')
dataEntryForm.geometry('800x450')
right = int(dataEntryForm.winfo_screenwidth()/2-800/2)
down = int(dataEntryForm.winfo_screenheight()/2-450/2)
dataEntryForm.geometry('+{}+{}'.format(right,down))
dataEntryForm.iconbitmap('images/health.ico')

heartAttackPredictFrame = tkinter.Frame(dataEntryForm)
heartAttackPredictFrame.pack()

heartAttackPredictInfo = tkinter.LabelFrame(heartAttackPredictFrame, text='Patient Information')
heartAttackPredictInfo.grid(row=0, column=0, sticky='news', padx=20, pady=10)

lblAge = Label(heartAttackPredictInfo, text='Age:')
lblAge.grid(row=2, column=0, padx=10, pady=10)
txtAge = StringVar()
entAge = ttk.Entry(heartAttackPredictInfo, width=10, textvariable=txtAge)
entAge.grid(row=2, column=1, padx=10, pady=10)

lblSex = Label(heartAttackPredictInfo, text='Sex:')
lblSex.grid(row=3, column=0)
intSex = IntVar()
entSexFemale = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Female', variable=intSex, value=1)
entSexFemale.grid(row=3, column=1)
entSexMale = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Male', variable=intSex, value=0)
entSexMale.grid(row=3, column=2)

lblcp = Label(heartAttackPredictInfo, text='Chest pain type:')
lblcp.grid(row=4, column=0)
intcp = IntVar()
entcp = ttk.Radiobutton(heartAttackPredictInfo, width=15, text='Typical angina', variable=intcp, value=0)
entcp.grid(row=4, column=1)
entcp = ttk.Radiobutton(heartAttackPredictInfo, width=15, text='atypical angina', variable=intcp, value=1)
entcp.grid(row=4, column=2)
entcp = ttk.Radiobutton(heartAttackPredictInfo, width=15, text='Non-anginal pain', variable=intcp, value=2)
entcp.grid(row=4, column=3)
entcp = ttk.Radiobutton(heartAttackPredictInfo, width=15, text='Asymptomatic', variable=intcp, value=3)
entcp.grid(row=4, column=4)

lbltrestbps = Label(heartAttackPredictInfo, text='Resting blood pressure:')
lbltrestbps.grid(row=5, column=0, padx=10, pady=10)
txttrestbps = StringVar()
enttrestbps = ttk.Entry(heartAttackPredictInfo, width=10, textvariable=txttrestbps)
enttrestbps.grid(row=5, column=1, padx=10, pady=10)

lblchol= Label(heartAttackPredictInfo, text='Cholestoral:')
lblchol.grid(row=6, column=0, padx=10, pady=10)
txtchol = StringVar()
entchol= ttk.Entry(heartAttackPredictInfo, width=10, textvariable=txtchol)
entchol.grid(row=6, column=1, padx=10, pady=10)

lblfbs = Label(heartAttackPredictInfo, text='Fasting blood sugar:')
lblfbs.grid(row=7, column=0)
intfbs= IntVar()
entfbsTrue = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Yes', variable=intfbs, value=1)
entfbsTrue.grid(row=7, column=1)
entfbsFalse = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='No', variable=intfbs, value=0)
entfbsFalse.grid(row=7, column=2)

lblrestecg = Label(heartAttackPredictInfo, text='Resting electrocardiographic:')
lblrestecg.grid(row=8, column=0)
intrestecg= IntVar()
entrestecgZero = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Yes', variable=intrestecg, value=0)
entrestecgZero.grid(row=8, column=1)
entrestecgOne = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='No', variable=intrestecg, value=1)
entrestecgOne.grid(row=8, column=2)
entrestecgTwo = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='No', variable=intrestecg, value=2)
entrestecgTwo.grid(row=8, column=3)

lblthalach = Label(heartAttackPredictInfo, text='Max heart rate achieved:')
lblthalach.grid(row=9, column=0, padx=10, pady=10)
txtthalach = StringVar()
entthalach = ttk.Entry(heartAttackPredictInfo, width=10, textvariable=txtthalach)
entthalach.grid(row=9, column=1, padx=10, pady=10)

lblexang = Label(heartAttackPredictInfo, text='Exercise induced angina:')
lblexang.grid(row=10, column=0)
intexang= IntVar()
entexangTrue = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Pain', variable=intexang, value=1)
entexangTrue.grid(row=10,column=1)
entexangFalse = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='No pain', variable=intexang, value=0)
entexangFalse.grid(row=10, column=2)

lbloldpeak = Label(heartAttackPredictInfo, text='ST depression:')
lbloldpeak.grid(row=11, column=0, padx=10, pady=10)
txtoldpeak = StringVar()
entoldpeak = ttk.Entry(heartAttackPredictInfo, width=10, textvariable=txtoldpeak)
entoldpeak.grid(row=11, column=1, padx=10, pady=10)

lblslope = Label(heartAttackPredictInfo, text='Level of peak exercise:')
lblslope.grid(row=12, column=0)
intslope= IntVar()
entslopeUp = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Up', variable=intslope, value=0)
entslopeUp.grid(row=12,column=1)
entslopeFlat = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Flat', variable=intslope, value=1)
entslopeFlat.grid(row=12, column=2)
entslopeDown = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Down', variable=intslope, value=2)
entslopeDown.grid(row=12, column=2)

lblca = Label(heartAttackPredictInfo, text='number of major vessels:')
lblca.grid(row=13, column=0)
intca= IntVar()
entslopeZero = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Zero', variable=intca, value=0)
entslopeZero.grid(row=13,column=1)
entslopeOne = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='One', variable=intca, value=1)
entslopeOne.grid(row=13, column=2)
entslopeTwo = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Two', variable=intca, value=2)
entslopeTwo.grid(row=13, column=3)
entslopeThree = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Three', variable=intca, value=3)
entslopeThree.grid(row=13, column=4)
entslopeFour = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Four', variable=intca, value=4)
entslopeFour.grid(row=13, column=5)

lblthal = Label(heartAttackPredictInfo, text='Thalassemia :')
lblthal.grid(row=14, column=0)
intthal= IntVar()
entthalZero = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Normal', variable=intthal, value=0)
entthalZero.grid(row=14,column=1)
entthalOne = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Fixed', variable=intthal, value=1)
entthalOne.grid(row=14, column=2)
entthalTwo = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Reversible', variable=intthal, value=2)
entthalTwo.grid(row=14, column=3)
entthalThree = ttk.Radiobutton(heartAttackPredictInfo, width=10, text='Non-reversible', variable=intthal, value=3)
entthalThree.grid(row=14, column=4)

# Naive Bayes Algorithm
def predictFunction():
    df = pd.read_csv('DataSets\Heart Attack Data Set.csv')
    X = df.iloc[:, 0:13]
    y = df.iloc[:, 13]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    Classifier = GaussianNB()
    Classifier.fit(X_train.values, y_train.values)


    age = int(txtAge.get())
    sex = intSex.get()
    cp = intcp.get()
    trestbps = int(txttrestbps.get())
    chol = int(txtchol.get())
    fbs = intfbs.get()
    restecg = intrestecg.get()
    thalach = int(txtthalach.get())
    exang = intexang.get()
    oldpeak = float(txtoldpeak.get())
    slope = intslope.get()
    ca = intca.get()
    thal = intthal.get()

    output1 = Classifier.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    output2 = Classifier.predict_proba([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]] )


    if output1==1:
        combined_output = 'Positive Result:'+ str(output1) + "\n" + 'Percentage:'+ str(output2)
        msg.showinfo("Heart Attack Result", combined_output)
    else:
        combined_output = 'Negative Result:'+ str(output1) + "\n" + 'Percentage:'+ str(output2)
        msg.showinfo("Heart Attack Result", combined_output)


for widget in heartAttackPredictInfo.winfo_children():
    widget.grid_configure(padx=5, pady=2)

style = ttk.Style()
style.configure("Custom.TButton", background="gray")
btnPredict = ttk.Button(heartAttackPredictFrame, text='Predict', width=10, command=predictFunction, style="Custom.TButton")
btnPredict.grid(row=15, column=0, padx=10, pady=20,sticky='news')

dataEntryForm.mainloop()