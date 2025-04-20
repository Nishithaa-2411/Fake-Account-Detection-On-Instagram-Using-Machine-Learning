#Importing necessary libraries 

from flask import *

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/view')
def view():
    global df, dataset
    df = pd.read_csv('data.csv')
    dataset = df.head(100)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())


@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        # Load and preprocess the dataset
        data = pd.read_csv('data.csv')
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Split the dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, stratify=y, random_state=42
        )

        # Retrieve the selected algorithm from the form
        s = int(request.form['algo'])

        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')

        elif s == 1:
            # Decision Tree Classifier
            from sklearn.tree import DecisionTreeClassifier

            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(x_train, y_train)
            y_pred = dt.predict(x_test)

            acc = accuracy_score(y_test, y_pred) * 100
            pre = precision_score(y_test, y_pred) * 100
            re = recall_score(y_test, y_pred) * 100
            f1 = f1_score(y_test, y_pred) * 100

            msg = f'The accuracy obtained by Decision Tree Classifier is {acc:.2f}%'
            msg1 = f'The precision obtained by Decision Tree Classifier is {pre:.2f}%'
            msg2 = f'The recall obtained by Decision Tree Classifier is {re:.2f}%'
            msg3 = f'The f1 score obtained by Decision Tree Classifier is {f1:.2f}%'

            return render_template('model.html', msg=msg, msg1=msg1, msg2=msg2, msg3=msg3)

        elif s == 2:
            # Random Forest Classifier
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(random_state=42)
            rf.fit(x_train, y_train)
            y_pred = rf.predict(x_test)

            acc = accuracy_score(y_test, y_pred) * 100
            pre = precision_score(y_test, y_pred) * 100
            re = recall_score(y_test, y_pred) * 100
            f1 = f1_score(y_test, y_pred) * 100

            msg = f'The accuracy obtained by Random Forest Classifier is {acc:.2f}%'
            msg1 = f'The precision obtained by Random Forest Classifier is {pre:.2f}%'
            msg2 = f'The recall obtained by Random Forest Classifier is {re:.2f}%'
            msg3 = f'The f1 score obtained by Random Forest Classifier is {f1:.2f}%'

            return render_template('model.html', msg=msg, msg1=msg1, msg2=msg2, msg3=msg3)

        elif s == 3:
            # Logistic Regression
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(x_train, y_train)
            y_pred = lr.predict(x_test)

            acc = accuracy_score(y_test, y_pred) * 100
            pre = precision_score(y_test, y_pred) * 100
            re = recall_score(y_test, y_pred) * 100
            f1 = f1_score(y_test, y_pred) * 100

            msg = f'The accuracy obtained by Logistic Regression is {acc:.2f}%'
            msg1 = f'The precision obtained by Logistic Regression is {pre:.2f}%'
            msg2 = f'The recall obtained by Logistic Regression is {re:.2f}%'
            msg3 = f'The f1 score obtained by Logistic Regression is {f1:.2f}%'

            return render_template('model.html', msg=msg, msg1=msg1, msg2=msg2, msg3=msg3)

        elif s == 4:
            # XGBoost Classifier
            from xgboost import XGBClassifier

            xgb = XGBClassifier(
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
            xgb.fit(x_train, y_train)
            y_pred = xgb.predict(x_test)

            acc = accuracy_score(y_test, y_pred) * 100
            pre = precision_score(y_test, y_pred) * 100
            re = recall_score(y_test, y_pred) * 100
            f1 = f1_score(y_test, y_pred) * 100

            msg = f'The accuracy obtained by XGBoost Classifier is {acc:.2f}%'
            msg1 = f'The precision obtained by XGBoost Classifier is {pre:.2f}%'
            msg2 = f'The recall obtained by XGBoost Classifier is {re:.2f}%'
            msg3 = f'The f1 score obtained by XGBoost Classifier is {f1:.2f}%'

            return render_template('model.html', msg=msg, msg1=msg1, msg2=msg2, msg3=msg3)

        elif s == 5:
            # Support Vector Machine (SVM) Classifier
            from sklearn.svm import SVC

            svm = SVC(kernel='linear', probability=True, random_state=42)
            svm.fit(x_train, y_train)
            y_pred = svm.predict(x_test)

            acc = accuracy_score(y_test, y_pred) * 100
            pre = precision_score(y_test, y_pred) * 100
            re = recall_score(y_test, y_pred) * 100
            f1 = f1_score(y_test, y_pred) * 100

            msg = f'The accuracy obtained by SVM Classifier is {acc:.2f}%'
            msg1 = f'The precision obtained by SVM Classifier is {pre:.2f}%'
            msg2 = f'The recall obtained by SVM Classifier is {re:.2f}%'
            msg3 = f'The f1 score obtained by SVM Classifier is {f1:.2f}%'

            return render_template('model.html', msg=msg, msg1=msg1, msg2=msg2, msg3=msg3)

        elif s == 6:
            # Naive Bayes Classifier
            from sklearn.naive_bayes import GaussianNB

            nb = GaussianNB()
            nb.fit(x_train, y_train)
            y_pred = nb.predict(x_test)

            acc = accuracy_score(y_test, y_pred) * 100
            pre = precision_score(y_test, y_pred) * 100
            re = recall_score(y_test, y_pred) * 100
            f1 = f1_score(y_test, y_pred) * 100

            msg = f'The accuracy obtained by Naive Bayes Classifier is {acc:.2f}%'
            msg1 = f'The precision obtained by Naive Bayes Classifier is {pre:.2f}%'
            msg2 = f'The recall obtained by Naive Bayes Classifier is {re:.2f}%'
            msg3 = f'The f1 score obtained by Naive Bayes Classifier is {f1:.2f}%'

            return render_template('model.html', msg=msg, msg1=msg1, msg2=msg2, msg3=msg3)

        else:
            return render_template('model.html', msg='Invalid Algorithm Selection')

    return render_template('model.html')


import pickle
@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = float(request.form['text'])
        f2 = float(request.form['f2'])
        f3 = float(request.form['f3'])
        f4 = float(request.form['f4'])
        f5 = float(request.form['f5'])
        f6 = float(request.form['f6'])
        f7 = float(request.form['f7'])
        f8 = float(request.form['f8'])
        f9 = float(request.form['f9'])
        f10 = float(request.form['f10'])
        f11 = float(request.form['f11'])

        print(f1)

        li = [[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11]]
        print(li)
        
        filename='Random_forest.sav'
        model = pickle.load(open(filename, 'rb'))

        result =model.predict(li)
        result=result[0]
        print(result)
        if result==0:
            msg = 'The account is Genuine'
        elif result==1:
            msg= 'This is a Fake Account'
               
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')


if __name__ =='__main__':
    app.run(debug=True)