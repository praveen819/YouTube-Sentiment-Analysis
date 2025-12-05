from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
from prediction import youtubeCommentsAnalysis


app = Flask(__name__)
app.secret_key = "comments"


@app.route("/")
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form["email"]
        psd = request.form["password"]
        r1 = pd.read_excel("user.xlsx")
        for index, row in r1.iterrows():
            if row["email"] == str(email) and row["password"] == str(psd):
                return redirect(url_for('home'))

        msg = 'Invalid Login'
        return render_template('login.html', msg=msg)
    return render_template('login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        Name = request.form['name']
        Email = request.form['email']
        Password = request.form['password']
        col_list = ["name", "email", "password"]
        r1 = pd.read_excel('user.xlsx', usecols=col_list)
        new_row = pd.DataFrame(
            {'name': [Name], 'email': [Email], 'password': [Password]})
        r1 = r1.append(new_row, ignore_index=True)
        r1.to_excel('user.xlsx', index=False)
        print("Records created successfully")
        msg = 'Registration Successfull !!'
        # return redirect(url_for('login', msg=msg))
        return render_template('login.html', msg=msg)
    return render_template('register.html')


def read_csv(file):
    df = pd.read_csv(file)
    cols = list(df.columns)
    df1 = np.asarray(df)
    length = len(df1)
    df2 = []
    count = length
    for i in range(length):
        df2.append(df1[count - 1])
        count -= 1
    print("df2: ", df2)
    return cols, df2


@app.route('/home', methods=['POST', 'GET'])
def home():
    return render_template('index1.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        url_name = request.form['input_file']
        try:
            prediction = youtubeCommentsAnalysis(url_name)
            print("prediction : ", prediction)
            
            success_message = "Youtube Comments Detection is Successfully Completed"

            positive_title = "List of   Positive Comments"
            positive_commnets = read_csv("static/result/positive_df.csv")
            
            negative_title = "List of Negative Comments"
            negative_data = read_csv("static/result/negative_df.csv")
            
            return render_template("result1.html", success=success_message, total_ss=prediction[0], positive_ss=prediction[1], negative_ss=prediction[2], i_t=positive_title, i_cols=positive_commnets[0], i_values=positive_commnets[1], r_t=negative_title, r_cols=negative_data[0], r_values=negative_data[1])
        except Exception as e:
            print(e)
            error_message = f"An error occurred: {str(e)}"
            return render_template("index1.html", error=error_message)
    return render_template("result1.html")


@app.route('/changepsd', methods=['POST', 'GET'])
def changepsd():
    if request.method == 'POST':
        cur_pass = request.form['currentpsd']
        new_pass = request.form['newpsd']
        verify_pass = request.form['reenterpsd']
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["password"] == str(cur_pass):
                if new_pass == verify_pass:
                    r1.replace(to_replace=cur_pass,
                               value=verify_pass, inplace=True)
                    r1.to_excel("user.xlsx", index=False)
                    msg1 = 'Password changed successfully'
                    return render_template('changepsd.html', msg1=msg1)
                else:
                    msg2 = 'Re-entered password is not matched'
                    return render_template('changepsd.html', msg2=msg2)
        else:
            msg3 = 'Incorrect password'
            return render_template('changepsd.html', msg3=msg3)
    return render_template('changepsd.html')

@app.route('/graph', methods=['POST', 'GET'])
def graph():
    try:
        if request.method == 'POST':
            graph_name = request.form['text']
            graph = ''
            name = ''

            if graph_name == "c_ac":
                model_name = "Convolutional Gated Recurrent Unit Model"
                name = "Accuracy Plot Graph "
                graph = "static/graphs/c_ac.png"
            elif graph_name == 'c_ls':
                model_name = "Convolutional Gated Recurrent Unit Model"
                name = "Loss Plor Graph"
                graph = "static/graphs/c_ls.png"
            elif graph_name == 'c_cr':
                model_name = "Convolutional Gated Recurrent Unit Model"
                name = "Classification Report"
                graph = "static/graphs/c_cr.png"
            elif graph_name == 'c_cm':
                model_name = "Convolutional Gated Recurrent Unit Model"
                name = "Confusion Matrix"
                graph = "static/graphs/c_cm.png"
            

            return render_template('graphs.html', mn=model_name, name=name, graph=graph)
    except Exception as e:
         msg = "Select the Graph."
         return render_template('graphs.html', msg=msg)
    
@app.route('/graphs', methods=['POST', 'GET'])
def graphs():
    return render_template('graphs.html')


@app.route('/logout')
def logout():
    msg = 'You are now logged out', 'success'
    return redirect(url_for('login', msg=msg))


if __name__ == '__main__':
    app.run(port=5026, debug=True)
