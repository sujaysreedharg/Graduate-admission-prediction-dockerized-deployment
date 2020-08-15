from flask import Flask, render_template, request
import pickle


filename = 'linearregressionmodel.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        grescore = int(request.form['gre'])
        toefl = int(request.form['toefl'])
        urating = int(request.form['Universityrating'])
        sop = int(request.form['sop'])
        lor = int(request.form['lor'])
        gpa = float(request.form['gpa'])
        re = float(request.form['re'])   
        return render_template('results.html',prediction=('{}%'.format(round(model.predict([[grescore,toefl,urating,sop,lor,gpa,re]])[0]*100, 1))))
if __name__ == '__main__':
	app.run()