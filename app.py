
from flask import Flask, request, render_template
from keras.models import load_model
global model

import pickle
with open('vectorizer.pickle','rb') as file:
    cv = pickle.load(file)
model = load_model('project.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods = ['post'])
def login():
    
    review = request.form['review']
    review = cv.transform([review])
    y_pred = model.predict(review)
    if(y_pred>0.5):
        y = 'A positive review'
    else:
        y = 'A negative review'
    return render_template('index.html',abc = y)


if __name__ == '__main__':
    app.run(debug = True)