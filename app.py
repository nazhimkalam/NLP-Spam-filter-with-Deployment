from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import numpy as np
from flask_cors import CORS
from flask_restful import Api, Resource, reqparse

# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)
API = Api(app)

class Predict(Resource):

    @staticmethod
    def post():
            parser = reqparse.RequestParser()
            parser.add_argument('message')
            args = parser.parse_args()  # creates dictionary
            data = args['message']
            vector = cv.transform([data]).toarray()
            prediction = clf.predict(vector)
            result = ["Not Spam", "Spam"]
            print(prediction)
            out = {'Prediction': result[prediction[0]]}
            return out, 200

API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)