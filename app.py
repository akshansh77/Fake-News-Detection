# # from flask import Flask, request, jsonify, render_template
# # import pickle
# # import numpy as np
# # from transformers import AutoModel, BertTokenizerFast
# # import torch
# # import requests
# # import nltk
# # from nltk.corpus import stopwords
# # import re
# # from nltk.stem.porter import PorterStemmer
# # from transformers import BertTokenizer
# # from keras.utils import pad_sequences
# # import torch
# # from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# # from flask_cors import CORS, cross_origin

# # app = Flask(__name__,template_folder='templates')
# # CORS(app)
# # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# # model = pickle.load(open('english_model.pkl', 'rb'))


# # @app.route('/predict', methods=['POST', 'GET'])
# # def predict(text):
# #     ps = PorterStemmer()
# #     model = pickle.load(open('english_model.pkl', 'rb'))
# #     tfidfvect = pickle.load(open('english_tfidfvect2.pkl', 'rb'))
# #     review = re.sub('[^a-zA-Z]', ' ', text)
# #     review = review.lower()
# #     review = review.split()
# #     review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
# #     review = ' '.join(review)
# #     review_vect = tfidfvect.transform([review]).toarray()
# #     prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    
# #     return prediction


# # if __name__ == '__main__':
# #     app.run()

# from flask import Flask, request, jsonify
# import pickle
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# app = Flask(__name__)

# # Load the pre-trained model and TfidfVectorizer
# model = pickle.load(open('english_model.pkl', 'rb'))
# tfidf = pickle.load(open('english_tfidfvect2.pkl', 'rb'))

# # Initialize the Porter stemmer from NLTK
# ps = PorterStemmer()

# # Set up a route to handle incoming requests to the '/predict' endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the input text from the request body
#     text = request.json['text']
    
#     # Preprocess the text using regex, stopword removal, and stemming
#     review = re.sub('[^a-zA-Z]', ' ', text)
#     review = review.lower()
#     review = review.split()
#     review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
#     review = ' '.join(review)
    
#     # Vectorize the preprocessed text using the TfidfVectorizer
#     review_vect = tfidf.transform([review]).toarray()
    
#     # Make the prediction using the pre-trained model
#     prediction = model.predict(review_vect)
    
#     # Return the prediction as a JSON response
#     return jsonify({'prediction': 'FAKE' if prediction == 0 else 'REAL'})
    
# if __name__ == '__main__':
#     app.run()
# # from flask import Flask, request, jsonify, render_template
# # import pickle
# # import numpy as np
# # from transformers import AutoModel, BertTokenizerFast
# # import torch
# # import requests
# # import nltk
# # from nltk.corpus import stopwords
# # import re
# # from nltk.stem.porter import PorterStemmer
# # from transformers import BertTokenizer
# # from keras.utils import pad_sequences
# # import torch
# # from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# # from flask_cors import CORS, cross_origin

# # app = Flask(__name__, template_folder='templates')
# # CORS(app)
# # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# # model = pickle.load(open('english_model.pkl', 'rb'))


# # def preprocess_text(text):
# #     ps = PorterStemmer()
# #     review = re.sub('[^a-zA-Z]', ' ', text)
# #     review = review.lower()
# #     review = review.split()
# #     review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
# #     review = ' '.join(review)
# #     return review


# # @app.route('/predict', methods=['POST', 'GET'])
# # def predict():
# #     if request.method == 'POST':
# #         text = request.form['text']
# #         review = preprocess_text(text)
# #         prediction = 'FAKE' if model.predict([review]) == 0 else 'REAL'
# #         return jsonify({'prediction': prediction})
# #     else:
# #         return render_template('index.html')


# # if __name__ == '__main__':
# #     app.run()
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle
import nltk
import traceback

app = Flask(__name__)
CORS(app)

nltk.data.path.append('./nltk_data')  # Add this line to specify the NLTK data path

def predict(text):
    ps = PorterStemmer()
    model = pickle.load(open('english_model.pkl', 'rb'))
    tfidfvect = pickle.load(open('english_tfidfvect2.pkl', 'rb'))
    
    # Check if NLTK stopwords resource is available
    if not nltk.corpus.stopwords.words('english'):
        nltk.download('stopwords')
    
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    try:
        prediction = model.predict(review_vect)[0]
    except Exception as e:
        traceback.print_exc()
        raise Exception('An error occurred during prediction: {}'.format(str(e)))
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    try:
        text = request.form['text']
        prediction = predict(text)
        prediction = int(prediction)  # Convert to Python integer
        return jsonify({'prediction': prediction})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during prediction: {}'.format(str(e))})

if __name__ == '__main__':
    app.run()
