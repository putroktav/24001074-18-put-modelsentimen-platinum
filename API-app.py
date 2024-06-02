from flasgger import swag_from
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flask import request
import re
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf

from flask import Flask, jsonify

# flask
app = Flask(__name__)

# swagerr
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info={
        'title': LazyString(lambda: 'API Documentation for Data Processing and Modeling'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda: 'Dokumentasi API untuk Data Processing dan Modeling'),
    },
    host=LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app)
# klasifikasi


def cleansing(sent):
    # Mengubah kata menjadi huruf kecil semua dengan menggunakan fungsi lower()
    string = sent.lower()
    # Menghapus emoticon dan tanda baca menggunakan "RegEx" dengan script di bawah
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    return string


sentiment = ['negative', 'neutral', 'positive']

file_lstm = open("./resources/lstm-resource/tokenizer.pickle", 'rb')
tokenizer_lstm = pickle.load(file_lstm)

model_lstm = tf.keras.models.load_model('./model/lstm-model/model-lstm-v2.h5')

file_nn = open("./resources/nn-resource/vectorizer.pkl", 'rb')
tokenizer_nn = pickle.load(file_nn)

file_nn_model = open("./model/nn-model/mlp_model.pkl", 'rb')
model_nn = pickle.load(file_nn_model)


# body api


@swag_from("./docs/lstm_text_processing.yml", methods=['POST'])
@app.route('/lstm-text-processing', methods=['POST'])
def lstm_text_processing():

    input_text = request.form.get('text')

    text = [cleansing(input_text)]
    predicted = tokenizer_lstm.texts_to_sequences(text)
    print("predicted", predicted)
    guess = tf.keras.preprocessing.sequence.pad_sequences(predicted, maxlen=82)
    print("guess", guess)

    prediction = model_lstm.predict(guess)
    print("prediction", prediction)
    polarity = np.argmax(prediction[0])
    print("polarity", polarity)

    print("Text: ", text[0])
    print("Sentiment: ", sentiment[polarity])

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'text': input_text,
        'sentiment': sentiment[polarity]
    }

    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/lstm_file_text_processing.yml", methods=['POST'])
@app.route('/lstm-text-processing-file', methods=['POST'])
def lstm_text_processing_file():

    # Upladed file
    file = request.files.getlist('file')[0]

    # Import file csv ke Pandas
    df = pd.read_csv(file)
    print("data:", df)

    # Ambil teks yang akan diproses dalam format list
    texts = df.iloc[:, 0].tolist()
    print("texts:", texts)

    # Lakukan cleansing pada teks
    cleaned_text = []
    label = []
    for input_text in texts:
        cleaned_text.append(input_text)
        text = [cleansing(input_text)]
        predicted = tokenizer_lstm.texts_to_sequences(text)
        print("predicted", predicted)
        guess = tf.keras.preprocessing.sequence.pad_sequences(predicted, maxlen=82)
        print("guess", guess)

        prediction = model_lstm.predict(guess)
        print("prediction", prediction)
        polarity = np.argmax(prediction[0])
        print("polarity", polarity)
        label.append(sentiment[polarity])
        print("label", label)

        print("Text: ", text[0])
        print("Sentiment: ", sentiment[polarity])

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data_text': cleaned_text,
        'data_sentiment': label
    }

    response_data = jsonify(json_response)
    return response_data


@swag_from("./docs/nn_text_processing.yml", methods=['POST'])
@app.route('/nn-text-processing', methods=['POST'])
def nn_text_processing():

    input_text = request.form.get('text')
        
    print("input_text:", input_text)

    text = [cleansing(input_text)]
    print("text:", text)
    predicted = tokenizer_nn.transform(text)
    print("predicted:", predicted)

    result = model_nn.predict(predicted)[0]
    print("result:", result)

    print("Text: ", text[0])
    print("Sentiment: ")
    if result == 0:
        print("Negative")
        sentiment = "Negative"
    elif result == 1:
        print("Positive")
        sentiment = "Positive"
    elif result == 2:
        print("Neutral")
        sentiment = "Neutral"

    print(sentiment)

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'text': input_text,
        'result': str(result),
        'sentiment': sentiment
    }

    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/nn_file_text_processing.yml", methods=['POST'])
@app.route('/nn-text-processing-file', methods=['POST'])
def nn_text_processing_file():

    # Upladed file
    file = request.files.getlist('file')[0]
    print("file:", file)

    # Import file csv ke Pandas
    df = pd.read_csv(file)
    print("data:", df)

    # Ambil teks yang akan diproses dalam format list
    texts = df.iloc[:, 0].tolist()
    print("texts:", texts)

    # Lakukan cleansing pada teks
    cleaned_text = []
    label = []
    for input_text in texts:
        cleaned_text.append(input_text)
        text = [cleansing(input_text)]
        print("text:", text)
        predicted = tokenizer_nn.transform(text)
        print("predicted:", predicted)

        result = model_nn.predict(predicted)[0]
        print("result:", result)

        if result == 0:
            print("Negative")
            sentiment = "Negative"
        elif result == 1:
            print("Positive")
            sentiment = "Positive"
        elif result == 2:
            print("Neutral")
            sentiment = "Neutral"
        
        label.append(sentiment)
        
        print("sentiment:", sentiment)
        print("cleaned_text:", cleaned_text)

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'text': cleaned_text,
        'result': str(result),
        'sentiment': sentiment,
        'data_sentiment': label
    }

    response_data = jsonify(json_response)
    return response_data


if __name__ == '__main__':
    app.run()
