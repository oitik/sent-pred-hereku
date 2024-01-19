from flask import Flask, request, render_template
import pickle
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.simplefilter("ignore", InconsistentVersionWarning)
with open('embeded_dict.pkl', 'rb') as f:
    emb_dict = pickle.load(f)


def bangla_token(text):
    vec = []
    input_token = text.split()
    for word in input_token:
        try:
            vec.append(emb_dict[word])
        except:
            pass
    return np.array(np.sum(vec, axis=0))


app = Flask(__name__)
with open('sen_classifier_gloveLR.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def hello_world():
    return render_template("sent_pred.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    review_text = request.form.get('bangla_sentence', '')  # Get the entered text or an empty string
    test_emb = [bangla_token(review_text)]
    result = model.predict(test_emb)

    return render_template('sent_pred.html', pred=result_message(result), result_class=result, entered_text=review_text)


def result_message(result):
    if result == 0:
        return 'Simple Sentence.(সরল বাক্য)\n'
    elif result == 1:
        return 'Complex Sentence.(জটিল বাক্য)\n'
    elif result == 2:
        return 'Compound sentence.(যৌগিক বাক্য)\n'
    else:
        return 'Error 071.\n'


if __name__ == '__main__':
    app.run(debug=True)
