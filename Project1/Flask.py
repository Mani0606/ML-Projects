from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("calories_predict.pickle", "rb"))


@app.route("/")
def hello_world():
    return render_template("home.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    features = [(x) for x in request.form.values()]
    if features[0] == "male" or features[0] == "Male":
        features[0] = 0
    elif features[0] == "female" or features[0] == "Female":
        features[0] = 1
    final = [np.array(features)]
    output = model.predict(final)
    return render_template("home.html", pred=output)


if __name__ == "__main__":
    app.run(debug=True)
