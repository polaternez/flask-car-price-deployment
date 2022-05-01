from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    new_data = [float(x) for x in request.form.values()]
    features = [np.array(new_data)]
    prediction = model.predict(features)

    output = round(prediction[0], 2)

    return render_template("index.html", prediction_text="Car price should be $ {}".format(output))
        


if __name__ == "__main__":
    app.run(debug=True)


