
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [
        int(request.form["airline"]),
        int(request.form["source"]),
        int(request.form["destination"]),
        int(request.form["stops"]),
        int(request.form["duration"]),
        int(request.form["day"]),
        int(request.form["month"]),
    ]
    prediction = model.predict([data])[0]
    return render_template("index.html", prediction_text=f"Estimated Airline Fare: ₹ {round(prediction,2)}")

if __name__ == "__main__":
    app.run(debug=True)
