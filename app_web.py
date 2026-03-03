from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("student_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        input_data = pd.DataFrame({"Hours": [hours]})
        prediction = round(model.predict(input_data)[0], 2)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)