import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from flask import Flask, render_template_string, request
import joblib

data = pd.read_csv(r"C:\Users\Admin\Desktop\Student_Score_Project\student_data.csv")

X = data[["Hours"]]
y = data["Score"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
accuracy = r2_score(y, y_pred)

print("Model Accuracy (R2 Score):", round(accuracy, 3))

joblib.dump(model, "student_model.pkl")
print("Model Saved Successfully!")

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Student Score Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>
*{margin:0;padding:0;box-sizing:border-box;}

:root{
    --glass:rgba(255,255,255,0.08);
    --border:rgba(255,255,255,0.15);
    --gold:#d4af37;
    --dark:#0b0f14;
}

body{
    font-family:'Inter',sans-serif;
    height:100vh;
    display:flex;
    justify-content:center;
    align-items:center;
    background:
        radial-gradient(circle at 20% 20%, rgba(212,175,55,0.15), transparent 40%),
        radial-gradient(circle at 80% 80%, rgba(0,255,255,0.1), transparent 50%),
        linear-gradient(135deg,#0b0f14,#111827,#0b0f14);
    color:white;
    overflow:hidden;
}

/* animated futuristic glow */
body::before{
    content:'';
    position:absolute;
    width:500px;
    height:500px;
    background:radial-gradient(circle, rgba(212,175,55,0.15), transparent 70%);
    filter:blur(120px);
    animation:float 10s infinite alternate ease-in-out;
}
@keyframes float{
    from{transform:translate(-100px,-50px);}
    to{transform:translate(100px,80px);}
}

/* glass card */
.card{
    width:420px;
    padding:50px 40px;
    border-radius:25px;
    backdrop-filter:blur(40px);
    background:var(--glass);
    border:1px solid var(--border);
    box-shadow:0 30px 80px rgba(0,0,0,0.6);
    text-align:center;
    animation:fade 0.8s ease;
}
@keyframes fade{
    from{opacity:0;transform:translateY(40px);}
    to{opacity:1;transform:translateY(0);}
}

h1{
    font-weight:700;
    font-size:26px;
    margin-bottom:8px;
    letter-spacing:1px;
}

.subtitle{
    font-size:12px;
    opacity:0.6;
    margin-bottom:30px;
    letter-spacing:2px;
}

/* input */
input{
    width:100%;
    padding:15px;
    border-radius:15px;
    border:1px solid var(--border);
    background:rgba(255,255,255,0.05);
    color:white;
    outline:none;
    font-size:15px;
    margin-bottom:20px;
    transition:0.3s;
}
input:focus{
    border-color:var(--gold);
    box-shadow:0 0 20px rgba(212,175,55,0.4);
}

/* button */
button{
    width:100%;
    padding:15px;
    border-radius:15px;
    border:none;
    font-weight:600;
    letter-spacing:2px;
    background:linear-gradient(135deg,#111,#222);
    color:var(--gold);
    cursor:pointer;
    transition:0.3s;
}
button:hover{
    background:linear-gradient(135deg,var(--gold),#8a6d1d);
    color:black;
    transform:translateY(-3px);
    box-shadow:0 10px 30px rgba(212,175,55,0.5);
}

/* result */
.result{
    margin-top:30px;
    padding:25px;
    border-radius:20px;
    background:rgba(255,255,255,0.05);
    border:1px solid var(--border);
    animation:pop 0.6s ease;
}
@keyframes pop{
    from{opacity:0;transform:scale(0.9);}
    to{opacity:1;transform:scale(1);}
}

.score{
    font-size:65px;
    font-weight:700;
    background:linear-gradient(135deg,var(--gold),white);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.footer{
    margin-top:25px;
    font-size:11px;
    opacity:0.5;
    letter-spacing:2px;
}
</style>
</head>

<body>

<div class="card">
    <h1>AI Student Score Predictor</h1>
    <div class="subtitle">LEARN DEPTH • INTERN • ABHI</div>

    <form method="POST">
        <input type="number" name="hours" step="0.1" min="0" placeholder="Enter Study Hours" required>
        <button type="submit">PREDICT SCORE</button>
    </form>

    {% if prediction %}
    <div class="result">
        <div style="font-size:12px;opacity:0.6;">Your Predicted Score</div>
        <div class="score">{{ prediction }}</div>
        <div style="margin-top:10px;font-size:12px;opacity:0.7;">
            Keep learning. Keep evolving.
        </div>
    </div>
    {% endif %}

    <div class="footer">Engineered with Intelligence by Abhilash</div>
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        input_data = pd.DataFrame({"Hours": [hours]})
        prediction = round(model.predict(input_data)[0], 2)
    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)