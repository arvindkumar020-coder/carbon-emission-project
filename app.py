from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io, base64
from flask import Flask, request, render_template_string

# -----------------------
# Paths & startup checks
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ml" / "model.pkl"
META_PATH = BASE_DIR / "ml" / "metadata.json"
DATA_CANDIDATES = [
    BASE_DIR / "data" / "vehicles_100_corrected.csv",
    BASE_DIR / "data" / "vehicles_100.csv",
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train model first.")

model = joblib.load(MODEL_PATH)

# Load metadata
if META_PATH.exists():
    with open(META_PATH, "r", encoding="utf8") as f:
        meta = json.load(f)
else:
    meta = {
        "categorical": ["Make", "Model", "Fuel", "Transmission"],
        "numeric": ["EngineSize", "Cylinders", "FuelConsumption"],
        "target": "CO2Emissions"
    }

CATEGORICAL = meta["categorical"]
NUMERIC = meta["numeric"]
TARGET = meta["target"]

# Load dataset for dropdowns & fleet avg
fleet_df = None
for p in DATA_CANDIDATES:
    if p.exists():
        try:
            fleet_df = pd.read_csv(p)
            break
        except Exception:
            pass

fleet_avg = float(fleet_df[TARGET].mean()) if fleet_df is not None else None
dropdown_values = {
    col: sorted(fleet_df[col].dropna().unique()) if col in fleet_df.columns else []
    for col in CATEGORICAL
}

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)

# -----------------------
# HTML Template
# -----------------------
# ... keep all Python logic same as your code above ...

HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Tata Motors Kavach</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background: url('{{url_for('static', filename='trees-bg.jpg')}}') no-repeat center center fixed;
            background-size: cover;
            margin: 0; padding: 0;
            color: #222;
        }
        .container {
            max-width: 1200px;
            margin: 20px auto;
            padding: 40px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border-radius: 16px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .logo {
            width: 220px;
            margin-bottom: 12px;
        }
        h1 {
            font-size: 42px;
            color: #14532d;
        }
        .content {
            display: flex;
            gap: 40px;
        }
        .form-section {
            flex: 1.2;
        }
        .graph-section {
            flex: 1;
            text-align: center;
        }
        form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }
        label {
            font-weight: 600;
            margin-bottom: 8px;
            display: block;
            font-size: 16px;
        }
        input, select, textarea {
            width: 100%;
            padding: 16px;
            border-radius: 10px;
            border: 1px solid #aaa;
            font-size: 17px;
        }
        button {
            grid-column: span 2;
            padding: 18px;
            background: #14532d;
            color: #fff;
            font-size: 20px;
            font-weight: bold;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover { background: #1e7d4d; }
        .result {
            margin-top: 25px;
            padding: 18px;
            background: #f0fdf4;
            border-left: 6px solid #16a34a;
            border-radius: 8px;
            font-size: 18px;
        }
        .suggestions {
            margin-top: 25px;
            padding: 20px;
            background: #fff8e1;
            border-left: 6px solid #f59e0b;
            border-radius: 8px;
        }
        .suggestions h3 { margin: 0 0 10px; color: #92400e; }
        .user-suggestions-box {
            margin-top: 20px;
            padding: 18px;
            background: #e0f7fa;
            border-left: 6px solid #0288d1;
            border-radius: 10px;
        }
        .user-suggestions-box h3 { margin-top: 0; color: #0288d1; }
        .extra-images {
            margin-top: 40px;
            text-align: center;
        }
        .extra-images img {
            width: 500px;
            margin: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img class="logo" src="data:image/jpeg;base64,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" alt="Tata Motors Logo">
            <h1>KAVACH</h1>
        </div>
        <div class="content">
            <div class="form-section">
                <form method="post">
                    {% for c in categorical %}
                        <div>
                            <label>{{c}}:</label>
                            <select name="{{c}}" required>
                                {% for val in dropdown_values[c] %}
                                    <option value="{{val}}">{{val}}</option>
                                {% endfor %}
                            </select>
                        </div>
                    {% endfor %}
                    {% for n in numeric %}
                        <div>
                            <label>{{n}}:</label>
                            <input type="number" step="any" name="{{n}}" required>
                        </div>
                    {% endfor %}
                    <div style="grid-column: span 2;">
                        <label>Share your Tip for a Greener Ride:</label>
                        <textarea name="user_suggestion" rows="3" maxlength="160" placeholder="Share your sustainability tip..."></textarea>
                    </div>
                    <button type="submit">Predict CO₂</button>
                </form>

                {% if prediction %}
                <div class="result">
                    <p><strong>Predicted CO₂:</strong> {{prediction}} g/km</p>
                    {% if fleet_avg %}
                    <p><strong>Fleet Average CO₂:</strong> {{fleet_avg}} g/km</p>
                    {% endif %}
                </div>

                <div class="suggestions">
                    <h3>Suggestions for a Greener Ride 🌱</h3>
                    <ul>
                        {% for s in suggestions %}
                        <li>{{s}}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}

                {% if user_suggestion %}
                <div class="user-suggestions-box">
                    <h3>Your Submitted Eco Tip</h3>
                    <ul><li>{{user_suggestion}}</li></ul>
                </div>
                {% endif %}
            </div>
            
            {% if graph %}
            <div class="graph-section">
                <h3>Graphical Analysis</h3>
                <img src="data:image/png;base64,{{graph}}" alt="Graph">
            </div>
            {% endif %}
        </div>

        <div class="extra-images">
            <h3>Eco Awareness</h3>
            <img src="{{url_for('static', filename='static/car.webp')}}" alt="Eco Car">
            <img src="{{url_for('static', filename='static/eco-car-logo-template-design_316488-465.jpg')}}" alt="CO₂ Awareness">
        </div>
    </div>
</body>
</html>
"""


# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction, graph, suggestions, user_suggestion = None, None, [], None
    if request.method == "POST":
        payload = {c: request.form.get(c) for c in CATEGORICAL + NUMERIC}
        user_suggestion = request.form.get("user_suggestion", "").strip()

        row = {c: float(payload[c]) if c in NUMERIC else payload[c] for c in CATEGORICAL + NUMERIC}
        X = pd.DataFrame([row], columns=CATEGORICAL + NUMERIC)

        try:
            pred = model.predict(X)
            prediction = round(float(pred[0]), 2)

            # Graph
            plt.figure(figsize=(5,4))
            plt.bar(["Your Car", "Fleet Avg"], [prediction, fleet_avg], color=["#0d3b66", "#66bb6a"])
            plt.ylabel("CO₂ g/km")
            plt.title("CO₂ Emission Comparison")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            graph = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            plt.close()

            # Suggestions
            if prediction > fleet_avg:
                suggestions = [
                    "Consider regular maintenance to improve fuel efficiency.",
                    "Carpool whenever possible to reduce per-person emissions.",
                    "Adopt smooth driving habits to reduce fuel use.",
                    "Explore hybrid or electric vehicles for the future."
                ]
            else:
                suggestions = [
                    "Great job! Your car is performing better than the fleet average.",
                    "Keep maintaining your vehicle regularly.",
                    "Try using biofuels or renewable energy options when possible."
                    "Also use safety measures to increase rider safety"
                ]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template_string(
        HTML_FORM,
        categorical=CATEGORICAL,
        numeric=NUMERIC,
        prediction=prediction,
        fleet_avg=fleet_avg,
        dropdown_values=dropdown_values,
        graph=graph,
        suggestions=suggestions,
        user_suggestion=user_suggestion
    )

# -----------------------
# Run Flask
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
