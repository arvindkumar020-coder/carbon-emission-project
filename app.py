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
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Tata Motors Kavach</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(135deg, #d4fc79, #96e6a1, #00c9ff, #92fe9d);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            margin: 0; padding: 0;
            color: #222;
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .container {
            max-width: 1200px;
            margin: 20px auto;
            padding: 40px;
            background: rgba(255, 255, 255, 0.9);
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
            flex-wrap: wrap;
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
            width: 400px;
            margin: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <!-- Tata Motors Logo -->
            <img class="logo" src="data:image/jpeg;base64,{{tata_logo}}" alt="Tata Motors Logo">
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

        <!-- Replaced Extra Images -->
        <div class="extra-images">
            <h3>Eco Awareness</h3>
            <img src="{{url_for('static', filename='reduce-co2.jpg')}}" alt="Reduce CO₂">
            <img src="{{url_for('static', filename='car.webp')}}" alt="Eco Car">
            <img src="{{url_for('static', filename='eco-car-logo-template-design_316488-465.jpg')}}" alt="CO₂ Awareness">
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
                    "Try using biofuels or renewable energy options when possible.",
                    "Also use safety measures to increase rider safety"
                ]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    tata_logo = ""  # You already had this embedded in base64 earlier
    return render_template_string(
        HTML_FORM,
        categorical=CATEGORICAL,
        numeric=NUMERIC,
        prediction=prediction,
        fleet_avg=fleet_avg,
        dropdown_values=dropdown_values,
        graph=graph,
        suggestions=suggestions,
        user_suggestion=user_suggestion,
        tata_logo=tata_logo
    )

# -----------------------
# Run Flask
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
