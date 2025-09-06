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

        <!-- Replaced Extra Images -->
        <div class="extra-images">
            <h3>Eco Awareness</h3>
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
