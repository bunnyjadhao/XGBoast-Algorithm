import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model (file in the same folder as app.py)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = ["XGBoost.pkl", "Decision_Tree_Algorithm.pkl"]

model = None
for name in MODEL_CANDIDATES:
    path = os.path.join(BASE_DIR, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        break

if model is None:
    raise FileNotFoundError("Place XGBoost.pkl next to app.py before deploying.")

# ---------------------------------------------------------------------------
# Feature encoding.
# The model was trained on: age, sex, bmi, children, smoker, region
# (age/sex/children/smoker/region are ints, bmi is float).
# These maps assume the usual alphabetical label encoding. If your notebook
# used a different encoding, change the numbers here.
# ---------------------------------------------------------------------------
SEX_MAP = {"female": 0, "male": 1}
SMOKER_MAP = {"no": 0, "yes": 1}
REGION_MAP = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Medical Insurance Cost Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Poppins',sans-serif;min-height:100vh;color:#fff;
    background:linear-gradient(135deg,#0f2027,#203a43,#2c5364,#6a11cb,#2575fc);
    background-size:400% 400%;animation:flow 18s ease infinite;
    display:flex;align-items:center;justify-content:center;padding:24px;
  }
  @keyframes flow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
  .card{
    width:100%;max-width:820px;padding:38px 40px;border-radius:26px;
    background:rgba(255,255,255,.12);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
    border:1px solid rgba(255,255,255,.28);box-shadow:0 25px 60px rgba(0,0,0,.35);
  }
  .head{text-align:center;margin-bottom:28px}
  .icon{font-size:46px}
  h1{font-size:30px;font-weight:700;background:linear-gradient(90deg,#43e97b,#38f9d7,#a1c4fd);
     -webkit-background-clip:text;background-clip:text;color:transparent;margin-top:6px}
  .sub{font-weight:300;font-size:14px;opacity:.85;margin-top:6px}
  form{display:grid;grid-template-columns:1fr 1fr;gap:20px}
  .field{display:flex;flex-direction:column}
  label{font-size:13px;font-weight:500;margin-bottom:7px;letter-spacing:.4px;opacity:.95}
  input,select{
    padding:13px 15px;border-radius:12px;border:1px solid rgba(255,255,255,.3);
    background:rgba(255,255,255,.16);color:#fff;font:inherit;font-size:15px;outline:none;transition:.25s;
  }
  input::placeholder{color:rgba(255,255,255,.55)}
  select option{color:#203a43}
  input:focus,select:focus{border-color:#38f9d7;background:rgba(255,255,255,.24);box-shadow:0 0 0 3px rgba(56,249,215,.25)}
  button{
    grid-column:1/-1;margin-top:6px;padding:15px;border:none;border-radius:14px;cursor:pointer;
    font:inherit;font-size:17px;font-weight:600;color:#0f2027;
    background:linear-gradient(90deg,#43e97b,#38f9d7);transition:.25s;
    box-shadow:0 10px 25px rgba(67,233,123,.35);
  }
  button:hover{transform:translateY(-3px);box-shadow:0 14px 30px rgba(56,249,215,.5)}
  .result{
    margin-top:26px;text-align:center;padding:24px;border-radius:18px;
    background:linear-gradient(135deg,rgba(67,233,123,.25),rgba(37,117,252,.25));
    border:1px solid rgba(255,255,255,.3);animation:pop .5s ease;
  }
  .result.err{background:rgba(255,80,80,.25)}
  @keyframes pop{from{opacity:0;transform:scale(.94)}to{opacity:1;transform:scale(1)}}
  .result .lbl{font-size:13px;letter-spacing:1.5px;text-transform:uppercase;opacity:.85}
  .result .amt{font-size:42px;font-weight:700;margin-top:6px}
  .note{text-align:center;font-size:12px;opacity:.7;margin-top:22px}
  .hidden{display:none}
  @media(max-width:640px){form{grid-template-columns:1fr}.card{padding:28px 22px}h1{font-size:24px}}
</style>
</head>
<body>
<div class="card">
  <div class="head">
    <div class="icon">🩺</div>
    <h1>Medical Insurance Cost Predictor</h1>
    <p class="sub">Estimate yearly medical charges with an XGBoost regression model</p>
  </div>

  <form id="form">
    <div class="field"><label>Age</label>
      <input type="number" name="age" min="18" max="100" placeholder="e.g. 32" required></div>
    <div class="field"><label>Sex</label>
      <select name="sex" required>
        <option value="" disabled selected>Select sex</option>
        <option value="female">Female</option><option value="male">Male</option>
      </select></div>
    <div class="field"><label>BMI</label>
      <input type="number" name="bmi" step="0.01" min="10" max="70" placeholder="e.g. 27.5" required></div>
    <div class="field"><label>Number of Children</label>
      <input type="number" name="children" min="0" max="10" placeholder="e.g. 2" required></div>
    <div class="field"><label>Smoker</label>
      <select name="smoker" required>
        <option value="" disabled selected>Select option</option>
        <option value="no">No</option><option value="yes">Yes</option>
      </select></div>
    <div class="field"><label>Region</label>
      <select name="region" required>
        <option value="" disabled selected>Select region</option>
        <option value="northeast">Northeast</option><option value="northwest">Northwest</option>
        <option value="southeast">Southeast</option><option value="southwest">Southwest</option>
      </select></div>
    <button type="submit" id="btn">Predict Insurance Cost</button>
  </form>

  <div id="result" class="result hidden">
    <div class="lbl" id="lbl">Estimated Annual Charges</div>
    <div class="amt" id="amt"></div>
  </div>
  <p class="note">For educational purposes only. Not an actual insurance quotation.</p>
</div>

<script>
const form = document.getElementById('form');
const box = document.getElementById('result');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn');
  btn.textContent = 'Calculating...';
  const payload = Object.fromEntries(new FormData(form).entries());
  try {
    const res = await fetch('/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
    });
    const data = await res.json();
    box.classList.remove('hidden', 'err');
    if (res.ok) {
      document.getElementById('lbl').textContent = 'Estimated Annual Charges';
      document.getElementById('amt').textContent = data.prediction;
    } else {
      box.classList.add('err');
      document.getElementById('lbl').textContent = 'Error';
      document.getElementById('amt').textContent = data.error || 'Something went wrong';
    }
  } catch (err) {
    box.classList.remove('hidden'); box.classList.add('err');
    document.getElementById('lbl').textContent = 'Error';
    document.getElementById('amt').textContent = 'Could not reach server';
  }
  btn.textContent = 'Predict Insurance Cost';
  box.scrollIntoView({behavior: 'smooth', block: 'center'});
});
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json(force=True)
        row = {
            "age": int(d["age"]),
            "sex": SEX_MAP[d["sex"].lower()],
            "bmi": float(d["bmi"]),
            "children": int(d["children"]),
            "smoker": SMOKER_MAP[d["smoker"].lower()],
            "region": REGION_MAP[d["region"].lower()],
        }
        df = pd.DataFrame([row], columns=FEATURES)
        value = float(model.predict(df)[0])
        return jsonify({"prediction": f"$ {value:,.2f}"})
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
