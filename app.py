from flask import Flask, request, render_template_string, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import requests, os, io

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")
last_df = None


# ---------------- DATA PROCESS ----------------
def auto_train(df):
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = df.loc[:, ~df.columns.duplicated()]

    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            except:
                df = df.drop(columns=[col])

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean(numeric_only=True))
    df = df.fillna(0)

    num = df.select_dtypes(include=np.number)

    if len(num.columns) < 2:
        df["Burnout"] = np.random.choice(["Low", "Medium", "High"], len(df))
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(num)
        km = KMeans(n_clusters=3, n_init=10, random_state=42)
        preds = km.fit_predict(X)
        df["Burnout"] = ["Low" if i == 0 else "Medium" if i == 1 else "High" for i in preds]

    stats = {
        "high": int((df["Burnout"] == "High").sum()),
        "medium": int((df["Burnout"] == "Medium").sum()),
        "low": int((df["Burnout"] == "Low").sum())
    }

    return df, stats


def add_productivity(df):
    df["Productivity"] = df["Burnout"].map({
        "Low": "High Productivity",
        "Medium": "Moderate Productivity",
        "High": "Low Productivity"
    })
    return df


def generate_recommendations(stats):
    rec = []
    total = sum(stats.values())

    if stats["high"] / total > 0.4:
        rec.append("Critical burnout levels detected. Reduce workload immediately.")
    elif stats["medium"] / total > 0.4:
        rec.append("Moderate burnout. Improve work-life balance.")
    else:
        rec.append("Healthy burnout levels observed.")

    rec.append("Ensure structured breaks and proper sleep cycles.")
    rec.append("Encourage productivity tracking and goal clarity.")

    return rec


# ---------------- AI CHAT ----------------
def ai_chat(q, df):
    if df is None:
        return "Upload dataset first."

    if not API_KEY:
        return "API key missing."

    try:
        summary = df.describe().to_string()

        res = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta/llama-3.3-70b-instruct",
                "messages": [{"role": "user", "content": f"{summary}\n\nQuestion: {q}"}],
                "max_tokens": 200
            },
            timeout=20
        )

        data = res.json()

        if "choices" not in data:
            return "AI response error."

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return str(e)


# ---------------- HTML (MODERN UI FIXED) ----------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body{margin:0;font-family:system-ui;background:#0f172a;color:#e2e8f0}

.container{max-width:1100px;margin:auto;padding:30px}

h1{text-align:center}

.upload{
display:block;
max-width:600px;
margin:20px auto;
padding:50px;
border:2px dashed #3b82f6;
border-radius:12px;
text-align:center;
cursor:pointer;
transition:0.3s;
}
.upload:hover{transform:scale(1.03);background:#020617}

.switch{display:flex;width:300px;margin:20px auto;background:#020617;border-radius:30px}
.option{flex:1;text-align:center;padding:10px;cursor:pointer}
.option.active{background:#3b82f6;border-radius:30px}

.stats{display:flex;justify-content:center;gap:20px;margin:20px}
.card{background:#020617;padding:15px;border-radius:10px;width:120px;text-align:center}

.table-box{max-height:300px;overflow:auto;margin-top:20px;border:1px solid #1e293b}
table{width:100%}
td,th{padding:8px;text-align:center}

button{padding:10px;margin:10px;background:#3b82f6;border:none;border-radius:6px;cursor:pointer}
button:hover{transform:scale(1.05)}

#chat{position:fixed;bottom:20px;right:20px;background:#3b82f6;padding:15px;border-radius:50%}
#chatbox{position:fixed;bottom:80px;right:20px;width:300px;height:400px;background:#020617;display:none;flex-direction:column}
#chat-body{flex:1;overflow:auto;padding:10px}
</style>

<script>
function switchView(type){
document.getElementById("burnout").style.display = type==="burnout"?"block":"none"
document.getElementById("prod").style.display = type==="prod"?"block":"none"
}

function toggleChat(){
let c=document.getElementById("chatbox")
c.style.display=c.style.display==="flex"?"none":"flex"
}

function sendMessage(){
let i=document.getElementById("chat_text")
let m=i.value.trim()
if(!m)return

let b=document.getElementById("chat-body")
b.innerHTML+=`<div>${m}</div>`

fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:m})})
.then(r=>r.json()).then(d=>{
b.innerHTML+=`<div>${d.reply}</div>`
})

i.value=""
}
</script>
</head>

<body>

<div class="container">

<h1>Burnout AI</h1>

<form method="POST" enctype="multipart/form-data">
<label class="upload">
Upload Dataset
<input type="file" name="file" hidden onchange="this.form.submit()">
</label>
</form>

<div class="switch">
<div class="option active" onclick="switchView('burnout')">Burnout</div>
<div class="option" onclick="switchView('prod')">Productivity</div>
</div>

<div id="burnout">
{% if stats %}
<div class="stats">
<div class="card">High {{stats.high}}</div>
<div class="card">Medium {{stats.medium}}</div>
<div class="card">Low {{stats.low}}</div>
</div>
<canvas id="chart1"></canvas>
{% endif %}
</div>

<div id="prod" style="display:none">
{% if prod %}
<div class="stats">
<div class="card">High {{prod.high}}</div>
<div class="card">Medium {{prod.medium}}</div>
<div class="card">Low {{prod.low}}</div>
</div>
<canvas id="chart2"></canvas>
{% endif %}
</div>

{% if table %}
<div class="table-box">
<table>
<tr>{% for k in table[0].keys() %}<th>{{k}}</th>{% endfor %}</tr>
{% for r in table[:30] %}
<tr>{% for v in r.values() %}<td>{{v}}</td>{% endfor %}</tr>
{% endfor %}
</table>
</div>
{% endif %}

{% if recommendations %}
<div>
<h3>Recommendations</h3>
<ul>
{% for r in recommendations %}
<li>{{r}}</li>
{% endfor %}
</ul>
</div>
{% endif %}

<a href="/download/burnout"><button>Download Burnout Report</button></a>
<a href="/download/productivity"><button>Download Productivity Report</button></a>

</div>

<div id="chat" onclick="toggleChat()">Chat</div>

<div id="chatbox">
<div id="chat-body"></div>
<input id="chat_text" onkeydown="if(event.key==='Enter'){sendMessage()}">
</div>

<script>
{% if stats %}
new Chart(document.getElementById('chart1'),{
type:'bar',
data:{labels:['Low','Medium','High'],datasets:[{data:[{{stats.low}},{{stats.medium}},{{stats.high}}]}]}
});
{% endif %}

{% if prod %}
new Chart(document.getElementById('chart2'),{
type:'bar',
data:{labels:['Low','Medium','High'],datasets:[{data:[{{prod.low}},{{prod.medium}},{{prod.high}}]}]}
});
{% endif %}
</script>

</body>
</html>
"""


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET","POST"])
def home():
    global last_df
    stats = None
    table = None
    prod = None
    rec = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            df = pd.read_csv(file, on_bad_lines="skip")
            df, stats = auto_train(df)
            df = add_productivity(df)
            last_df = df

            table = df.to_dict(orient="records")

            prod = {
                "high": int((df["Productivity"]=="High Productivity").sum()),
                "medium": int((df["Productivity"]=="Moderate Productivity").sum()),
                "low": int((df["Productivity"]=="Low Productivity").sum())
            }

            rec = generate_recommendations(stats)

    return render_template_string(HTML, stats=stats, table=table, prod=prod, recommendations=rec)


@app.route("/chat", methods=["POST"])
def chat():
    msg = request.get_json().get("message")
    return jsonify({"reply": ai_chat(msg, last_df)})


@app.route("/download/<type>")
def download(type):
    if last_df is None:
        return "No data"

    output = io.StringIO()

    if type == "burnout":
        last_df[["Burnout"]].to_csv(output, index=False)
    else:
        last_df[["Productivity"]].to_csv(output, index=False)

    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), download_name=f"{type}.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
