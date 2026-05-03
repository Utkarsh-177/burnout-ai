from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import requests, os

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")

last_df = None
model = None


# -------------------- ML PART --------------------
def auto_train(df):
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = df.loc[:, ~df.columns.duplicated()]

    for col in df.columns:
        if df[col].dtype == "object":
            try:
                if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    df = df.drop(columns=[col])
                else:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            except:
                df = df.drop(columns=[col])

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean(numeric_only=True))
    df = df.fillna(0)

    num = df.select_dtypes(include=np.number)

    if len(num.columns) < 2:
        df["Burnout"] = np.random.choice(["Low", "Medium", "High"], len(df))
        stats = {"high": 0, "medium": 0, "low": 0}
        return df, None, stats

    try:
        scaler = StandardScaler()
        X = scaler.fit_transform(num)

        km = KMeans(n_clusters=3, n_init=10, random_state=42)
        preds = km.fit_predict(X)

    except:
        preds = np.random.choice([0, 1, 2], len(df))

    df["Burnout"] = ["Low" if i == 0 else "Medium" if i == 1 else "High" for i in preds]

    stats = {
        "high": int((df["Burnout"] == "High").sum()),
        "medium": int((df["Burnout"] == "Medium").sum()),
        "low": int((df["Burnout"] == "Low").sum())
    }

    return df, None, stats


def add_productivity(df):
    df["Productivity"] = df["Burnout"].map({
        "Low": "High Productivity",
        "Medium": "Moderate Productivity",
        "High": "Low Productivity"
    })
    return df


def recommendations(stats):
    rec = []
    total = stats["high"] + stats["medium"] + stats["low"]

    if total == 0:
        return ["No data available"]

    if stats["high"] / total > 0.4:
        rec.append("High burnout detected. Reduce workload immediately.")
    elif stats["medium"] / total > 0.4:
        rec.append("Moderate burnout detected. Monitor workload.")
    else:
        rec.append("Burnout levels are stable.")

    rec.append("Encourage sleep and exercise.")
    rec.append("Maintain balanced work environment.")

    return rec


# -------------------- AI CHAT FIXED --------------------
def ai_chat(q, df):
    if df is None:
        return "Upload dataset first."

    if not API_KEY:
        return "API key missing."

    try:
        summary = df.describe().to_string()

        prompt = f"""
You are an AI assistant for burnout prediction system.

Dataset:
{summary}

User question:
{q}

Give short clear answer.
"""

        res = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta/llama-3.3-70b-instruct",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 200
            },
            timeout=25
        )

        try:
            data = res.json()
        except Exception:
            return f"API error: {res.text}"

        if "choices" not in data:
            return str(data)

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return str(e)


# -------------------- YOUR ORIGINAL UI (UNCHANGED) --------------------
HTML = """ 
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body{margin:0;font-family:system-ui;background:#0f172a;color:#e2e8f0}

.container{
max-width:1100px;
margin:auto;
padding:30px;
animation:fadeIn 0.5s ease;
}

@keyframes fadeIn{
from{opacity:0;transform:translateY(10px)}
to{opacity:1;transform:translateY(0)}
}

h1{text-align:center;margin-bottom:10px}

.upload{
display:block;
max-width:600px;
margin:0 auto 30px;
padding:50px;
border:2px dashed #3b82f6;
border-radius:12px;
text-align:center;
cursor:pointer;
transition:0.3s;
}
.upload:hover{transform:scale(1.02);background:#020617}

.switch-wrapper{display:flex;justify-content:center;margin-bottom:25px}
.switch{
position:relative;width:260px;height:45px;background:#020617;
border-radius:30px;display:flex;align-items:center;overflow:hidden
}
.option{flex:1;text-align:center;z-index:2;cursor:pointer;color:#94a3b8}
.option.active{color:white;font-weight:600}
.slider{
position:absolute;width:50%;height:100%;background:#3b82f6;
border-radius:30px;transition:0.3s;left:0
}

.view-container{overflow:hidden}
.views{display:flex;width:200%;transition:transform 0.4s ease}
.screen{width:100%}

.stats{display:flex;justify-content:center;gap:20px;flex-wrap:wrap;margin-bottom:20px}
.card{background:#020617;padding:15px;border-radius:10px;width:140px;text-align:center}

.table-box{
border:1px solid #1e293b;border-radius:10px;overflow:auto;max-height:350px
}
table{width:100%;border-collapse:collapse}
td,th{padding:10px;border-bottom:1px solid #1e293b;text-align:center}
tr:hover{background:#1e293b}

#chat{position:fixed;bottom:20px;right:20px;background:#3b82f6;padding:14px;border-radius:50%;cursor:pointer}
#chatbox{
position:fixed;bottom:80px;right:20px;width:320px;height:420px;
background:#020617;display:none;flex-direction:column;border-radius:10px;border:1px solid #1e293b
}
#chat-body{flex:1;overflow:auto;padding:10px}
.msg{margin:6px;padding:8px;border-radius:6px;font-size:13px}
.user{background:#3b82f6}
.ai{background:#1e293b}
</style>

<script>
function switchView(index){
document.getElementById("views").style.transform=`translateX(-${index*50}%)`
let slider=document.getElementById("slider")
slider.style.left=index===0?"0%":"50%"
let options=document.querySelectorAll(".option")
options.forEach(o=>o.classList.remove("active"))
options[index].classList.add("active")
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
b.innerHTML+=`<div class='msg user'>${m}</div>`

let t=document.createElement("div")
t.className="msg ai"
t.innerHTML="Analyzing..."
b.appendChild(t)

fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:m})})
.then(r=>r.json()).then(d=>{
t.innerHTML=d.reply
b.scrollTop=b.scrollHeight
})

i.value=""
}
</script>
</head>

<body>

<div class="container">

<h1>Burnout AI</h1>
<p style="text-align:center;color:#94a3b8">AI-powered Burnout & Productivity Insights</p>

<form method="POST" enctype="multipart/form-data">
<label class="upload">
Upload Dataset
<input type="file" name="file" hidden onchange="this.form.submit()">
</label>
</form>

<div class="switch-wrapper">
<div class="switch">
<div class="slider" id="slider"></div>
<div class="option active" onclick="switchView(0)">Burnout</div>
<div class="option" onclick="switchView(1)">Productivity</div>
</div>
</div>

<div class="view-container">
<div class="views" id="views">

<div class="screen">
{% if stats %}
<div class="stats">
<div class="card">High<br>{{stats.high}}</div>
<div class="card">Medium<br>{{stats.medium}}</div>
<div class="card">Low<br>{{stats.low}}</div>
</div>
<canvas id="chart1"></canvas>
{% endif %}
</div>

<div class="screen">
{% if prod %}
<div class="stats">
<div class="card">High<br>{{prod.high}}</div>
<div class="card">Medium<br>{{prod.medium}}</div>
<div class="card">Low<br>{{prod.low}}</div>
</div>
<canvas id="chart2"></canvas>
{% endif %}
</div>

</div>
</div>

{% if table %}
<div class="table-box">
<table>
<tr>{% for k in table[0].keys() %}<th>{{k}}</th>{% endfor %}</tr>
{% for r in table[:20] %}
<tr>{% for v in r.values() %}<td>{{v}}</td>{% endfor %}</tr>
{% endfor %}
</table>
</div>
{% endif %}

{% if recommendations %}
<div style="margin-top:40px">
<h3 style="text-align:center">Recommendations</h3>
<div style="background:#020617;padding:20px;border-radius:12px;max-width:700px;margin:20px auto">
<ul style="list-style:none;padding:0">
{% for r in recommendations %}
<li style="margin:10px 0;padding:10px;background:#0f172a;border-left:4px solid #3b82f6">
{{r}}
</li>
{% endfor %}
</ul>
</div>
</div>
{% endif %}

</div>

<div id="chat" onclick="toggleChat()">Chat</div>

<div id="chatbox">
<div id="chat-body"></div>
<input id="chat_text" placeholder="Ask..." onkeydown="if(event.key==='Enter'){sendMessage()}">
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


# -------------------- ROUTES --------------------
@app.route("/", methods=["GET", "POST"])
def home():
    global last_df

    stats = None
    table = None
    prod = None
    rec = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
            df, _, stats = auto_train(df)
            df = add_productivity(df)

            last_df = df
            table = df.to_dict(orient="records")

            prod = {
                "high": int((df["Productivity"]=="High Productivity").sum()),
                "medium": int((df["Productivity"]=="Moderate Productivity").sum()),
                "low": int((df["Productivity"]=="Low Productivity").sum())
            }

            rec = recommendations(stats)

    return render_template_string(HTML, stats=stats, table=table, prod=prod, recommendations=rec)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = data.get("message", "")
    return jsonify({"reply": ai_chat(msg, last_df)})


if __name__ == "__main__":
    app.run(debug=True)
