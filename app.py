from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import requests, os

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")

last_df = None
model = None

def auto_train(df):
    df.columns = df.columns.str.lower()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df = df.fillna(df.mean(numeric_only=True))

    num = df.select_dtypes(include=np.number)

    if len(num.columns) < 2:
        return df, None, {"accuracy":0,"high":0,"medium":0,"low":0}

    target_cols = [c for c in df.columns if any(x in c for x in ["target","label","burnout","stress","output"])]

    acc = 0

    if target_cols:
        target = target_cols[0]
        X = num.drop(columns=[target], errors='ignore')
        y = df[target]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        global model
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train,y_train)
        preds = model.predict(X)

        acc = round(accuracy_score(y_test, model.predict(X_test))*100,2)

    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(num)

        km = KMeans(n_clusters=3, n_init=10)
        preds = km.fit_predict(X)

    df["Burnout"] = ["Low" if i==0 else "Medium" if i==1 else "High" for i in preds]

    stats = {
        "high": int((df["Burnout"]=="High").sum()),
        "medium": int((df["Burnout"]=="Medium").sum()),
        "low": int((df["Burnout"]=="Low").sum()),
        "accuracy": acc
    }

    return df, model, stats

def insights(df):
    text = []
    if (df["Burnout"]=="High").sum() > len(df)*0.4:
        text.append("High burnout risk across dataset")
    text.append("Balanced workload and rest is recommended")
    return text

def ai_chat(q, df):
    if df is None:
        return "Upload dataset first."

    if not API_KEY:
        return "API key missing."

    try:
        summary = df.describe().to_string()

        prompt = f"""
You are a data analyst.

Dataset Summary:
{summary}

User Question:
{q}

Give short insights, not raw data.
"""

        res = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta/llama3-8b-instruct",
                "messages":[{"role":"user","content":prompt}],
                "max_tokens":200
            }
        )

        return res.json()["choices"][0]["message"]["content"]

    except:
        return "AI error"

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI Pro</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body{margin:0;background:#0f172a;color:white;font-family:sans-serif}
.container{max-width:1100px;margin:auto;padding:30px;text-align:center}
.upload{border:2px dashed #3b82f6;padding:40px;border-radius:12px;margin-bottom:20px;cursor:pointer}
.grid{display:flex;justify-content:center;gap:15px;margin:20px}
.card{background:#020617;padding:15px;border-radius:10px;width:120px}
table{width:100%;margin-top:20px;border-collapse:collapse}
td,th{padding:10px;border-bottom:1px solid #333}
#chat{position:fixed;bottom:20px;right:20px;background:#3b82f6;padding:12px;border-radius:50%}
#chatbox{position:fixed;bottom:80px;right:20px;width:300px;height:400px;background:#020617;display:none;flex-direction:column}
#chat-body{flex:1;overflow:auto;padding:10px}
</style>

<script>
function toggleChat(){
let c=document.getElementById("chatbox")
c.style.display=c.style.display==="flex"?"none":"flex"
}

function sendMessage(){
let i=document.getElementById("chat_text")
let m=i.value.trim()
if(!m)return

let b=document.getElementById("chat-body")
b.innerHTML+=`<div style='background:#3b82f6;padding:5px'>${m}</div>`

fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:m})})
.then(r=>r.json()).then(d=>{
b.innerHTML+=`<div style='background:#1e293b;padding:5px'>${d.reply}</div>`
})

i.value=""
}
</script>
</head>

<body>

<div class="container">

<h1>🔥 Burnout AI Pro</h1>

<form method="POST" enctype="multipart/form-data">
<label class="upload">
Upload CSV
<input type="file" name="file" hidden onchange="this.form.submit()">
</label>
</form>

{% if stats %}
<div class="grid">
<div class="card">High<br>{{stats.high}}</div>
<div class="card">Medium<br>{{stats.medium}}</div>
<div class="card">Low<br>{{stats.low}}</div>
<div class="card">Accuracy<br>{{stats.accuracy}}</div>
</div>
{% endif %}

{% if table %}
<table>
<tr>{% for k in table[0].keys() %}<th>{{k}}</th>{% endfor %}</tr>
{% for r in table[:20] %}
<tr>{% for v in r.values() %}<td>{{v}}</td>{% endfor %}</tr>
{% endfor %}
</table>

<canvas id="chart"></canvas>

<script>
new Chart(document.getElementById('chart'),{
type:'bar',
data:{labels:['Low','Medium','High'],datasets:[{data:[{{stats.low}},{{stats.medium}},{{stats.high}}]}]}
})
</script>

{% endif %}

{% if insights %}
<ul>
{% for i in insights %}
<li>{{i}}</li>
{% endfor %}
</ul>
{% endif %}

</div>

<div id="chat" onclick="toggleChat()">💬</div>

<div id="chatbox">
<div id="chat-body"></div>
<input id="chat_text" placeholder="Ask..." onkeydown="if(event.key==='Enter'){sendMessage()}">
</div>

</body>
</html>
"""

@app.route("/",methods=["GET","POST"])
def home():
    global last_df
    stats=None
    table=None
    ins=None

    if request.method=="POST":
        file=request.files.get("file")
        if file:
            df=pd.read_csv(file)
            df,_,stats=auto_train(df)
            last_df=df
            table=df.to_dict(orient="records")
            ins=insights(df)

    return render_template_string(HTML,stats=stats,table=table,insights=ins)

@app.route("/chat",methods=["POST"])
def chat():
    return jsonify({"reply":ai_chat(request.get_json()["message"],last_df)})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",10000)))
