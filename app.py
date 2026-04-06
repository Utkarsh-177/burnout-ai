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

def auto_train(df):
    df.columns = df.columns.str.lower()
    df = df.loc[:, ~df.columns.duplicated()]

    for col in df.select_dtypes(include=['object']).columns:
        try:
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                df = df.drop(columns=[col])
            else:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        except:
            df = df.drop(columns=[col])

    df = df.fillna(df.mean(numeric_only=True))
    num = df.select_dtypes(include=np.number)

    if len(num.columns) < 2:
        return df, None, {"high":0,"medium":0,"low":0}

    target_cols = [c for c in df.columns if any(x in c for x in ["target","label","burnout","stress","output"])]

    if target_cols:
        target = target_cols[0]
        X = num.drop(columns=[target], errors='ignore')
        y = df[target]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        global model
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train,y_train)

        preds = model.predict(X)

    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(num)
        km = KMeans(n_clusters=3, n_init=10)
        preds = km.fit_predict(X)

    df["Burnout"] = ["Low" if i==0 else "Medium" if i==1 else "High" for i in preds]

    stats = {
        "high": int((df["Burnout"]=="High").sum()),
        "medium": int((df["Burnout"]=="Medium").sum()),
        "low": int((df["Burnout"]=="Low").sum())
    }

    return df, model, stats

def recommendations(stats):
    rec = []
    total = stats["high"] + stats["medium"] + stats["low"]

    if total == 0:
        return ["No data available"]

    high_ratio = stats["high"] / total
    medium_ratio = stats["medium"] / total

    if high_ratio > 0.4:
        rec.append("High burnout levels detected. Immediate action required.")
        rec.append("Reduce workload and enforce regular breaks.")
        rec.append("Provide mental health support.")

    elif medium_ratio > 0.4:
        rec.append("Moderate burnout detected. Monitor closely.")
        rec.append("Improve work-life balance and task distribution.")

    else:
        rec.append("Burnout levels are under control.")
        rec.append("Maintain current work practices.")

    rec.append("Encourage proper sleep and physical activity.")
    rec.append("Create a positive and supportive environment.")

    return rec

def smart_answer(q, df):
    q = q.lower()

    try:
        if "average" in q or "mean" in q:
            return df.mean(numeric_only=True).to_string()

        if "max" in q:
            return df.max(numeric_only=True).to_string()

        if "min" in q:
            return df.min(numeric_only=True).to_string()

        if "correlation" in q:
            return df.corr(numeric_only=True).to_string()

        if "count" in q or "rows" in q:
            return str(len(df))

        if "columns" in q:
            return ", ".join(df.columns)

        if "burnout" in q:
            return df["Burnout"].value_counts().to_string()

    except:
        return None

    return None

# ⚠️ CHAT FUNCTION LEFT EXACTLY SAME
def ai_chat(q, df):
    if df is None:
        return "Upload dataset first."

    local = smart_answer(q, df)

    if not API_KEY:
        return local or "API key missing."

    try:
        summary = df.describe().to_string()

        prompt = f"""
You are a data analyst.

Dataset summary:
{summary}

User question:
{q}

Computed answer:
{local}

Explain clearly in simple words. No raw tables.
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
            },
            timeout=20
        )

        data = res.json()

        if "choices" not in data:
            return str(data)

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return str(e)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body{margin:0;font-family:system-ui;background:#0f172a;color:#e2e8f0}
.container{max-width:1100px;margin:auto;padding:30px}
h1{text-align:center;margin-bottom:20px}

.upload{
display:block;
max-width:600px;
margin:0 auto 30px;
padding:40px;
border:2px dashed #334155;
border-radius:12px;
text-align:center;
cursor:pointer;
}

.stats{
display:flex;
justify-content:center;
gap:20px;
flex-wrap:wrap;
margin-bottom:30px;
}

.card{
background:#020617;
padding:15px;
border-radius:10px;
width:140px;
text-align:center;
}

.section{margin-top:40px}

.table-box{
border:1px solid #1e293b;
border-radius:10px;
overflow:auto;
}

table{width:100%;border-collapse:collapse}
td,th{padding:10px;border-bottom:1px solid #1e293b;text-align:center}

#chat{position:fixed;bottom:20px;right:20px;background:#3b82f6;padding:14px;border-radius:50%;cursor:pointer}
#chatbox{position:fixed;bottom:80px;right:20px;width:320px;height:420px;background:#020617;display:none;flex-direction:column;border-radius:10px;border:1px solid #1e293b}
#chat-body{flex:1;overflow:auto;padding:10px}
.msg{margin:5px;padding:8px;border-radius:6px}
.user{background:#3b82f6}
.ai{background:#1e293b}
</style>

<script>
function toggleChat(){
let c=document.getElementById("chatbox")
c.style.display = c.style.display==="flex"?"none":"flex"
}

function sendMessage(){
let i=document.getElementById("chat_text")
let m=i.value.trim()
if(!m)return

let b=document.getElementById("chat-body")
b.innerHTML+=`<div class='msg user'>${m}</div>`

let t=document.createElement("div")
t.className="msg ai"
t.innerHTML="Processing..."
b.appendChild(t)

fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:m})})
.then(r=>r.json()).then(d=>{t.innerHTML=d.reply})

i.value=""
}
</script>
</head>

<body>

<div class="container">

<h1>Burnout AI Dashboard</h1>

<form method="POST" enctype="multipart/form-data">
<label class="upload">
Upload CSV Dataset
<input type="file" name="file" hidden onchange="this.form.submit()">
</label>
</form>

{% if stats %}
<div class="stats">
<div class="card">High<br>{{stats.high}}</div>
<div class="card">Medium<br>{{stats.medium}}</div>
<div class="card">Low<br>{{stats.low}}</div>
</div>
{% endif %}

{% if table %}
<div class="section">
<canvas id="chart"></canvas>

<script>
new Chart(document.getElementById('chart'),{
type:'bar',
data:{labels:['Low','Medium','High'],datasets:[{data:[{{stats.low}},{{stats.medium}},{{stats.high}}]}]}
})
</script>
</div>

<div class="section">
<div class="table-box">
<table>
<tr>{% for k in table[0].keys() %}<th>{{k}}</th>{% endfor %}</tr>
{% for r in table[:20] %}
<tr>{% for v in r.values() %}<td>{{v}}</td>{% endfor %}</tr>
{% endfor %}
</table>
</div>
</div>
{% endif %}

{% if recommendations %}
<div class="section">
<h3>Recommendations</h3>
<div style="background:#020617;padding:20px;border-radius:10px">
<ul>
{% for r in recommendations %}
<li style="margin:10px 0">{{r}}</li>
{% endfor %}
</ul>
</div>
</div>
{% endif %}

</div>

<div id="chat" onclick="toggleChat()">Chat</div>

<div id="chatbox">
<div id="chat-body"></div>
<input id="chat_text" placeholder="Ask about dataset" onkeydown="if(event.key==='Enter'){sendMessage()}">
</div>

</body>
</html>
"""

@app.route("/",methods=["GET","POST"])
def home():
    global last_df
    stats=None
    table=None
    rec=None

    if request.method=="POST":
        file=request.files.get("file")
        if file:
            df=pd.read_csv(file)
            df,_,stats=auto_train(df)
            last_df=df
            table=df.to_dict(orient="records")
            rec=recommendations(stats)

    return render_template_string(HTML,stats=stats,table=table,recommendations=rec)

@app.route("/chat",methods=["POST"])
def chat():
    return jsonify({"reply":ai_chat(request.get_json()["message"],last_df)})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",10000)))
