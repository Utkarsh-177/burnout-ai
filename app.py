from flask import Flask, request, render_template_string, send_file, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests, os, numpy as np

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")

last_df = None
model = None
feature_cols = []
accuracy = None
ai_cache = {}

def auto_train(df):
    df.columns = df.columns.str.lower()
    num = df.select_dtypes(include=['int64','float64'])

    if len(num.columns) < 2:
        return None, None, None, None, None

    target = num.columns[-1]
    X = num.iloc[:,:-1]
    y_raw = num[target]

    y = pd.qcut(y_raw, q=3, labels=[0,1,2], duplicates='drop')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    m = RandomForestClassifier(n_estimators=120)
    m.fit(X_train, y_train)

    preds = m.predict(X_test)
    acc = round(accuracy_score(y_test, preds)*100,2)

    df["Burnout"] = [["Low","Medium","High"][int(i)] for i in m.predict(X)]

    stats = {
        "avg": round(df["Burnout"].map({"Low":1,"Medium":2,"High":3}).mean(),2),
        "high": int((df["Burnout"]=="High").sum()),
        "medium": int((df["Burnout"]=="Medium").sum()),
        "low": int((df["Burnout"]=="Low").sum()),
        "rows": len(df),
        "cols": len(df.columns),
        "accuracy": acc
    }

    return df, m, list(X.columns), stats, acc

def smart_ai(q, df, model):
    if df is None:
        return "Upload dataset first."

    q = q.lower()

    try:
        if "summary" in q:
            return df.describe().to_string()

        if "columns" in q:
            return ", ".join(df.columns)

        if "rows" in q or "size" in q:
            return f"{len(df)} rows and {len(df.columns)} columns"

        if "missing" in q:
            return df.isnull().sum().to_string()

        if "correlation" in q:
            return df.corr(numeric_only=True).to_string()

        if "average" in q:
            return df.mean(numeric_only=True).to_string()

        if "distribution" in q:
            return df["Burnout"].value_counts().to_string()

        if "high burnout" in q:
            return df[df["Burnout"]=="High"].head(10).to_string()

        if "risk" in q:
            return df[df["Burnout"]=="High"].describe().to_string()

        if "important" in q:
            imp = pd.Series(model.feature_importances_, index=df.select_dtypes(include=['int64','float64']).columns[:-1])
            return imp.sort_values(ascending=False).to_string()

        if "outlier" in q:
            z = np.abs((df.select_dtypes(include=['int64','float64']) - df.mean()) / df.std())
            return (z > 3).sum().to_string()

        if "compare" in q:
            high = df[df["Burnout"]=="High"].mean(numeric_only=True)
            low = df[df["Burnout"]=="Low"].mean(numeric_only=True)
            return "High:\n"+high.to_string()+"\n\nLow:\n"+low.to_string()

        if "recommend" in q:
            return "Reduce workload, increase breaks, and monitor high-risk employees."

        if "insight" in q:
            return "Higher numeric workload patterns strongly correlate with burnout."

    except:
        return None

    return None

def gemma_ai(q, df):
    if q in ai_cache:
        return ai_cache[q]

    local = smart_ai(q, df, model)
    if local:
        return local

    if not API_KEY:
        return local if local else "AI unavailable."

    try:
        summary = df.head(10).to_string() if df is not None else ""

        res = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}","Content-Type":"application/json"},
            json={
                "model":"meta/llama3-8b-instruct",
                "messages":[{"role":"user","content":q + "\\nDataset:\\n" + summary}],
                "max_tokens":300
            },
            timeout=25
        )

        data = res.json()
        if "choices" in data:
            reply = data["choices"][0]["message"]["content"]
            ai_cache[q] = reply
            return reply

        return local if local else "Temporary delay. Try again."

    except:
        return local if local else "Temporary delay. Try again."

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{margin:0;font-family:sans-serif;background:#0b0f19;color:white;display:flex}
.sidebar{width:220px;background:#020617;padding:20px;height:100vh;position:fixed}
.sidebar button{width:100%;margin:10px 0;padding:10px;background:#1f2937;border:none;color:white}
.main{margin-left:240px;padding:20px;width:100%}
.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}
.card{background:#111827;padding:15px;border-radius:8px;text-align:center}
.layout{display:grid;grid-template-columns:3fr 1fr;gap:10px}
table{width:100%;border-collapse:collapse}
td,th{padding:8px;border-bottom:1px solid #333}
.high{background:#5a1e1e}
.medium{background:#5a3e1e}
#chat{position:fixed;bottom:20px;right:20px;background:#2563eb;padding:12px;border-radius:50%;cursor:pointer}
#chatbox{position:fixed;bottom:80px;right:20px;width:300px;height:400px;background:#020617;display:none;flex-direction:column}
#chat-body{flex:1;overflow:auto;padding:10px}
.msg{margin:5px;padding:6px;border-radius:5px}
.user{background:#2563eb}
.ai{background:#1f2937}
</style>

<script>
function toggleChat(){
let c=document.getElementById("chatbox");
c.style.display=c.style.display==="flex"?"none":"flex";
}
function sendMessage(){
let i=document.getElementById("chat_text");
let m=i.value.trim();
if(!m)return;
let b=document.getElementById("chat-body");
b.innerHTML+=`<div class='msg user'>${m}</div>`;
let t=document.createElement("div");
t.className="msg ai";
t.innerHTML="Thinking...";
b.appendChild(t);
fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:m})})
.then(r=>r.json()).then(d=>{t.innerHTML=d.reply});
i.value="";
}
</script>
</head>

<body>

<div class="sidebar">
<button onclick="sendMessage('summary')">Summary</button>
<button onclick="sendMessage('correlation')">Correlation</button>
<button onclick="sendMessage('important')">Important Features</button>
<button onclick="sendMessage('recommend')">Recommendation</button>
</div>

<div class="main">

<div class="grid">
<div class="card">Avg<br>{{stats.avg}}</div>
<div class="card">High<br>{{stats.high}}</div>
<div class="card">Medium<br>{{stats.medium}}</div>
<div class="card">Low<br>{{stats.low}}</div>
<div class="card">Rows<br>{{stats.rows}}</div>
<div class="card">Accuracy<br>{{stats.accuracy}}%</div>
</div>

<div class="layout">

<div>
<form method="POST" enctype="multipart/form-data">
<input type="file" name="file">
<button>Upload</button>
</form>

{% if table %}
<table>
<tr>
{% for k in table[0].keys() %}
<th>{{k}}</th>
{% endfor %}
</tr>
{% for r in table[:50] %}
<tr class="{% if r['Burnout']=='High' %}high{% elif r['Burnout']=='Medium' %}medium{% endif %}">
{% for v in r.values() %}
<td>{{v}}</td>
{% endfor %}
</tr>
{% endfor %}
</table>

<canvas id="c"></canvas>

<script>
new Chart(document.getElementById("c"), {
type:'bar',
data:{labels:["Low","Medium","High"],
datasets:[{data:[{{stats.low}},{{stats.medium}},{{stats.high}}]}]}
});
</script>
{% endif %}
</div>

<div>
<h3>Insights</h3>
<p>AI-powered burnout prediction system analyzing workload patterns.</p>
</div>

</div>

</div>

<div id="chat" onclick="toggleChat()">Chat</div>

<div id="chatbox">
<div id="chat-body"></div>
<input id="chat_text" onkeydown="if(event.key==='Enter'){sendMessage()}">
</div>

</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():
    global last_df, model, feature_cols, accuracy
    table=None; stats=None

    if request.method=="POST":
        file = request.files.get("file")
        if file:
            df = pd.read_csv(file)
            df, model, feature_cols, stats, accuracy = auto_train(df)
            if df is not None:
                last_df = df
                table = df.to_dict(orient="records")

    return render_template_string(HTML, table=table, stats=stats)

@app.route("/chat", methods=["POST"])
def chat():
    return jsonify({"reply": gemma_ai(request.get_json()["message"], last_df)})

@app.route("/download")
def download():
    if last_df is not None:
        last_df.to_csv("report.csv", index=False)
        return send_file("report.csv", as_attachment=True)
    return "No data"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
