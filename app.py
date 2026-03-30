from flask import Flask, request, render_template_string, send_file, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import io, base64, requests, time, os

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")

last_df = None
model = None
feature_cols = []
chat_memory = []
ai_cache = {}

def auto_train(df):
    df.columns = df.columns.str.lower()

    target = None
    for col in df.columns:
        if "burnout" in col or "target" in col:
            target = col
            break

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

    if target and target in numeric_cols:
        y = df[target]
        X = df.drop(columns=[target]).select_dtypes(include=['int64','float64'])
    else:
        if len(numeric_cols) < 2:
            return None, None, None, None
        target = numeric_cols[-1]
        X = df[numeric_cols[:-1]]
        y = df[target]

    y = pd.qcut(y, q=3, labels=[0,1,2], duplicates='drop')

    m = RandomForestClassifier(n_estimators=200)
    m.fit(X, y)

    df["Burnout"] = [["Low","Medium","High"][int(i)] for i in m.predict(X)]

    avg = df["Burnout"].map({"Low":1,"Medium":2,"High":3}).mean()
    stats = {
        "avg": round(avg,2),
        "high": int((df["Burnout"]=="High").sum()),
        "medium": int((df["Burnout"]=="Medium").sum()),
        "low": int((df["Burnout"]=="Low").sum())
    }

    return df, m, list(X.columns), stats

def generate_bar(df):
    plt.figure()
    df["Burnout"].value_counts().plot(kind="bar")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close()
    return img

def generate_pie(df):
    plt.figure()
    df["Burnout"].value_counts().plot(kind="pie", autopct="%1.1f%%")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close()
    return img

def feature_importance(m, cols):
    imp = m.feature_importances_
    plt.figure()
    plt.bar(cols, imp)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close()
    return img

def gemma_ai(question, df_context=None):
    global chat_memory, ai_cache

    if not API_KEY:
        return "API key not configured"

    if question in ai_cache:
        return ai_cache[question]

    for _ in range(3):
        try:
            messages = [{"role":"system","content":"You are an expert AI analyzing datasets and explaining insights clearly."}]

            for m in chat_memory[-2:]:
                messages.append({"role":"user","content":m["user"]})
                messages.append({"role":"assistant","content":m["ai"]})

            if df_context is not None:
                summary = df_context.describe().to_string()
                question += f"\nDataset Summary:\n{summary}"

            messages.append({"role":"user","content":question})

            res = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model":"meta/llama3-8b-instruct",
                    "messages":messages,
                    "temperature":0.5,
                    "max_tokens":400
                },
                timeout=25
            )

            reply = res.json()["choices"][0]["message"]["content"]

            chat_memory.append({"user":question,"ai":reply})
            ai_cache[question] = reply

            return reply

        except:
            time.sleep(1)

    return "AI is currently busy. Please try again."

HTML = """ 
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
body {font-family:'Inter';background:#0b1120;color:#e5e7eb;margin:0}
.header {padding:20px;text-align:center;background:#111827;font-size:24px}
.container {width:92%;margin:auto;padding:20px}
.card {background:#111827;border-radius:12px;padding:20px;margin-bottom:20px;border:1px solid #1f2937}
.grid {display:grid;grid-template-columns:repeat(4,1fr);gap:15px}
.kpi {padding:20px;border-radius:10px;text-align:center}
.kpi strong {display:block;font-size:22px}
.high{background:#7f1d1d}.medium{background:#78350f}.low{background:#064e3b}.avg{background:#1e3a8a}
input,button{padding:10px;border-radius:6px;border:none;margin:5px}
input{background:#1f2937;color:white}
button{background:#2563eb;color:white;cursor:pointer}
table{width:100%}
td,th{padding:10px;border-bottom:1px solid #1f2937;text-align:center}
#chat-toggle{position:fixed;bottom:20px;right:20px;background:#2563eb;padding:14px;border-radius:50%;cursor:pointer}
#chatbox{position:fixed;bottom:80px;right:20px;width:320px;height:420px;background:#111827;border-radius:10px;display:none;flex-direction:column;border:1px solid #1f2937}
#chat-body{flex:1;overflow-y:auto;padding:10px}
.msg{margin:6px;padding:8px;border-radius:6px}
.user{background:#1d4ed8}.ai{background:#374151}
</style>

<script>
function toggleChat(){
let b=document.getElementById("chatbox");
b.style.display=b.style.display==="flex"?"none":"flex";
}
function sendMessage(){
let i=document.getElementById("chat_text");
let m=i.value.trim();
if(!m)return;
let body=document.getElementById("chat-body");
body.innerHTML+=`<div class='msg user'>${m}</div>`;
let t=document.createElement("div");
t.className="msg ai";
t.innerHTML="Analyzing...";
body.appendChild(t);
fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:m})})
.then(r=>r.json()).then(d=>{t.innerHTML=d.reply;body.scrollTop=body.scrollHeight});
i.value="";
}
</script>
</head>

<body>

<div class="header">Burnout AI Analytics Dashboard</div>

<div class="container">

<div class="card">
<form method="POST" enctype="multipart/form-data">
<input type="file" name="file">
<button>Upload Dataset</button>
</form>
</div>

{% if stats %}
<div class="grid">
<div class="kpi avg">Average<strong>{{stats.avg}}</strong></div>
<div class="kpi high">High<strong>{{stats.high}}</strong></div>
<div class="kpi medium">Medium<strong>{{stats.medium}}</strong></div>
<div class="kpi low">Low<strong>{{stats.low}}</strong></div>
</div>
{% endif %}

{% if table %}
<div class="card">
<table>
<tr><th>Data Preview</th></tr>
{% for r in table[:10] %}
<tr><td>{{r}}</td></tr>
{% endfor %}
</table>
</div>
{% endif %}

{% if bar %}<div class="card"><img src="data:image/png;base64,{{bar}}"></div>{% endif %}
{% if pie %}<div class="card"><img src="data:image/png;base64,{{pie}}"></div>{% endif %}
{% if fi %}<div class="card"><img src="data:image/png;base64,{{fi}}"></div>{% endif %}

<a href="/download"><button>Download Report</button></a>

</div>

<div id="chat-toggle" onclick="toggleChat()">Chat</div>

<div id="chatbox">
<div id="chat-body"></div>
<input id="chat_text" placeholder="Ask..." onkeydown="if(event.key==='Enter'){sendMessage()}">
</div>

</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():
    global last_df, model, feature_cols
    table=None; stats=None; bar=None; pie=None; fi=None

    if request.method=="POST":
        if "file" in request.files and request.files["file"].filename!="":
            df = pd.read_csv(request.files["file"])
            df, model, feature_cols, stats = auto_train(df)

            if df is not None:
                last_df = df
                table = df.to_dict(orient="records")
                bar = generate_bar(df)
                pie = generate_pie(df)
                fi = feature_importance(model, feature_cols)

    return render_template_string(HTML, table=table, stats=stats, bar=bar, pie=pie, fi=fi)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    return jsonify({"reply": gemma_ai(data["message"], last_df)})

@app.route("/download")
def download():
    global last_df
    if last_df is not None:
        last_df.to_csv("report.csv", index=False)
        return send_file("report.csv", as_attachment=True)
    return "No data"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
