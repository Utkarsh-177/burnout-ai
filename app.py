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
            return None, None, None, "Not enough numeric data"
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

    for _ in range(2):
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
                json={"model":"meta/llama3-70b-instruct","messages":messages,"temperature":0.5,"max_tokens":400},
                timeout=10
            )

            reply = res.json()["choices"][0]["message"]["content"]
            chat_memory.append({"user":question,"ai":reply})
            ai_cache[question] = reply

            return reply

        except:
            time.sleep(1)

    return "AI busy"

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI Ultimate</title>
<style>
body{font-family:sans-serif;background:#0f172a;color:white;margin:0}
.container{width:90%;margin:auto;padding:20px}
.card{background:#1e293b;padding:20px;border-radius:12px;margin:15px 0}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.kpi{padding:15px;border-radius:10px;text-align:center}
.high{background:#dc2626}
.medium{background:#f59e0b}
.low{background:#16a34a}
.avg{background:#2563eb}
input,button{padding:10px;margin:5px;border:none;border-radius:8px}
button{background:#3b82f6;color:white;cursor:pointer}
table{width:100%}
td,th{padding:10px;text-align:center}
#chatbox{position:fixed;bottom:80px;right:20px;width:300px;height:400px;background:black;display:none;flex-direction:column}
#chat-body{flex:1;overflow-y:auto;padding:10px}
.msg{margin:5px;padding:8px;border-radius:8px}
.user{background:#2563eb}
.ai{background:#374151}
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
t.innerHTML="⚡ Thinking...";
body.appendChild(t);
fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:m})})
.then(r=>r.json()).then(d=>{t.innerHTML=d.reply;body.scrollTop=body.scrollHeight});
i.value="";
}
</script>
</head>

<body>
<div class="container">
<h1>🔥 Burnout AI Ultimate</h1>

<div class="card">
<form method="POST" enctype="multipart/form-data">
<input type="file" name="file">
<button>Upload Dataset & Train</button>
</form>
</div>

{% if stats %}
<div class="grid">
<div class="kpi avg">Avg<br>{{stats.avg}}</div>
<div class="kpi high">High<br>{{stats.high}}</div>
<div class="kpi medium">Medium<br>{{stats.medium}}</div>
<div class="kpi low">Low<br>{{stats.low}}</div>
</div>
{% endif %}

{% if table %}
<div class="card">
<table>
<tr><th>Preview</th></tr>
{% for r in table[:10] %}
<tr><td>{{r}}</td></tr>
{% endfor %}
</table>
</div>
{% endif %}

{% if bar %}
<div class="card"><img src="data:image/png;base64,{{bar}}"></div>
{% endif %}

{% if pie %}
<div class="card"><img src="data:image/png;base64,{{pie}}"></div>
{% endif %}

{% if fi %}
<div class="card"><img src="data:image/png;base64,{{fi}}"></div>
{% endif %}

<a href="/download"><button>Download</button></a>

</div>

<div onclick="toggleChat()" style="position:fixed;bottom:20px;right:20px;background:#3b82f6;padding:15px;border-radius:50%">💬</div>

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