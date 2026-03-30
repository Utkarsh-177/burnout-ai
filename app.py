from flask import Flask, request, render_template_string, send_file, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import io, base64, requests, os

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")

last_df = None
model = None
feature_cols = []
ai_cache = {}

def auto_train(df):
    try:
        df.columns = df.columns.str.lower()

        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        if len(numeric_cols) < 2:
            return None, None, None, None

        target = numeric_cols[-1]
        X = df[numeric_cols[:-1]]
        y = df[target]

        y = pd.qcut(y, q=3, labels=[0,1,2], duplicates='drop')

        m = RandomForestClassifier(n_estimators=120)
        m.fit(X, y)

        df["Burnout"] = [["Low","Medium","High"][int(i)] for i in m.predict(X)]

        stats = {
            "avg": round(df["Burnout"].map({"Low":1,"Medium":2,"High":3}).mean(),2),
            "high": int((df["Burnout"]=="High").sum()),
            "medium": int((df["Burnout"]=="Medium").sum()),
            "low": int((df["Burnout"]=="Low").sum())
        }

        return df, m, list(X.columns), stats

    except:
        return None, None, None, None

def generate_chart(df, kind="bar"):
    try:
        plt.figure()
        if kind == "bar":
            df["Burnout"].value_counts().plot(kind="bar")
        else:
            df["Burnout"].value_counts().plot(kind="pie", autopct="%1.1f%%")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close()
        return img
    except:
        return None

def feature_importance(m, cols):
    try:
        plt.figure()
        plt.bar(cols, m.feature_importances_)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close()
        return img
    except:
        return None

def local_ai(q, df):
    if df is None:
        return "Upload dataset first."

    q = q.lower()

    try:
        if "average" in q:
            return f"Average burnout: {round(df['Burnout'].map({'Low':1,'Medium':2,'High':3}).mean(),2)}"

        if "high" in q:
            return f"{(df['Burnout']=='High').sum()} employees are high burnout."

        if "low" in q:
            return f"{(df['Burnout']=='Low').sum()} employees are low burnout."

        if "insight" in q:
            return "Higher workload and lower sleep correlate with higher burnout."

    except:
        return None

    return None

def gemma_ai(q, df=None):
    if q in ai_cache:
        return ai_cache[q]

    local = local_ai(q, df)
    if local:
        return local

    if not API_KEY:
        return "AI unavailable. Showing basic insights."

    try:
        res = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta/llama3-8b-instruct",
                "messages": [{"role":"user","content":q}],
                "max_tokens": 300
            },
            timeout=10
        )

        data = res.json()

        if "choices" in data:
            reply = data["choices"][0]["message"]["content"]
            ai_cache[q] = reply
            return reply

        return local if local else "AI response error."

    except:
        return local if local else "AI temporarily unavailable."

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI</title>
<style>
body{margin:0;font-family:sans-serif;background:#0b1120;color:white;display:flex}
.sidebar{width:220px;background:#020617;height:100vh;padding:20px}
.sidebar button{width:100%;margin:10px 0;padding:10px;background:#1f2937;color:white;border:none}
.main{flex:1;padding:20px}
.card{background:#111827;padding:20px;margin-bottom:20px;border-radius:10px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.kpi{padding:20px;text-align:center;border-radius:10px}
.high{background:#7f1d1d}.medium{background:#78350f}.low{background:#064e3b}.avg{background:#1e3a8a}
.hidden{display:none}
table{width:100%}
td,th{padding:10px;text-align:center}
tr.high-risk{background:#7f1d1d}
#chatbox{position:fixed;bottom:80px;right:20px;width:300px;height:400px;background:#111827;display:none;flex-direction:column}
#chat-body{flex:1;overflow:auto;padding:10px}
.msg{margin:5px;padding:8px;border-radius:6px}
.user{background:#2563eb}.ai{background:#374151}
</style>

<script>
function showTab(id){
document.getElementById("dashboard").classList.add("hidden");
document.getElementById("analytics").classList.add("hidden");
document.getElementById(id).classList.remove("hidden");
}

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
t.innerHTML="Analyzing...";
b.appendChild(t);
fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:m})})
.then(r=>r.json())
.then(d=>{t.innerHTML=d.reply})
.catch(()=>{t.innerHTML="Error"});
i.value="";
}
</script>
</head>

<body>

<div class="sidebar">
<h3>Burnout AI</h3>
<button onclick="showTab('dashboard')">Dashboard</button>
<button onclick="showTab('analytics')">Analytics</button>
</div>

<div class="main">

<div class="card">
<form method="POST" enctype="multipart/form-data">
<input type="file" name="file">
<button>Upload</button>
</form>
</div>

<div id="dashboard">

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
<tr class="{{ 'high-risk' if r['Burnout']=='High' else '' }}">
<td>{{r}}</td>
</tr>
{% endfor %}
</table>
</div>
{% endif %}

</div>

<div id="analytics" class="hidden">

{% if bar %}<div class="card"><img src="data:image/png;base64,{{bar}}"></div>{% endif %}
{% if pie %}<div class="card"><img src="data:image/png;base64,{{pie}}"></div>{% endif %}
{% if fi %}<div class="card"><img src="data:image/png;base64,{{fi}}"></div>{% endif %}

</div>

<a href="/download"><button>Download</button></a>

</div>

<div onclick="toggleChat()" style="position:fixed;bottom:20px;right:20px;background:#2563eb;padding:15px;border-radius:50%">Chat</div>

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
        file = request.files.get("file")
        if file:
            df = pd.read_csv(file)
            df, model, feature_cols, stats = auto_train(df)

            if df is not None:
                last_df = df
                table = df.to_dict(orient="records")
                bar = generate_chart(df, "bar")
                pie = generate_chart(df, "pie")
                fi = feature_importance(model, feature_cols)

    return render_template_string(HTML, table=table, stats=stats, bar=bar, pie=pie, fi=fi)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    return jsonify({"reply": gemma_ai(data.get("message",""), last_df)})

@app.route("/download")
def download():
    if last_df is not None:
        last_df.to_csv("report.csv", index=False)
        return send_file("report.csv", as_attachment=True)
    return "No data"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
