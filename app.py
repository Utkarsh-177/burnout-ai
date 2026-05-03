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
        df["Burnout"] = np.random.choice(["Low","Medium","High"], len(df))
        stats = {"high":0,"medium":0,"low":0}
        return df, stats

    scaler = StandardScaler()
    X = scaler.fit_transform(num)

    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    preds = km.fit_predict(X)

    df["Burnout"] = ["Low" if i==0 else "Medium" if i==1 else "High" for i in preds]

    stats = {
        "high": int((df["Burnout"]=="High").sum()),
        "medium": int((df["Burnout"]=="Medium").sum()),
        "low": int((df["Burnout"]=="Low").sum())
    }

    return df, stats


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

    if stats["high"]/total > 0.4:
        rec.append("High burnout detected. Reduce workload.")
    elif stats["medium"]/total > 0.4:
        rec.append("Moderate burnout detected. Maintain balance.")
    else:
        rec.append("Burnout levels are healthy.")

    rec.append("Maintain sleep schedule.")
    rec.append("Take breaks and stay active.")
    return rec


# ---------------- AI CHAT ----------------
def ai_chat(q, df):
    if df is None:
        return "Upload dataset first."

    if not API_KEY:
        return "API key missing."

    try:
        summary = df.describe(include='all').to_string()

        prompt = f"""
Dataset summary:
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
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 300
            },
            timeout=30
        )

        data = res.json()

        return data.get("choices",[{}])[0].get("message",{}).get("content","No response")

    except Exception as e:
        return f"API Error: {str(e)}"


# ---------------- DOWNLOAD ----------------
@app.route("/download/burnout")
def download_burnout():
    global last_df
    if last_df is None:
        return "No data"

    buffer = io.StringIO()
    last_df.to_csv(buffer, index=False)

    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        download_name="burnout_report.csv",
        as_attachment=True
    )


@app.route("/download/productivity")
def download_productivity():
    global last_df
    if last_df is None:
        return "No data"

    buffer = io.StringIO()
    last_df.to_csv(buffer, index=False)

    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        download_name="productivity_report.csv",
        as_attachment=True
    )


# ---------------- HTML ----------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body{margin:0;font-family:system-ui;background:#0f172a;color:#e2e8f0}

.container{max-width:1100px;margin:auto;padding:30px}

.upload{
display:block;max-width:600px;margin:0 auto 30px;padding:50px;
border:2px dashed #3b82f6;border-radius:12px;text-align:center;
cursor:pointer;transition:0.3s;
}
.upload:hover{transform:scale(1.02);background:#020617}

.option,#chat,a{transition:0.25s}
.option:hover,#chat:hover,a:hover{transform:scale(1.05);background:#020617}
.option:active,#chat:active,a:active{transform:scale(0.95)}

.switch{display:flex;justify-content:center;margin-bottom:20px}
.option{padding:10px 20px;cursor:pointer}
.option.active{color:#3b82f6;font-weight:bold}

.table-box{max-height:300px;overflow:auto;border:1px solid #1e293b}

#chat{position:fixed;bottom:20px;right:20px;background:#3b82f6;padding:12px;border-radius:50%}

#chatbox{
position:fixed;bottom:80px;right:20px;width:300px;height:400px;
background:#020617;display:none;flex-direction:column
}
</style>

<script>
function switchView(i){
document.getElementById("views").style.transform=`translateX(-${i*100}%)`
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
<div class="option" onclick="switchView(0)">Burnout</div>
<div class="option" onclick="switchView(1)">Productivity</div>
</div>

<div id="views" style="display:flex;width:200%;transition:0.4s">

<div style="width:100%">
{% if stats %}
<p>High: {{stats.high}}</p>
<p>Medium: {{stats.medium}}</p>
<p>Low: {{stats.low}}</p>
<canvas id="chart1"></canvas>
{% endif %}
</div>

<div style="width:100%">
{% if prod %}
<p>High: {{prod.high}}</p>
<p>Medium: {{prod.medium}}</p>
<p>Low: {{prod.low}}</p>
<canvas id="chart2"></canvas>
{% endif %}
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
<ul>
{% for r in recommendations %}
<li>{{r}}</li>
{% endfor %}
</ul>
{% endif %}

<a href="/download/burnout">Download Burnout Report</a><br>
<a href="/download/productivity">Download Productivity Report</a>

</div>

<div id="chat" onclick="toggleChat()">💬</div>

<div id="chatbox">
<div id="chat-body"></div>
<input id="chat_text" onkeydown="if(event.key==='Enter'){sendMessage()}">
</div>

</body>
</html>
"""


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET","POST"])
def home():
    global last_df
    stats=None
    table=None
    prod=None
    rec=None

    if request.method=="POST":
        file=request.files.get("file")
        if file:
            df=pd.read_csv(file,on_bad_lines="skip")
            df,stats=auto_train(df)
            df=add_productivity(df)
            last_df=df

            table=df.to_dict("records")

            prod={
                "high":int((df["Productivity"]=="High Productivity").sum()),
                "medium":int((df["Productivity"]=="Moderate Productivity").sum()),
                "low":int((df["Productivity"]=="Low Productivity").sum())
            }

            rec=recommendations(stats)

    return render_template_string(HTML,stats=stats,table=table,prod=prod,recommendations=rec)


@app.route("/chat",methods=["POST"])
def chat():
    return jsonify({"reply":ai_chat(request.get_json()["message"],last_df)})


if __name__=="__main__":
    app.run(debug=True)
