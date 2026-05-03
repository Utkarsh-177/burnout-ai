from flask import Flask, request, render_template_string, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import requests, os, io

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")
last_df = None


# ---------------- DATA PROCESSING ----------------
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
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)

    num = df.select_dtypes(include=np.number)

    if len(num.columns) < 2:
        df["Burnout"] = np.random.choice(["Low","Medium","High"], len(df))
    else:
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


# ---------------- SMART RECOMMENDATIONS ----------------
def recommendations(df, stats):
    rec = []
    total = sum(stats.values())

    if total == 0:
        return ["No data available"]

    high_ratio = stats["high"] / total
    med_ratio = stats["medium"] / total

    if high_ratio > 0.4:
        rec.append("High burnout detected. Immediate workload reduction recommended.")
    elif med_ratio > 0.4:
        rec.append("Moderate burnout detected. Monitor workload and balance tasks.")
    else:
        rec.append("Burnout levels are stable.")

    try:
        numeric = df.select_dtypes(include=np.number)
        if not numeric.empty:
            corr = numeric.corr().abs().unstack().sort_values(ascending=False)
            top = corr[corr < 1].head(1)
            if len(top) > 0:
                c1, c2 = top.index[0]
                rec.append(f"Strong relationship observed between {c1} and {c2}.")
    except:
        pass

    rec.append("Encourage proper sleep and regular breaks.")
    rec.append("Promote a healthy and balanced work environment.")

    return rec


# ---------------- AI CHAT ----------------
def ai_chat(q, df):
    if df is None:
        return "Upload dataset first."

    if not API_KEY:
        return f"Rows: {len(df)}, Columns: {len(df.columns)}"

    try:
        summary = df.describe().to_string()

        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta/llama-3.3-70b-instruct",
                "messages":[{"role":"user","content":f"{summary}\n\nQuestion: {q}"}],
                "max_tokens":200
            },
            timeout=20
        )

        data = response.json()

        if "choices" not in data:
            return str(data)

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"AI Error: {str(e)}"


# ---------------- HTML ----------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body{margin:0;font-family:system-ui;background:#0f172a;color:#e2e8f0}
.container{max-width:1200px;margin:auto;padding:30px}

h1{text-align:center}
.subtitle{text-align:center;color:#94a3b8;margin-bottom:30px}

.upload{
display:block;
max-width:600px;
margin:0 auto 30px;
padding:40px;
border:2px dashed #3b82f6;
border-radius:12px;
text-align:center;
cursor:pointer;
}

.switch-wrapper{display:flex;justify-content:center;margin-bottom:30px}
.switch{position:relative;width:260px;height:45px;background:#020617;border-radius:30px;display:flex}
.option{flex:1;text-align:center;line-height:45px;cursor:pointer;color:#94a3b8}
.option.active{color:white;font-weight:600}
.slider{position:absolute;width:50%;height:100%;background:#3b82f6;border-radius:30px;transition:0.3s}

.views{display:flex;width:200%;transition:0.4s}
.screen{width:100%}

.stats{display:flex;justify-content:center;gap:20px;margin-bottom:20px}
.card{background:#020617;padding:15px 25px;border-radius:10px;text-align:center}

.section{margin-top:40px;background:#020617;padding:20px;border-radius:12px}

.downloads{margin-top:20px}
button{background:#3b82f6;border:none;padding:10px 15px;border-radius:8px;cursor:pointer;margin-right:10px}

#chat{position:fixed;bottom:20px;right:20px;background:#3b82f6;padding:14px;border-radius:50%;cursor:pointer}
#chatbox{position:fixed;bottom:80px;right:20px;width:320px;height:420px;background:#020617;display:none;flex-direction:column}
#chat-body{flex:1;overflow:auto;padding:10px}
.msg{margin:6px;padding:8px;border-radius:6px}
.user{background:#3b82f6}
.ai{background:#1e293b}
</style>

<script>
function switchView(index){
document.getElementById("views").style.transform=`translateX(-${index*50}%)`
document.getElementById("slider").style.left=index===0?"0%":"50%"
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
<p class="subtitle">AI-powered Burnout & Productivity Insights</p>

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

{% if recommendations %}
<div class="section">
<h3>Recommendations</h3>
<ul>
{% for r in recommendations %}
<li>{{r}}</li>
{% endfor %}
</ul>

<div class="downloads">
<a href="/download/burnout"><button>Download Burnout</button></a>
<a href="/download/productivity"><button>Download Productivity</button></a>
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


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET","POST"])
def home():
    global last_df
    stats=prod=rec=None

    if request.method=="POST":
        file=request.files.get("file")
        if file:
            df=pd.read_csv(file,on_bad_lines='skip')
            df,stats=auto_train(df)
            df=add_productivity(df)
            last_df=df

            prod={
                "high":int((df["Productivity"]=="High Productivity").sum()),
                "medium":int((df["Productivity"]=="Moderate Productivity").sum()),
                "low":int((df["Productivity"]=="Low Productivity").sum())
            }

            rec=recommendations(df, stats)

    return render_template_string(HTML,stats=stats,prod=prod,recommendations=rec)


@app.route("/chat",methods=["POST"])
def chat():
    return jsonify({"reply":ai_chat(request.get_json()["message"],last_df)})


@app.route("/download/<type>")
def download(type):
    if last_df is None:
        return "No data"

    df = last_df.copy()

    if type=="burnout":
        df=df[["Burnout"]]
    else:
        df=df[["Productivity"]]

    buf=io.BytesIO()
    df.to_csv(buf,index=False)
    buf.seek(0)

    return send_file(buf,as_attachment=True,download_name=f"{type}.csv")


if __name__=="__main__":
    app.run(debug=True)
