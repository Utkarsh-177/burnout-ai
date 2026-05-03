from flask import Flask, request, render_template_string, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import requests, os, io

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")
last_df = None


# ---------------- ML ----------------
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


def recommendations(stats):
    rec = []
    total = sum(stats.values())

    if stats["high"]/total > 0.4:
        rec.append("⚠ High burnout detected. Reduce workload immediately.")
    elif stats["medium"]/total > 0.4:
        rec.append("⚡ Moderate burnout. Improve work-life balance.")
    else:
        rec.append("✅ Burnout is under control.")

    rec.append("💤 Ensure proper sleep.")
    rec.append("🏃 Encourage physical activity.")
    rec.append("🤝 Maintain healthy environment.")

    return rec


# ---------------- AI ----------------
def ai_chat(q, df):
    if df is None:
        return "Upload dataset first."

    if not API_KEY:
        return f"Rows: {len(df)} | Columns: {len(df.columns)}"

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
                "messages":[{"role":"user","content":f"{summary}\n\nQ: {q}"}],
                "max_tokens":200
            }
        )

        data = res.json()
        return data.get("choices",[{}])[0].get("message",{}).get("content","Error")

    except:
        return "AI error"


# ---------------- HTML ----------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body{margin:0;background:#0f172a;color:white;font-family:system-ui}
.container{max-width:1100px;margin:auto;padding:30px}

.upload{padding:40px;border:2px dashed #3b82f6;text-align:center;border-radius:10px;cursor:pointer}

.switch{display:flex;margin:20px auto;width:260px;background:#020617;border-radius:30px;position:relative}
.option{flex:1;text-align:center;padding:10px;cursor:pointer}
.option.active{color:white;font-weight:bold}
.slider{position:absolute;width:50%;height:100%;background:#3b82f6;border-radius:30px;transition:0.3s}

.stats{display:flex;gap:20px;justify-content:center;margin:20px}
.card{background:#020617;padding:15px;border-radius:10px}

button{
background:#3b82f6;
border:none;
padding:10px 15px;
margin:5px;
border-radius:8px;
cursor:pointer;
transition:0.3s;
}
button:hover{transform:scale(1.1)}

#chat{position:fixed;bottom:20px;right:20px;background:#3b82f6;padding:15px;border-radius:50%;cursor:pointer}
#chatbox{position:fixed;bottom:80px;right:20px;width:300px;height:400px;background:#020617;display:none;flex-direction:column}
#chat-body{flex:1;overflow:auto}
</style>

<script>
function switchView(i){
document.getElementById("views").style.transform=`translateX(-${i*50}%)`
document.getElementById("slider").style.left=i==0?"0%":"50%"
}

function toggleChat(){
let c=document.getElementById("chatbox")
c.style.display=c.style.display=="flex"?"none":"flex"
}

function sendMessage(){
let i=document.getElementById("chat_text")
let b=document.getElementById("chat-body")

b.innerHTML+=`<div>${i.value}</div>`

fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:i.value})})
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
<div class="slider" id="slider"></div>
<div class="option active" onclick="switchView(0)">Burnout</div>
<div class="option" onclick="switchView(1)">Productivity</div>
</div>

<div id="views" style="display:flex;width:200%;transition:0.4s">

<div style="width:100%">
{% if stats %}
<div class="stats">
<div class="card">High {{stats.high}}</div>
<div class="card">Medium {{stats.medium}}</div>
<div class="card">Low {{stats.low}}</div>
</div>
<canvas id="c1"></canvas>
{% endif %}
</div>

<div style="width:100%">
{% if prod %}
<div class="stats">
<div class="card">High {{prod.high}}</div>
<div class="card">Medium {{prod.medium}}</div>
<div class="card">Low {{prod.low}}</div>
</div>
<canvas id="c2"></canvas>
{% endif %}
</div>

</div>

{% if recommendations %}
<h3>Recommendations</h3>
<ul>
{% for r in recommendations %}
<li>{{r}}</li>
{% endfor %}
</ul>
{% endif %}

<a href="/download/burnout"><button>Download Burnout</button></a>
<a href="/download/productivity"><button>Download Productivity</button></a>

</div>

<div id="chat" onclick="toggleChat()">💬</div>

<div id="chatbox">
<div id="chat-body"></div>
<input id="chat_text" onkeydown="if(event.key==='Enter'){sendMessage()}">
</div>

<script>
{% if stats %}
new Chart(document.getElementById('c1'),{
type:'bar',
data:{labels:['Low','Medium','High'],datasets:[{data:[{{stats.low}},{{stats.medium}},{{stats.high}}]}]}
});
{% endif %}

{% if prod %}
new Chart(document.getElementById('c2'),{
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

            rec=recommendations(stats)

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
