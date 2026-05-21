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


def recommendations(stats):
    rec = []
    total = sum(stats.values())

    if stats["high"]/total > 0.4:
        rec.append("Critical burnout detected. Immediate intervention required.")
    elif stats["medium"]/total > 0.4:
        rec.append("Moderate burnout observed. Improve balance and reduce overload.")
    else:
        rec.append("Burnout levels are stable.")

    rec.append("Maintain consistent sleep schedule.")
    rec.append("Encourage breaks and physical activity.")
    rec.append("Use productivity tracking for performance improvement.")

    return rec


# ---------------- AI CHAT ----------------
def ai_chat(q, df):
    if df is None:
        return "Upload dataset first."

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
                "messages": [{"role": "user", "content": f"{summary}\n\nQuestion: {q}"}],
                "max_tokens": 200
            },
            timeout=20
        )

        data = res.json()

        if "choices" not in data:
            return "AI response error."

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------- HTML (PRO UI) ----------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
}

body{
font-family:system-ui;
background:#020617;
color:#e2e8f0;
overflow-x:hidden;
}

body::before{
content:"";
position:fixed;
width:700px;
height:700px;
background:radial-gradient(circle,#2563eb33,transparent 70%);
top:-250px;
right:-250px;
z-index:-1;
animation:floatGlow 8s infinite alternate ease-in-out;
}

@keyframes floatGlow{
from{transform:translateY(0)}
to{transform:translateY(40px)}
}

.container{
max-width:1300px;
margin:auto;
padding:40px 25px;
animation:fadeIn 1s ease;
}

@keyframes fadeIn{
from{
opacity:0;
transform:translateY(20px);
}
to{
opacity:1;
transform:translateY(0);
}
}

h1{
text-align:center;
font-size:52px;
font-weight:800;
margin-bottom:10px;
}

.subtitle{
text-align:center;
color:#94a3b8;
margin-bottom:40px;
font-size:17px;
}

/* FLOATING BAR */

.floating-bar{
position:fixed;
top:18px;
left:50%;
transform:translateX(-50%);
display:flex;
gap:30px;
padding:14px 30px;
background:rgba(15,23,42,0.75);
backdrop-filter:blur(14px);
border:1px solid #1e293b;
border-radius:20px;
z-index:999;
box-shadow:0 10px 30px rgba(0,0,0,0.3);
}

.floating-bar div{
text-align:center;
}

.floating-bar h3{
font-size:22px;
}

.floating-bar p{
color:#94a3b8;
font-size:13px;
}

.upload{
display:flex;
justify-content:center;
align-items:center;
flex-direction:column;
gap:12px;
max-width:720px;
margin:0 auto 40px;
padding:65px;
border:2px dashed #2563eb;
border-radius:28px;
background:rgba(15,23,42,0.85);
backdrop-filter:blur(10px);
cursor:pointer;
transition:0.45s;
position:relative;
overflow:hidden;
}

.upload::before{
content:"";
position:absolute;
width:120%;
height:120%;
background:linear-gradient(
120deg,
transparent,
rgba(255,255,255,0.08),
transparent
);
transform:translateX(-100%);
transition:0.8s;
}

.upload:hover::before{
transform:translateX(100%);
}

.upload:hover{
transform:translateY(-6px) scale(1.01);
box-shadow:0 0 40px #2563eb55;
}

.switch-wrapper{
display:flex;
justify-content:center;
margin:45px 0;
}

.switch{
position:relative;
width:430px;
height:62px;
background:#0f172a;
border-radius:50px;
padding:6px;
display:flex;
align-items:center;
overflow:hidden;
box-shadow:
0 10px 30px rgba(0,0,0,0.45),
inset 0 0 12px rgba(255,255,255,0.03);
}

.slider{
position:absolute;
width:50%;
height:50px;
left:6px;
background:linear-gradient(135deg,#2563eb,#3b82f6);
border-radius:40px;
transition:0.45s cubic-bezier(.77,0,.18,1);
box-shadow:0 12px 30px rgba(37,99,235,0.4);
}

.option{
flex:1;
z-index:2;
text-align:center;
cursor:pointer;
font-weight:600;
font-size:15px;
color:#94a3b8;
transition:0.3s;
user-select:none;
}

.option.active{
color:white;
}

.view-container{
overflow:hidden;
width:100%;
}

.views{
display:flex;
width:200%;
transition:transform 0.6s cubic-bezier(.77,0,.18,1);
}

.screen{
width:100%;
padding:10px;
}

.stats{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
gap:25px;
margin-bottom:35px;
}

.card{
background:linear-gradient(145deg,#0f172a,#111827);
padding:35px;
border-radius:25px;
text-align:center;
border:1px solid #1e293b;
transition:0.35s;
}

.card:hover{
transform:translateY(-8px);
box-shadow:0 20px 40px rgba(0,0,0,0.35);
}

.card h2{
font-size:42px;
margin-bottom:10px;
}

.chart-box{
background:#0f172a;
padding:30px;
border-radius:25px;
border:1px solid #1e293b;
margin-bottom:40px;
box-shadow:0 15px 40px rgba(0,0,0,0.25);
}

.table-section{
margin-top:45px;
}

.table-title{
font-size:26px;
font-weight:700;
margin-bottom:20px;
}

.table-box{
max-height:430px;
overflow:auto;
border-radius:20px;
border:1px solid #1e293b;
background:#0f172a;
}

.table-box::-webkit-scrollbar{
width:10px;
height:10px;
}

.table-box::-webkit-scrollbar-thumb{
background:#2563eb;
border-radius:20px;
}

table{
width:100%;
border-collapse:collapse;
}

th{
position:sticky;
top:0;
background:#111827;
z-index:2;
}

td,th{
padding:14px;
text-align:center;
border-bottom:1px solid #1e293b;
}

tr{
transition:0.2s;
}

tr:hover{
background:#172554;
}

/* INSIGHTS */

.insight-panel{
margin-top:60px;
}

.insight-header h2{
font-size:34px;
margin-bottom:10px;
}

.insight-header p{
color:#94a3b8;
margin-bottom:25px;
}

.insight-grid{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
gap:22px;
}

.insight-card{
background:#0f172a;
padding:28px;
border-radius:22px;
border:1px solid #1e293b;
transition:0.35s;
}

.insight-card:hover{
transform:translateY(-5px);
border-color:#2563eb;
box-shadow:0 20px 40px rgba(37,99,235,0.15);
}

.insight-card h3{
margin-bottom:14px;
font-size:22px;
}

.recommend-section{
margin-top:65px;
}

.recommend-grid{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
gap:22px;
margin-top:25px;
}

.recommend-card{
background:#0f172a;
padding:25px;
border-radius:22px;
border:1px solid #1e293b;
transition:0.35s;
line-height:1.7;
}

.recommend-card:hover{
transform:translateY(-5px);
border-color:#2563eb;
}

.downloads{
display:flex;
gap:20px;
margin-top:50px;
flex-wrap:wrap;
}

.download-btn{
padding:15px 24px;
border:none;
border-radius:14px;
background:linear-gradient(135deg,#2563eb,#3b82f6);
color:white;
font-weight:600;
cursor:pointer;
transition:0.35s;
font-size:15px;
box-shadow:0 12px 30px rgba(37,99,235,0.35);
}

.download-btn:hover{
transform:translateY(-4px) scale(1.02);
box-shadow:0 20px 40px rgba(37,99,235,0.5);
}

#chat{
position:fixed;
bottom:25px;
right:25px;
width:68px;
height:68px;
background:linear-gradient(135deg,#2563eb,#3b82f6);
border-radius:50%;
display:flex;
justify-content:center;
align-items:center;
font-weight:700;
cursor:pointer;
transition:0.35s;
box-shadow:0 18px 40px rgba(37,99,235,0.45);
z-index:999;
}

#chat:hover{
transform:scale(1.08);
}

#chatbox{
position:fixed;
bottom:105px;
right:25px;
width:360px;
height:510px;
background:#0f172a;
border-radius:24px;
display:none;
flex-direction:column;
border:1px solid #1e293b;
overflow:hidden;
box-shadow:0 25px 60px rgba(0,0,0,0.45);
z-index:999;
animation:chatOpen 0.3s ease;
}

@keyframes chatOpen{
from{
opacity:0;
transform:translateY(20px) scale(0.95);
}
to{
opacity:1;
transform:translateY(0) scale(1);
}
}

#chat-body{
flex:1;
overflow:auto;
padding:18px;
display:flex;
flex-direction:column;
gap:12px;
}

.msg{
padding:12px 14px;
border-radius:14px;
max-width:85%;
line-height:1.5;
font-size:14px;
animation:fadeIn 0.3s ease;
}

.user{
background:#2563eb;
align-self:flex-end;
}

.ai{
background:#1e293b;
align-self:flex-start;
}

.chat-input{
display:flex;
border-top:1px solid #1e293b;
}

.chat-input input{
flex:1;
padding:15px;
background:#020617;
border:none;
outline:none;
color:white;
font-size:14px;
}

.chat-input button{
width:80px;
border:none;
background:#2563eb;
color:white;
font-weight:600;
cursor:pointer;
transition:0.3s;
}

.chat-input button:hover{
background:#3b82f6;
}

/* LOADER */

#loader{
position:fixed;
inset:0;
background:#020617;
display:none;
justify-content:center;
align-items:center;
z-index:5000;
}

.loader-box{
text-align:center;
}

.loader-circle{
width:90px;
height:90px;
border:6px solid #1e293b;
border-top:6px solid #2563eb;
border-radius:50%;
margin:auto auto 25px;
animation:spin 1s linear infinite;
}

@keyframes spin{
100%{
transform:rotate(360deg);
}
}

</style>

<script>

function switchView(index){

document.getElementById("views").style.transform =
`translateX(-${index*50}%)`

let slider=document.getElementById("slider")

slider.style.left = index===0 ? "6px" : "50%"

let options=document.querySelectorAll(".option")

options.forEach(o=>o.classList.remove("active"))

options[index].classList.add("active")
}

function toggleChat(){

let c=document.getElementById("chatbox")

c.style.display = c.style.display==="flex" ? "none" : "flex"
}

function showLoader(){
document.getElementById("loader").style.display="flex"
}

function sendMessage(){

let i=document.getElementById("chat_text")

let m=i.value.trim()

if(!m)return

let b=document.getElementById("chat-body")

b.innerHTML += `
<div class='msg user'>${m}</div>
`

let t=document.createElement("div")

t.className="msg ai"

t.innerHTML="Analyzing dataset..."

b.appendChild(t)

b.scrollTop=b.scrollHeight

fetch("/chat",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
message:m
})
})
.then(r=>r.json())
.then(d=>{
t.innerHTML=d.reply
b.scrollTop=b.scrollHeight
})

i.value=""
}

</script>
</head>

<body>

{% if stats %}
<div class="floating-bar">

<div>
<h3>{{stats.high}}</h3>
<p>High</p>
</div>

<div>
<h3>{{stats.medium}}</h3>
<p>Medium</p>
</div>

<div>
<h3>{{stats.low}}</h3>
<p>Low</p>
</div>

</div>
{% endif %}

<div class="container">

<h1>Burnout AI</h1>

<p class="subtitle">
Advanced AI-powered burnout detection, productivity analytics, and intelligent workforce insights
</p>

<form method="POST" enctype="multipart/form-data">

<label class="upload">

<h2>Upload Dataset</h2>

<p style="color:#94a3b8">
Drag and analyze workforce productivity & burnout datasets
</p>

<input type="file"
name="file"
hidden
onchange="showLoader(); this.form.submit()">

</label>

</form>

<div class="switch-wrapper">

<div class="switch">

<div class="slider" id="slider"></div>

<div class="option active" onclick="switchView(0)">
Burnout Analytics
</div>

<div class="option" onclick="switchView(1)">
Productivity Insights
</div>

</div>

</div>

<div class="view-container">

<div class="views" id="views">

<div class="screen">

{% if stats %}

<div class="stats">

<div class="card">
<h2>{{stats.high}}</h2>
<p>High Burnout</p>
</div>

<div class="card">
<h2>{{stats.medium}}</h2>
<p>Medium Burnout</p>
</div>

<div class="card">
<h2>{{stats.low}}</h2>
<p>Low Burnout</p>
</div>

</div>

<div class="chart-box">
<canvas id="chart1"></canvas>
</div>

{% endif %}

</div>

<div class="screen">

{% if prod %}

<div class="stats">

<div class="card">
<h2>{{prod.high}}</h2>
<p>High Productivity</p>
</div>

<div class="card">
<h2>{{prod.medium}}</h2>
<p>Moderate Productivity</p>
</div>

<div class="card">
<h2>{{prod.low}}</h2>
<p>Low Productivity</p>
</div>

</div>

<div class="chart-box">
<canvas id="chart2"></canvas>
</div>

{% endif %}

</div>

</div>

</div>

{% if table %}

<div class="table-section">

<div class="table-title">
Dataset Preview
</div>

<div class="table-box">

<table>

<tr>

{% for k in table[0].keys() %}
<th>{{k}}</th>
{% endfor %}

</tr>

{% for r in table[:50] %}

<tr>

{% for v in r.values() %}
<td>{{v}}</td>
{% endfor %}

</tr>

{% endfor %}

</table>

</div>

</div>

{% endif %}

{% if stats %}

<div class="insight-panel">

<div class="insight-header">
<h2>AI Insights</h2>
<p>Live analytical observations generated from uploaded dataset</p>
</div>

<div class="insight-grid">

<div class="insight-card">
<h3>Burnout Risk</h3>
<p>
{% if stats.high > stats.medium and stats.high > stats.low %}
High burnout ratio detected across workforce groups.
{% elif stats.medium > stats.low %}
Moderate burnout trend observed.
{% else %}
Overall burnout appears controlled and stable.
{% endif %}
</p>
</div>

<div class="insight-card">
<h3>Productivity Stability</h3>
<p>
{% if prod.high > prod.low %}
Productivity levels remain healthy for most records.
{% else %}
Low productivity clusters are increasing rapidly.
{% endif %}
</p>
</div>

<div class="insight-card">
<h3>AI Recommendation</h3>
<p>
AI suggests balancing workloads, improving work-life structure,
and monitoring high-risk employee segments continuously.
</p>
</div>

</div>

</div>

{% endif %}

{% if recommendations %}

<div class="recommend-section">

<h2 style="margin-bottom:10px">
AI Recommendations
</h2>

<p style="color:#94a3b8">
Strategic insights generated from burnout and productivity patterns
</p>

<div class="recommend-grid">

{% for r in recommendations %}

<div class="recommend-card">
{{r}}
</div>

{% endfor %}

</div>

</div>

{% endif %}

{% if stats %}

<div class="downloads">

<a href="/download/burnout">

<button class="download-btn">
Download Burnout Report
</button>

</a>

<a href="/download/productivity">

<button class="download-btn">
Download Productivity Report
</button>

</a>

</div>

{% endif %}

</div>

<div id="chat" onclick="toggleChat()">
AI
</div>

<div id="chatbox">

<div id="chat-body"></div>

<div class="chat-input">

<input id="chat_text"
placeholder="Ask AI about the dataset..."
onkeydown="if(event.key==='Enter'){sendMessage()}">

<button onclick="sendMessage()">
Send
</button>

</div>

</div>

<div id="loader">

<div class="loader-box">

<div class="loader-circle"></div>

<h2>Analyzing Dataset...</h2>

<p>AI engine is processing burnout and productivity patterns</p>

</div>

</div>

<script>

{% if stats %}

new Chart(document.getElementById('chart1'),{

type:'bar',

data:{
labels:['Low','Medium','High'],
datasets:[{
data:[
{{stats.low}},
{{stats.medium}},
{{stats.high}}
],
borderRadius:14,
barThickness:60
}]
},

options:{
responsive:true,
animation:{
duration:2000,
easing:'easeOutQuart'
},
plugins:{
legend:{display:false}
},
scales:{
y:{
grid:{color:'#1e293b'},
ticks:{color:'#94a3b8'}
},
x:{
grid:{display:false},
ticks:{color:'#94a3b8'}
}
}
}

});

{% endif %}

{% if prod %}

new Chart(document.getElementById('chart2'),{

type:'line',

data:{
labels:['Low','Medium','High'],
datasets:[{
data:[
{{prod.low}},
{{prod.medium}},
{{prod.high}}
],
tension:0.4,
fill:true
}]
},

options:{
responsive:true,
animation:{
duration:2200
},
plugins:{
legend:{display:false}
},
scales:{
y:{
grid:{color:'#1e293b'},
ticks:{color:'#94a3b8'}
},
x:{
grid:{display:false},
ticks:{color:'#94a3b8'}
}
}
}

});

{% endif %}

</script>

</body>
</html>
"""


# ---------------- ROUTES ----------------
@app.route("/",methods=["GET","POST"])
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

            table=df.to_dict(orient="records")

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


@app.route("/download/<type>")
def download(type):
    if last_df is None:
        return "No data"

    output = io.StringIO()

    if type=="burnout":
        last_df[["Burnout"]].to_csv(output,index=False)
    else:
        last_df[["Productivity"]].to_csv(output,index=False)

    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()),download_name=f"{type}.csv",as_attachment=True)


if __name__=="__main__":
    app.run(debug=True)
