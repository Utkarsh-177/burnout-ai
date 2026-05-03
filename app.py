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
body{margin:0;font-family:system-ui;background:#0f172a;color:#e2e8f0}

.container{max-width:1100px;margin:auto;padding:30px}

h1{text-align:center;margin-bottom:10px}

.upload{
display:block;
max-width:600px;
margin:20px auto;
padding:50px;
border:2px dashed #3b82f6;
border-radius:12px;
text-align:center;
cursor:pointer;
transition:0.3s;
}
.upload:hover{transform:scale(1.03);background:#020617}

.switch-wrapper{display:flex;justify-content:center;margin:25px 0}

.switch{
position:relative;
width:280px;
height:45px;
background:#020617;
border-radius:30px;
display:flex;
align-items:center;
overflow:hidden;
}

.option{
flex:1;
text-align:center;
z-index:2;
cursor:pointer;
color:#94a3b8;
transition:0.3s;
}

.option.active{
color:white;
font-weight:600;
}

.slider{
position:absolute;
width:50%;
height:100%;
background:#3b82f6;
border-radius:30px;
transition:all 0.35s ease;
left:0;
}

.view-container{overflow:hidden}

.views{
display:flex;
width:200%;
transition:transform 0.45s cubic-bezier(0.4,0,0.2,1);
}

.screen{width:100%}

.stats{display:flex;justify-content:center;gap:20px;margin-bottom:20px}

.card{background:#020617;padding:15px;border-radius:10px;width:140px;text-align:center}

.table-box{
max-height:300px;
overflow:auto;
margin-top:20px;
border:1px solid #1e293b;
border-radius:10px
}

table{width:100%}
td,th{padding:8px;text-align:center}

button{
padding:10px 15px;
background:#3b82f6;
border:none;
border-radius:6px;
cursor:pointer;
transition:0.3s;
}
button:hover{transform:scale(1.05)}

#chat{
position:fixed;
bottom:20px;
right:20px;
background:#3b82f6;
padding:14px;
border-radius:50%;
cursor:pointer
}

#chatbox{
position:fixed;
bottom:80px;
right:20px;
width:320px;
height:420px;
background:#020617;
display:none;
flex-direction:column;
border-radius:10px;
border:1px solid #1e293b
}

#chat-body{flex:1;overflow:auto;padding:10px}
</style>

<script>
function switchView(index){
document.getElementById("views").style.transform=`translateX(-${index*50}%)`

let slider=document.getElementById("slider")
slider.style.left=index===0?"0%":"50%"

let options=document.querySelectorAll(".option")
options.forEach(o=>o.classList.remove("active"))
options[index].classList.add("active")
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
b.scrollTop=b.scrollHeight
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

<div class="switch-wrapper">
<div class="switch">
<div class="slider" id="slider"></div>
<div class="option active" onclick="switchView(0)">Burnout</div>
<div class="option" onclick="switchView(1)">Productivity</div>
</div>
</div>

<div class="view-container">
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
</div>

{% if table %}
<div class="table-box">
<table>
<tr>{% for k in table[0].keys() %}<th>{{k}}</th>{% endfor %}</tr>
{% for r in table[:30] %}
<tr>{% for v in r.values() %}<td>{{v}}</td>{% endfor %}</tr>
{% endfor %}
</table>
</div>
{% endif %}

{% if recommendations %}
<div style="margin-top:40px">
<h3>Recommendations</h3>
<ul>
{% for r in recommendations %}
<li>{{r}}</li>
{% endfor %}
</ul>
</div>
{% endif %}

{% if stats %}
<div style="display:flex;justify-content:space-between;margin-top:30px">
<div>
<a href="/download/burnout"><button>Download Burnout Report</button></a>
<a href="/download/productivity"><button>Download Productivity Report</button></a>
</div>
</div>
{% endif %}

</div>

<div id="chat" onclick="toggleChat()">Chat</div>

<div id="chatbox">
<div id="chat-body"></div>
<input id="chat_text" onkeydown="if(event.key==='Enter'){sendMessage()}">
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
