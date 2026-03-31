from flask import Flask, request, render_template_string, send_file, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests, os

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")

last_df = None
model = None
feature_cols = []
ai_cache = {}

def auto_train(df):
    df.columns = df.columns.str.lower()
    num = df.select_dtypes(include=['int64','float64'])

    if len(num.columns) < 2:
        return None, None, None, None

    possible_targets = [c for c in df.columns if any(k in c for k in ["burnout","stress","target","label","output"])]

    if possible_targets:
        target = possible_targets[0]
        X = num.drop(columns=[target], errors='ignore')
        y = df[target]

        if y.nunique() > 10:
            y = pd.qcut(y, q=3, labels=[0,1,2], duplicates='drop')

        X = X.fillna(X.mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        m = RandomForestClassifier(n_estimators=200, random_state=42)
        m.fit(X_train, y_train)

        acc = round(accuracy_score(y_test, m.predict(X_test))*100,2)
        preds = m.predict(X)

    else:
        X = num.fillna(num.mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        preds = km.fit_predict(X_scaled)

        acc = 0

    df["Burnout"] = [["Low","Medium","High"][int(i)] for i in preds]

    stats = {
        "avg": round(df["Burnout"].map({"Low":1,"Medium":2,"High":3}).mean(),2),
        "high": int((df["Burnout"]=="High").sum()),
        "medium": int((df["Burnout"]=="Medium").sum()),
        "low": int((df["Burnout"]=="Low").sum()),
        "rows": len(df),
        "cols": len(df.columns),
        "accuracy": acc
    }

    return df, None, list(num.columns), stats

def smart_ai(q, df):
    if df is None:
        return "Upload dataset first."

    q = q.lower()

    try:
        if "summary" in q:
            return df.describe().to_string()
        if "columns" in q:
            return ", ".join(df.columns)
        if "rows" in q:
            return f"{len(df)} rows"
        if "missing" in q:
            return df.isnull().sum().to_string()
        if "correlation" in q:
            return df.corr(numeric_only=True).to_string()
        if "average" in q:
            return df.mean(numeric_only=True).to_string()
        if "distribution" in q:
            return df["Burnout"].value_counts().to_string()
        if "high" in q:
            return f"{(df['Burnout']=='High').sum()} high burnout employees"
        if "important" in q and model is not None:
            imp = pd.Series(model.feature_importances_, index=df.select_dtypes(include=['int64','float64']).columns[:-1])
            return imp.sort_values(ascending=False).to_string()
        if "recommend" in q:
            return "Reduce workload, increase breaks, monitor high-risk employees."
    except:
        return None

    return None

def gemma_ai(q, df):
    if q in ai_cache:
        return ai_cache[q]

    local = smart_ai(q, df)
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

        return local if local else "Try again."

    except:
        return local if local else "Try again."

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body{
margin:0;
font-family:system-ui;
background:#0b0f19;
color:#e5e7eb;
display:flex;
animation:fadeIn 0.6s ease-in;
}

@keyframes fadeIn{
from{opacity:0;transform:translateY(8px)}
to{opacity:1;transform:translateY(0)}
}

.sidebar{
width:250px;
background:#020617;
padding:25px;
height:100vh;
position:fixed;
border-right:1px solid #111827;
}

.sidebar h2{
font-size:18px;
margin-bottom:20px;
color:#9ca3af;
}

.sidebar button{
width:100%;
margin:10px 0;
padding:12px;
background:#111827;
border:none;
color:#e5e7eb;
border-radius:8px;
cursor:pointer;
transition:all 0.2s ease;
}

.sidebar button:hover{
background:#1f2937;
transform:translateY(-2px);
}

.main{
margin-left:270px;
padding:30px;
width:100%;
}

.grid{
display:grid;
grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
gap:20px;
margin-bottom:25px;
}

.card{
background:#111827;
padding:20px;
border-radius:12px;
text-align:center;
box-shadow:0 4px 20px rgba(0,0,0,0.25);
transition:all 0.25s ease;
}

.card:hover{
transform:translateY(-4px);
}

.layout{
display:grid;
grid-template-columns:2.5fr 1fr;
gap:20px;
}

input[type="file"]{
padding:10px;
background:#020617;
border:none;
color:white;
border-radius:6px;
}

button{
padding:10px 15px;
border:none;
border-radius:6px;
cursor:pointer;
}

table{
width:100%;
border-collapse:collapse;
margin-top:15px;
font-size:14px;
}

td,th{
padding:10px;
border-bottom:1px solid #1f2937;
text-align:center;
}

tr:hover{
background:#1f2937;
}

.high{background:#3f1d1d}
.medium{background:#3f2d1d}

#chat{
position:fixed;
bottom:20px;
right:20px;
background:#2563eb;
padding:14px;
border-radius:50%;
cursor:pointer;
transition:0.2s;
}

#chat:hover{
transform:scale(1.05);
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
border:1px solid #1f2937;
}

#chat-body{
flex:1;
overflow:auto;
padding:10px;
}

.msg{
margin:6px;
padding:8px;
border-radius:6px;
font-size:13px;
}

.user{background:#2563eb}
.ai{background:#1f2937}

.download-btn{
margin-top:15px;
padding:12px;
background:#16a34a;
border:none;
color:white;
width:100%;
border-radius:8px;
transition:0.2s;
}

.download-btn:hover{
background:#15803d;
}
</style>

<script>
function toggleChat(){
document.getElementById("chatbox").style.display="flex";
}

function sendMessage(msg=null){
let i=document.getElementById("chat_text");
let m = msg || i.value.trim();
if(!m)return;

document.getElementById("chatbox").style.display="flex";

let b=document.getElementById("chat-body");
b.innerHTML+=`<div class='msg user'>${m}</div>`;

let t=document.createElement("div");
t.className="msg ai";
t.innerHTML="Processing...";
b.appendChild(t);

fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:m})})
.then(r=>r.json()).then(d=>{
t.innerHTML=d.reply;
b.scrollTop=b.scrollHeight;
});

if(i) i.value="";
}
</script>
</head>

<body>

<div class="sidebar">
<h2>Burnout AI</h2>
<button onclick="sendMessage('summary')">Summary</button>
<button onclick="sendMessage('correlation')">Correlation</button>
<button onclick="sendMessage('important')">Important Features</button>
<button onclick="sendMessage('recommend')">Recommendation</button>

<a href="/download">
<button class="download-btn">Download Report</button>
</a>
</div>

<div class="main">

<div class="grid">
<div class="card">Average<br><strong>{{stats.avg if stats else '-'}}</strong></div>
<div class="card">High<br><strong>{{stats.high if stats else '-'}}</strong></div>
<div class="card">Medium<br><strong>{{stats.medium if stats else '-'}}</strong></div>
<div class="card">Low<br><strong>{{stats.low if stats else '-'}}</strong></div>
<div class="card">Rows<br><strong>{{stats.rows if stats else '-'}}</strong></div>
<div class="card">Accuracy<br><strong>{{stats.accuracy if stats else '-'}}%</strong></div>
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
<p>Adaptive AI automatically switches between supervised and unsupervised learning based on dataset structure.</p>
</div>

</div>

</div>

<div id="chat" onclick="toggleChat()">Chat</div>

<div id="chatbox">
<div id="chat-body"></div>
<input id="chat_text" placeholder="Ask about dataset..." onkeydown="if(event.key==='Enter'){sendMessage()}">
</div>

</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():
    global last_df, model
    table=None; stats=None

    if request.method=="POST":
        file = request.files.get("file")
        if file:
            df = pd.read_csv(file)
            df, model, feature_cols, stats = auto_train(df)
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
        last_df.to_csv("burnout_report.csv", index=False)
        return send_file("burnout_report.csv", as_attachment=True)
    return "No data"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
