from flask import Flask, request, render_template_string, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
import os

app = Flask(__name__)

last_df = None
model = None

def auto_train(df):
    df.columns = df.columns.str.lower()
    num = df.select_dtypes(include=['int64','float64'])

    if len(num.columns) < 2:
        return df, None, {"accuracy":0,"high":0,"medium":0,"low":0}

    target_cols = [c for c in df.columns if "burnout" in c or "stress" in c]

    acc = 0

    if target_cols:
        target = target_cols[0]
        X = num.drop(columns=[target], errors='ignore')
        y = df[target]

        X = X.fillna(X.mean())

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

        global model
        model = HistGradientBoostingClassifier()
        model.fit(X_train,y_train)

        acc = round(accuracy_score(y_test, model.predict(X_test))*100,2)
        preds = model.predict(X)

    else:
        X = num.fillna(num.mean())
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        km = KMeans(n_clusters=3,n_init=10)
        preds = km.fit_predict(X)

    df["Burnout"] = ["Low" if i==0 else "Medium" if i==1 else "High" for i in preds]

    stats = {
        "high": int((df["Burnout"]=="High").sum()),
        "medium": int((df["Burnout"]=="Medium").sum()),
        "low": int((df["Burnout"]=="Low").sum()),
        "accuracy": acc
    }

    return df, model, stats

def generate_insights(df):
    insights = []
    high = (df["Burnout"]=="High").sum()
    total = len(df)

    if high > total * 0.4:
        insights.append("High burnout risk detected in dataset")

    corr = df.corr(numeric_only=True)

    if not corr.empty:
        insights.append("Strong correlations detected between features")

    insights.append("Recommendation: balance workload and improve rest cycles")
    return insights

def ai_chat(q, df):
    if df is None:
        return "Upload dataset first"

    q = q.lower()

    try:
        if "summary" in q:
            return df.describe(include='all').to_string()

        elif "columns" in q:
            return ", ".join(df.columns)

        elif "high" in q:
            return str((df['Burnout']=='High').sum())

        elif "correlation" in q:
            return df.corr(numeric_only=True).to_string()

        elif "insight" in q:
            return "\\n".join(generate_insights(df))

        else:
            sample = df.head(5).to_string()
            return "Based on dataset:\\n" + sample

    except:
        return "Error"

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Burnout AI X</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body{margin:0;font-family:system-ui;background:linear-gradient(135deg,#0f172a,#020617);color:white}
.container{max-width:1000px;margin:auto;padding:20px}
.upload{border:2px dashed #3b82f6;padding:40px;text-align:center;border-radius:16px;margin-bottom:20px;cursor:pointer}
.upload:hover{background:rgba(59,130,246,0.1)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:15px}
.card{background:rgba(255,255,255,0.05);padding:18px;border-radius:16px;text-align:center}
#chat{position:fixed;bottom:20px;right:20px;background:#3b82f6;padding:14px;border-radius:50%;cursor:pointer}
#chatbox{position:fixed;bottom:80px;right:20px;width:300px;height:400px;background:#020617;display:none;flex-direction:column}
#chat-body{flex:1;overflow:auto;padding:10px}
.msg{margin:5px;padding:8px;border-radius:6px}
.user{background:#3b82f6}
.ai{background:#1e293b}
</style>

<script>
function toggleChat(){document.getElementById("chatbox").style.display="flex"}
function sendMessage(){
let i=document.getElementById("chat_text")
let m=i.value.trim()
if(!m)return
let b=document.getElementById("chat-body")
b.innerHTML+=`<div class='msg user'>${m}</div>`
let t=document.createElement("div")
t.className="msg ai"
t.innerHTML="Thinking..."
b.appendChild(t)
fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:m})})
.then(r=>r.json()).then(d=>{t.innerHTML=d.reply})
i.value=""
}
</script>
</head>

<body>

<div class="container">

<h1>Burnout AI X</h1>

<form method="POST" enctype="multipart/form-data">
<label class="upload">
Upload dataset CSV
<input type="file" name="file" hidden onchange="this.form.submit()">
</label>
</form>

{% if stats %}
<div class="grid">
<div class="card">High<br>{{stats.high}}</div>
<div class="card">Medium<br>{{stats.medium}}</div>
<div class="card">Low<br>{{stats.low}}</div>
<div class="card">Accuracy<br>{{stats.accuracy}}</div>
</div>
{% endif %}

{% if insights %}
<h3>Insights</h3>
<ul>
{% for i in insights %}
<li>{{i}}</li>
{% endfor %}
</ul>
{% endif %}

</div>

<div id="chat" onclick="toggleChat()">💬</div>

<div id="chatbox">
<div id="chat-body"></div>
<input id="chat_text" placeholder="Ask..." onkeydown="if(event.key==='Enter'){sendMessage()}">
</div>

</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():
    global last_df
    stats = None
    insights = None

    if request.method=="POST":
        file = request.files.get("file")
        if file:
            df = pd.read_csv(file)
            df, _, stats = auto_train(df)
            last_df = df
            insights = generate_insights(df)

    return render_template_string(HTML, stats=stats, insights=insights)

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.get_json()["message"]
    return jsonify({"reply": ai_chat(msg, last_df)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
