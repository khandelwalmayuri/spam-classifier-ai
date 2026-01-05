from flask import Flask, render_template, request
from spam import predict_spam

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        message = request.form["message"]
        result = predict_spam(message)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
