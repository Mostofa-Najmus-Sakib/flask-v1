from flask import Flask, render_template, request 
import pickle


app = Flask(__name__)
tokenizer= pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

# cv = pickle.load(open("models/cv.pkl"))
# clf = pickle.load(open("models/clf.pkl"))

@app.route('/')
def home():
    text = ""
    if request.method == 'POST':
        text = request.form.get('content')
    return render_template('index.html', text=text)

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get('content')
    tokenized_email = tokenizer.transform([email_text]) # X 
    prediction = model.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, text=email_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)


# if __name__ == '__main__':
#     app.run( debug=True)