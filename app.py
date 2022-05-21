# import library
from flask import Flask, request
from flask_restful import Api
from flask_cors import CORS
import os
import librosa
import librosa.feature
from module import MyModule

# init object flask
app = Flask(__name__)

# init object flask restfull
# biar bisa di push
api = Api(app)

# init cors
CORS(app)

res = {}
modul = MyModule()

@app.before_first_request
def before_first_request():
    modul.fit()

@app.route("/")
def landing():
    return res

@app.route("/data", methods=["GET", "POST"])
def get_prediction():
    if request.method == "GET":
        return res
    if request.method == 'POST':
        save_path = os.path.join("audio/", "temp.wav")
        request.files['audio_data'].save(save_path)
        request_model = request.form["request_model"]
        try:
            x, sr = librosa.load("audio/temp.wav")
            label, dist = modul.predict(x,sr,request_model)
            label = label[0]
        except:
            label = "error#TidakAdaSuara."
            dist = ["null","null"]
        res["result"] = label
        res["distance"] = dist
        return res

if __name__ == "__main__":
    app.run(debug=True, port = int(os.environ.get('PORT', 4000)))