from flask import Flask
from flask import request
import requests 

# Flask app.
app = Flask(__name__)

def server(host='0.0.0.0',port=5000):
    app.run(host=host, port=port)

@app.route('/')
def home():
    return "Welcome to the MNIST classifier launch server!"

@app.route('/launch_inference', methods=['GET','POST'])
def launch_inference():
    """ Launch inference by contacting the inference server. """
    if request.method=='GET':
        message=requests.get('http://0.0.0.0:5001/run_inference')
        return "The results of the inference are: " + message.text