from flask import Flask
from flask import request
import requests 
import torch

from mnist_classifier.model import Model
from mnist_classifier.config import Config
from mnist_classifier.mnist_dataset import MnistDataset


# Configuration.
config = Config()

# Set inference to cpu.
torch_device=torch.device('cpu')    

# Get the model and set the optimizer, learning rate scheduler, and 
model = Model()
model.load_state_dict(torch.load(config.model_evaluate_checkpoint_path, map_location='cpu'))
model.to(torch_device)
model.eval()
test_dataset = MnistDataset("test")

# Flask app.
app = Flask(__name__)

def server(host='0.0.0.0',port=5001):
    app.run(host=host, port=port)

@app.route('/')
def home():
    return "Welcome to the MNIST classifier run inference server!"
        
@app.route('/run_inference', methods=['GET','POST'])
def run_inference():
    """ Inference server receives message to run inference from query server. """
    if request.method=='GET':
        message=inference()
        return message

def inference():
    """ Get 5 images from the test MNIST dataset, run inference 
        on them, and then return the results and true labels.
    """
    with torch.no_grad():
        label_outputs=[]
        prediction_outputs=[]
        for i in range(6):
            image,label=test_dataset[i]
            image=image.unsqueeze(0)
            if torch_device == 'cuda':
                image=image.cuda()
            image.to(torch_device)
            output=model(image)
            prediction=output.argmax(dim=1, keepdim=True)
            label_outputs.append(label.numpy())
            prediction_outputs.append(prediction.numpy())
    return str("Real Label: {} -- prediction: {}".format(label_outputs, prediction_outputs))