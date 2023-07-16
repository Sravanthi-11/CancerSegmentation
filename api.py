from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import json
from flask_cors import CORS
import tensorflow as tf
import json 
import os
from model_definition import SegmentationModel 

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"],}})

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = SegmentationModel().model
model.load_weights('cancer_weights.h5') 

@app.route('/', methods=['POST'])
def scoring_endpoint(): 
    f = request.files['file']
    print(f.filename)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename)) 
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image)
    image = tf.cast(image, tf.float32) / 255.0
    yhat = model.predict(tf.expand_dims(image, axis=0))
    print(image)
    return {"prediction": json.dumps(yhat.tolist())}


if __name__ == '__main__': 
    app.run(port=4000)
