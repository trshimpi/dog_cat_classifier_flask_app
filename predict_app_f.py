import numpy as np
import base64
import io
from PIL import Image
import tensorflow as tf
import keras
from keras.models import Sequential,load_model
from keras.layers import Flatten,Conv2D,Dense
from keras.preprocessing.image import ImageDataGenerator,img_to_array

from flask import Flask
from flask import request
from flask import jsonify

app=Flask(__name__)

def get_model():
	global model
	model = load_model("vgg16_re_dc.h5")
	# this is key : save the graph after loading the model
	global graph
	graph = tf.get_default_graph()
	print(" *model loaded")

def preprocessing(image,target_size):
	if image.mode !="RGB":
		image = image.convert("RBG")
	image = image.resize(target_size)
	image = img_to_array(image)
	image = np.expand_dims(image,axis=0)

	return image

print(" *loading the model")
get_model()

@app.route("/predict",methods=["POST"])
def predict():
	message = request.get_json(force=True)
	encoded = message["image"]
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))

	processed_image = preprocessing(image,target_size=(224,224))

	with graph.as_default():
		prediction = model.predict(processed_image).tolist()

	response = {
	'prediction':{
					'dog':prediction[0][0],
					'cat':prediction[0][1]
				}
	}
	return jsonify(response)

