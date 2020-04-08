import os
import urllib.request
import cloudinary.uploader
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify
from keras.preprocessing import image
import cv2

app = Flask(__name__)

cloudinary.config(
    cloud_name="ds04o8pmi",
    api_key="618563954112198",
    api_secret="17I684A-O6lVnHh0fRQ9IPje1zQ"
)

img_width, img_height = 224, 224
inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}
weightspath = 'model'
metaname = 'model.meta_eval'
ckptname = 'model-6207'
image_tensor_name = 'input_1:0'
pred_tensor_name = 'dense_3/Softmax:0'

def load_image(image_path):
    url = cloudinary.utils.cloudinary_url(image_path)
    urllib.request.urlretrieve(url[0], image_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype('float32') / 255.0

    return img

def get_result(x):
    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))
    saver.restore(sess, os.path.join(weightspath, ckptname))
    graph = tf.get_default_graph()
    image_tensor = graph.get_tensor_by_name(image_tensor_name)
    pred_tensor = graph.get_tensor_by_name(pred_tensor_name)

    pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})

    return pred

@app.route('/todo/api/v1.0/prediagnosis/<string:image_url>', methods=['GET'])
def get_prediagnosis(image_url):
    img_tensor = load_image(image_url)
    print ('image loaded')
    result = get_result(img_tensor)
    print ('before final return')
    os.remove(image_url)
    print('Prediction: {}'.format(inv_mapping[result.argmax(axis=1)[0]]))

    return jsonify({'result': result.tolist()})

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
