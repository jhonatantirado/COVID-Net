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

bruise_width, bruise_height, bruise_channels = 224, 224, 3

inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

def load_image_v2(image_path):
    # url = cloudinary.utils.cloudinary_url(image_path)
    # urllib.request.urlretrieve(url[0], image_path)

    img = image.load_img(image_path, target_size=(bruise_width, bruise_height, bruise_channels))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    return img

def load_image(image_path):
    url = cloudinary.utils.cloudinary_url(image_path)
    print (url)
    urllib.request.urlretrieve(url[0], image_path)
    print (image_path)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0

    return img

def print_image_data(image_tensor):
    print  (type(image_tensor))
    print  (image_tensor.shape)
    print  (image_tensor)
    print ('-------------------')

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def get_result_v2(img_tensor):
    tf_graph = load_pb('output_graph.pb')
    # init_op = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session(graph=tf_graph)
    # sess.run(init_op)
    data_tf = tf.convert_to_tensor(img_tensor, np.float32)
    img_tensor_v = sess.run(data_tf)
    # img_tensor.eval(session=sess)
    output_tensor = tf.nn.softmax(tf_graph.get_tensor_by_name('dense_3/Softmax:0'))
    input_tensor = tf_graph.get_tensor_by_name('input_1:0')
    output = sess.run(output_tensor, feed_dict={input_tensor: img_tensor_v})

    return output

def get_result(x):
    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join('model', 'model.meta_eval'))
    saver.restore(sess, os.path.join('model', 'model-6207'))

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name("input_1:0")
    pred_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")

    pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})

    return pred

@app.route('/todo/api/v1.0/prediagnosis/<string:image_url>', methods=['GET'])
def get_prediagnosis(image_url):
    img_tensor = load_image(image_url)
    print ('image loaded')
    result = get_result(img_tensor)
    print ('before final return')
    # os.remove(image_url)
    print('Prediction: {}'.format(inv_mapping[result.argmax(axis=1)[0]]))

    # return result
    return jsonify({'result': result.tolist()})

if __name__ == '__main__':
    # http://res.cloudinary.com/digrubrgw/image/upload/v1566523367/rxg3y4xbfvog4qgdnomi.png
    # image_url = "assets/ex-covid.jpeg"
    # image_url = "f5aipsq8svdge34jkpkr.png"
    # result = get_prediagnosis(image_url)
    # print(result)
    app.run(host="0.0.0.0", debug=True)
