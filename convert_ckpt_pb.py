import tensorflow as tf

meta_path = 'D:\\COVID-Net\\model\\'
# Your .meta file
output_node_names = ['dense_3/Softmax']    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path + 'model.meta_eval')

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint(meta_path))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
