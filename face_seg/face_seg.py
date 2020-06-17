import os
import cv2
import numpy as np
import tensorflow as tf


class FaceSeg(object):
  def __init__(self, model_path, in_hw):
    self.model = tf.saved_model.load('asset/seg_model_saved').signatures["serving_default"]
    self.in_hw = in_hw

  def input_transform(self, imgs, resize_all=True):
    imgs = tf.image.resize(imgs, self.in_hw, 'area')
    img_input = (imgs / 255.)
    return img_input

  def output_transform(self, output, shape):
    output = tf.image.resize(output, shape)
    image_output = tf.cast(output * 255., tf.uint8)
    return image_output

  def get_mask(self, image: np.ndarray) -> np.ndarray:
    assert image.ndim == 4, f'images shape must be [b,h,w,c] now is {image.shape}'
    img_input = self.input_transform(image)
    output = self.model(img_input)['out']
    return self.output_transform(output, shape=image.shape[1:3]).numpy()


def convert_pb_to_savemodel():
  import tensorflow as tf
  from tensorflow.python.saved_model import signature_constants
  from tensorflow.python.saved_model import tag_constants

  export_dir = 'asset/seg_model_saved'
  graph_pb = 'asset/seg_model_384.pb'

  builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

  with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  sigs = {}

  with tf.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()
    inp = g.get_tensor_by_name("input_1:0")
    out = g.get_tensor_by_name("sigmoid/Sigmoid:0")

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = tf.saved_model.signature_def_utils.predict_signature_def({
                                                                                                                           "in": inp}, {"out": out})

    builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], signature_def_map=sigs)

  builder.save()


if __name__ == "__main__":
  # face_seg = FaceSeg(model_path='/home/zqh/workspace/playground/asset/seg_model_384.pb')
  # convert_pb_to_savemodel()
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  model = tf.saved_model.load('asset/seg_model_saved').signatures["serving_default"]
  img = tf.image.decode_jpeg(tf.io.read_file('/home/zqh/workspace/data512x512/27732.jpg'), 3)
  img = tf.image.resize(img, (384, 384), 'area')
  img_input = (img / 255.)[None, ...]
  out = model(img_input)['out']
  import matplotlib.pyplot as plt
  plt.imshow(tf.cast(out[0, ..., 0] * 255, tf.uint8))
