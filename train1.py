import math
import os
import random
import sys
import numpy as np
import tensorflow as tf


def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
    
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _image_to_tfexample(image_data, width, height, paramList):
  example = tf.train.Example(features=tf.train.Features(feature={
      'paramList': float_feature(paramList),    
      'image/raw': bytes_feature(image_data),
      'image/width': int64_feature(width),
      'image/height': int64_feature(height)}))  
  return example
  


def _create_paramList(numParams):
    """Returns numParams floats in range[0,1) as numpy.ndarray"""
    paramList = np.random.random_sample(numParams)
    return paramList


def _create_image(paramList, width, height):
    """Generate image from a paramList"""
    
    img = np.zeros( (height,width) )

    spot = 0
    for param in paramList:
        spot = spot + 1
        v = param
        img[ spot ][ spot ] += v
        
    return img



def _build1(output_filename, numParams, numImages, width, height):
  with tf.Graph().as_default():

    with tf.Session('') as sess:

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                      
          for i in range(0, numImages):

            # create random parameters
            paramList = _create_paramList( numParams )

            # create image from parameters
            img = _create_image(paramList, width, height)

            # convert to [0,255]
            z = img * 255.0
            zz = z.astype(int)
            data = tf.compat.as_bytes(zz.tobytes())
            # write as tf record            
            example = _image_to_tfexample( data, width, height, paramList)
            tfrecord_writer.write(example.SerializeToString())


            
def testReadExample(filename):

  with tf.Graph().as_default():

    with tf.Session('') as sess:
  
      filename_queue = tf.train.string_input_producer([filename])
      reader = tf.TFRecordReader()

      for i in range(0, 10):
        print( "read record:" + str(i))
        
        _, serialized_example = reader.read(filename_queue)
  
        features = tf.parse_single_example(
          serialized_example,
          features={
            'paramList': tf.FixedLenFeature([], tf.float32),        
            'image/raw': tf.FixedLenFeature([], tf.string),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/height': tf.FixedLenFeature([], tf.int64),
          })

        image = tf.decode_raw(features['image/raw'], tf.uint8)
        height = tf.cast(features['image/height'], tf.int64)
        width = tf.cast(features['image/width'], tf.int64)
      
        #print( width.eval() )
  
            
    
if __name__ == '__main__':
    filename = 'test1.records'
    width, height = 200, 200
    numParams = 10
    numImages = 50
    
    #_build1(filename, numParams, numImages, width, height)


    testReadExample( filename )
            
