#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import model
import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
import os

tf.set_random_seed(50)


random.seed(729)
np.random.seed(1005)

def main():
  z_dim = 100
  t_dim = 256
  image_size = 64
  model_type = 'gan'
  n_images = 5

  # model Parameters
  gf_dim = 64
  df_dim = 64

  # fully connected
  gfc_dim = 1024
  caption_vector_length = 600

  MODEL_DIR = 'gan-models'
  MODEL_NAME = 'final_model.ckpt'
  OUTPUT_DIR='samples'

  caption_vectors_name = 'test_caption_vectors.hdf5'

  model_options = {
    'z_dim' : z_dim,
    't_dim' : t_dim,
    'batch_size' : n_images,
    'image_size' : image_size,
    'gf_dim' : gf_dim,
    'df_dim' : df_dim,
    'gfc_dim' : gfc_dim,
    'caption_vector_length' : caption_vector_length
  }

  gan = model.GAN(model_options)
  _, _, _, _, _ = gan.build_model()
  sess = tf.Session()
  saver = tf.train.Saver()
  saver.restore(sess,
                join(MODEL_DIR, model_type, 'Models', MODEL_NAME))

  input_tensors, outputs = gan.build_generator()

  h = h5py.File(join(MODEL_DIR, model_type, caption_vectors_name))
  caption_image_dic = {}

  for i, key in enumerate(h):
    caption_images = []
    z_noise = np.random.uniform(-5, 5, [n_images, z_dim])
    caption = np.array([h[key][0, :caption_vector_length]
                       for i in range(n_images)])

    [gen_image] =\
      sess.run([outputs['generator']],
               feed_dict = {input_tensors['t_real_caption'] : caption,
                            input_tensors['t_z'] : z_noise} )

    caption_image_dic[key] =\
      [gen_image[i, :, :, :] for i in range(0, n_images)]

  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
  for key in h:
    for i, im in enumerate(caption_image_dic[key]):
      scipy.misc.imsave(join(OUTPUT_DIR, 'sample_'+key+'_'+str(i + 1)+'.jpg'), im)

if __name__ == '__main__':
  main()
