#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import model_train as model
import argparse
import pickle
import h5py
import skimage
import skimage.io
import scipy.misc
import random
import json
import os
import shutil
from os.path import join
from Utils import image_processing

def main():
  z_dim = 100   # Noise Dimension
  t_dim = 100   # Text feature dim
  batch_size = 64
  image_size = 64

  gf_dim = 64       # of first Gen Layer kernels
  df_dim = 64       # of first Dis Layer kernels
  gfc_dim = 1024    # of fully connected kernels

  caption_vector_length = 600
  TRAINING_TYPE = 'gan_train'
  TRAINING_DIR = 'faces'
  IMG_DIR = 'imgs'
  caption_vectors = 'caption_vectors.hdf5'

  save_feq = 30
  epochs = 550
  learning_rate = 0.0002
  beta1 = 0.5

  parser = argparse.ArgumentParser()
  parser.add_argument('--resume', type=str, default=None, help='Pre-Trained Model Path, to resume from')
  parser.add_argument('--dis_updates', '-du', type=int, default=1, help='discriminator update feq')
  parser.add_argument('--gen_updates', '-gu', type=int, default=2, help='generator update feq')
  args = parser.parse_args()


  model_options = {
    'z_dim' : z_dim,
    't_dim' : t_dim,
    'batch_size' : batch_size,
    'image_size' : image_size,
    'gf_dim' : gf_dim,
    'df_dim' : df_dim,
    'gfc_dim' : gfc_dim,
    'caption_vector_length' : caption_vector_length
  }

  gan = model.GAN(model_options)
  input_tensors, variables, loss, outputs, checks = gan.build_model()
  with tf.variable_scope(tf.get_variable_scope(), reuse=False):
    d_optim =\
      tf.train.AdamOptimizer(\
        learning_rate, beta1 = beta1
      ).minimize(loss['d_loss'], var_list=variables['d_vars'])
    g_optim =\
      tf.train.AdamOptimizer(\
        learning_rate, beta1 = beta1
      ).minimize(loss['g_loss'], var_list=variables['g_vars'])

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  init = tf.global_variables_initializer()
  sess.run(init)

  saver = tf.train.Saver(max_to_keep=None)
  if args.resume:
    saver.restore(sess, args.resume)

  loaded_data = load_training_data(TRAINING_DIR, TRAINING_TYPE,
                                   IMG_DIR, caption_vectors)

  for i in range(1, epochs+1):
    batch_no = 0
    while batch_no*batch_size < loaded_data['data_length']:
      real_images, fake_images, caption_vectors, z_noise, image_files =\
        get_training_batch(batch_no, batch_size, image_size,
                           z_dim, caption_vector_length, 'train',
                           TRAINING_TYPE, IMG_DIR, TRAINING_DIR,
                           loaded_data)

      # DISCR UPDATE
      for j in range(args.dis_updates):
        check_ts = [checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]
        _, d_loss, gen, d1, d2, d3 =\
          sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
                   feed_dict = {
                     input_tensors['t_real_image'] : real_images,
                     input_tensors['t_wrong_image'] : fake_images,
                     input_tensors['t_real_caption'] : caption_vectors,
                     input_tensors['t_z'] : z_noise,
                   })

      print('d1 = {:5f} d2 = {:5f} d3 = {:5f} D = {:5f}'.format(d1, d2, d3, d_loss))

      # GEN UPDATE
      for j in range(args.gen_updates):
        _, g_loss, gen =\
          sess.run([g_optim, loss['g_loss'], outputs['generator']],
                   feed_dict = {
                     input_tensors['t_real_image'] : real_images,
                     input_tensors['t_wrong_image'] : fake_images,
                     input_tensors['t_real_caption'] : caption_vectors,
                     input_tensors['t_z'] : z_noise,
                   })

      print('d_loss = {:5f} g_loss = {:5f} batch_no = {} epochs = {}'.format(d_loss, g_loss, batch_no, i))
      print('-'*60)
      batch_no += 1
      if (batch_no % save_feq) == 0:
        save_for_vis(TRAINING_DIR, TRAINING_TYPE, real_images,
                     gen, image_files)
        save_path =\
          saver.save(sess, join(TRAINING_DIR, TRAINING_TYPE, 'Models',
                                'latest_model_'
                                '{}_temp.ckpt'.format(TRAINING_DIR)))
    if i%5 == 0:
      save_path =\
        saver.save(sess, join(TRAINING_DIR,
                              TRAINING_TYPE, 'Models', 'model_after_'
                              '{}_epoch_{}.ckpt'.format(TRAINING_DIR, i)))

def load_training_data(training_dir, training_type, imgs_dir, caption_vectors):
  h = h5py.File(join(training_dir, training_type, caption_vectors), 'r')
  flower_captions = {}
  for ds in h.items(): flower_captions[ds[0]] = np.array(ds[1])
  image_list = [key for key in flower_captions]
  image_list.sort()
  random.shuffle(image_list)
  return {
    'image_list' : image_list,
    'captions' : flower_captions,
    'data_length' : len(image_list)
  }

def save_for_vis(training_dir, training_type, real_images,
                 generated_images, image_files):

  shutil.rmtree(join(training_dir, training_type, 'samples'))
  os.makedirs(join(training_dir, training_type, 'samples'))
  for i in range(0, real_images.shape[0]):
    real_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
    real_images_255 = (real_images[i,:,:,:])
    scipy.misc.imsave(\
      join(training_dir, training_type, 'samples',
           '{}_{}'.format(i, image_files[i].split('/')[-1] )),
      real_images_255)
    fake_image_255 = np.zeros((64,64,3), dtype=np.uint8)
    fake_images_255 = (generated_images[i,:,:,:])
    scipy.misc.imsave(join(training_dir, training_type, 'samples',
                           'fake_image_{}.jpg'.format(i)),
                      fake_images_255)

def get_training_batch(batch_no, batch_size, image_size, z_dim,
                       caption_vector_length, split, training_type, imgs_dir,
                       training_dir, loaded_data):
  real_images = np.zeros((batch_size, 64, 64, 3))
  fake_images = np.zeros((batch_size, 64, 64, 3))
  captions = np.zeros((batch_size, caption_vector_length))

  count = 0
  image_files = []
  for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
    idx = i % len(loaded_data['image_list'])
    image_file =  join(training_dir, imgs_dir, loaded_data['image_list'][idx])
    image_array = image_processing.load_image_array(image_file, image_size)
    real_images[count,:,:,:] = image_array

    # Improve this selection of wrong image
    wrong_image_id = random.randint(0, len(loaded_data['image_list'])-1)
    wrong_image_file =  join(training_dir, imgs_dir,
                             loaded_data['image_list'][wrong_image_id])
    wrong_image_array = image_processing.load_image_array(wrong_image_file,
                                                          image_size)
    fake_images[count, :,:,:] = wrong_image_array

    captions[count,:] = loaded_data\
        ['captions'][loaded_data['image_list'][idx]][0][:caption_vector_length]
    image_files.append(image_file)
    count += 1

  z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
  return real_images, fake_images, captions, z_noise, image_files

if __name__ == '__main__':
  main()
