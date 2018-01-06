#!/usr/bin/python3
import os
from os.path import join, isfile
import numpy as np
import argparse
import h5py
import multiprocessing
from multiprocessing import Process, Queue

cpus = multiprocessing.cpu_count()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--text_file', '-tf', type=str, help='caption file', default='sample_testing_text.txt')
  args = parser.parse_args()

  caption_vectors_name = 'test_caption_vectors.hdf5'
  model_type = 'gan'
  MODEL_DIR = 'gan-models'
  GLOVE_PATH = MODEL_DIR + '/glove.6B.300d.txt'

  with open(args.text_file) as f:
    captions = f.read().splitlines()
  captions = [cap for cap in captions]
  captions = [cap.split(',') for cap in captions]
  captions = dict([[cap[0], cap[1].split()]
    for i, cap in enumerate(captions)])

  #word to vec convertion
  wordvecs = open(GLOVE_PATH, 'r').read().splitlines()
  wordvecs = [wordvec.split() for wordvec in wordvecs]
  wordvecs = dict([[wordvec[0], np.array([wordvec[1:]], dtype=np.float32)] for wordvec in wordvecs])
  caption_vectors = {}
  for key, val in captions.items():
    new_val = [val[0], val[2]] if val[1] == 'hair' else [val[2], val[0]]
    caption_vectors[key] =\
      np.hstack((wordvecs[new_val[0]], wordvecs[new_val[1]]))

  filename = join(MODEL_DIR, model_type, caption_vectors_name)
  if os.path.isfile(filename):
    os.remove(filename)
  h = h5py.File(filename, 'w')
  for key in caption_vectors.keys():
    h.create_dataset(key, data=caption_vectors[key])
  h.close()

if __name__ == '__main__':
  main()
