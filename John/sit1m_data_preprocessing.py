from sift1m_dataset_builder import Builder
import tensorflow_datasets.public_api as tfds
from etils.epath import Path
import os
import torch
import numpy as np

def pwd():
  print("dir: " + os.getcwd())

def build_sift1m(path_str):
  print("Building Sift1M")
  path = Path(path_str)
  download_manager = tfds.download.DownloadManager(download_dir=path)
  b = Builder()
  splits = b._split_generators(download_manager)
  return splits

def get_train_split(splits):
  print("Building train split array")

  file_path = 'train_data.npy'

  if os.path.exists(file_path):
    print("Found existing train data file")
    train_input_array = np.load(file_path)
    return train_input_array
  else:
    embeddingTuples = splits["database"]
    tensor_list = []
    i = 0

    for tup in embeddingTuples:
      dict = tup[1]
      tensor_list.append(torch.tensor(dict["embedding"]))
      i=i+1

      if i % 100000 == 0:
        print("Progress: " + str(i/10000) + "%")

    combined_tensor = torch.stack(tensor_list)

    train_input_array = combined_tensor.numpy()
    np.save(file_path, train_input_array)
    return train_input_array

def get_test_split(splits):
  print("Building test split array")
  embeddingTuples = splits["test"]
  tensor_list = []
  neighbors_list = []
  i = 0

  for tup in embeddingTuples:
      dict = tup[1]
      tensor_list.append(torch.tensor(dict["embedding"]))
      neighbors_list.append(dict["neighbors"])
      i=i+1
      if i % 5000 == 0:
          print("Progress: " + str(i/100) + "%")

  combined_tensor_test = torch.stack(tensor_list)

  input_array_test = combined_tensor_test.numpy()
  return input_array_test, neighbors_list