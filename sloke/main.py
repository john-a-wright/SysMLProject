from sit1m_data_preprocessing import *
import numpy as np
"""checking how to use the sift1m data preprocessing
"""
download_path = "../sift1m"
splits = build_sift1m(download_path)

train_splits = get_test_split(splits)