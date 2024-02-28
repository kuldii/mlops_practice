import os

if not os.path.exists('datasets/train'):
    os.makedirs('datasets/train')
if not os.path.exists('datasets/test'):
    os.makedirs('datasets/test')