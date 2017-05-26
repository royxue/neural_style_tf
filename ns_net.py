import numpy as np
import tensorflow as tf

from mpimg import imread
from vgg19 import Vgg19


def ns_net(content, style):
	vgg = Vgg19(vgg19_npy_path='./vgg19.npy')
	content = imread(content)
	