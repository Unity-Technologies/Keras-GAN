from PIL import Image
import numpy as np
from tensorboardX import SummaryWriter
from tensorboardX import utils

class UnityModel(object):
	def __init__(self):
		self.writer = SummaryWriter()

	def save_image(self, title, nparray_of_pixels, iter):
		#nparray_of_pixels = np.transpose(nparray_of_pixels,(2,0,1))
		#formatted = (nparray_of_pixels * 255 / np.max(nparray_of_pixels)).astype('uint8')
		self.writer.add_image(title, utils.make_grid(nparray_of_pixels, ncols=4), iter)
