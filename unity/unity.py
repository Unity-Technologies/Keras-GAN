from PIL import Image
import numpy as np
import glob
from keras.preprocessing.text import Tokenizer

def load_labels(filename, seen_labels):
	with open(filename) as f:
		content = f.readlines()[0]
	t = sorted(content.split(" "))
	seen_labels.extend(t)
	return t

def load_image(filename, resize=None) :
    img = Image.open(filename).convert('RGB')
    if resize is not None:
    	img = img.resize((resize,resize))

    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def encode_word_presence(words, all_words):
	array = np.zeros(len(all_words))
	for w in words:
		i = all_words.index(w)
		array[i] = 1.0
	return array.tolist()

def augment(x_train, y_train, resize):
	images_to_augments = len(x_train)
	for i in range(images_to_augments):
		im = Image.fromarray(x_train[i].astype('uint8').reshape((resize,resize,3)))
		f1 = im.transpose(Image.FLIP_LEFT_RIGHT)
		f2 = im.transpose(Image.FLIP_TOP_BOTTOM)
		x_train.append(np.asarray(f1, dtype="int32"))
		x_train.append(np.asarray(f2, dtype="int32"))
		y_train.append(y_train[i])
		y_train.append(y_train[i])

def semantic_maps(resize=None):
	seen_labels = list()
	x_train = ([load_image(f, resize) for f in glob.glob("../data/semantic_maps/train/img/*.png")])
	x_test = ([load_image(f, resize) for f in glob.glob("../data/semantic_maps/test/img/*.png")])
	y_train = ([load_labels(f, seen_labels) for f in glob.glob("../data/semantic_maps/train/txt/*.txt")])
	y_test = ([load_labels(f, seen_labels) for f in glob.glob("../data/semantic_maps/test/txt/*.txt")])
	augment	(x_train, y_train, resize)
	seen_labels = sorted(list(set(seen_labels)))
	y_train = ([encode_word_presence(a, seen_labels) for a in y_train])
	y_test = ([encode_word_presence(a, seen_labels) for a in y_test])
	return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

def get_number_of_unique_classes():
	seen_labels = list()
	y_train = [load_labels(f, seen_labels) for f in glob.glob("../data/semantic_maps/train/txt/*.txt")]
	unique_labels = len(list(set(seen_labels)))
	return unique_labels

def semantic_maps_shape():
	return 64, 3, get_number_of_unique_classes()