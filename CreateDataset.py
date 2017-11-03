import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
class ASLDataset:
    def _parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string, channels = 3)
        image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 28, 28)
        image_resized = tf.reshape(image_resized, [28 * 28 * 3])
        return image_resized, label

    def size(self):
        return len(self.files)

    def __init__(self):
        self.path = join(os.getcwd(), "Dataset")
        self.alphabet = []
        for i in range(0, 26):
            self.alphabet.extend([0.0])
        self.labels = []
        self.files = [join(self.path, f) for f in listdir(self.path) if isfile(join(self.path, f)) and ord(f[0]) - 65 >= 0]
        hotLabels = [ord(f[0]) - 65 for f in listdir(self.path) if isfile(join(self.path, f)) and ord(f[0]) - 65 >= 0]
        for pos in hotLabels:
            newAlphabet = list(self.alphabet)
            newAlphabet[pos] = 1.0
            self.labels.extend([newAlphabet])
        
        self.Tensorfiles = tf.constant(self.files)
        self.labels = tf.constant(self.labels)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.Tensorfiles, self.labels))
        self.dataset = self.dataset.map(self._parse_function)
