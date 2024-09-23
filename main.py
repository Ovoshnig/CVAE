import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K

hidden_dim = 2
num_classes = 33
batch_size = 100 

# Загрузка датасета
dataset = np.load('dataset.npz')
x_test, y_test_cat = dataset['x_test'], dataset['y_test_cat']

# Загрузка обученных моделей
encoder = load_model('models/encoder.keras', custom_objects={'noiser': noiser}, safe_mode=False)
decoder = load_model('models/decoder.keras', custom_objects={'noiser': noiser}, safe_mode=False)

lb = lb_dec = y_test_cat
h = encoder.predict([x_test, lb], batch_size=batch_size)
plt.scatter(h[:, 0], h[:, 1])
plt.show()

n = 4
total = 2*n+1
input_lbl = np.zeros((1, num_classes))
input_lbl[0, 5] = 1

plt.figure(figsize=(total, total))

h = np.zeros((1, hidden_dim))
num = 1
for i in range(-n, n+1):
    for j in range(-n, n+1):
        ax = plt.subplot(total, total, num)
        num += 1
        h[0, :] = [1*i/n, 1*j/n]
        img = decoder.predict([h, input_lbl])
        plt.imshow(img.squeeze(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show() 