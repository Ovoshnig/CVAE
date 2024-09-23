import os
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from PIL import Image

num_classes = 33
dataset = []
labels = []
target_size = (28, 28)

# Путь к корневой папке с подпапками, содержащими изображения букв
root_folder = 'Cyrillic'

# Обход папок в корневой папке
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)

    if os.path.isdir(folder_path):
        # Обход изображений в каждой папке
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            if img_path.endswith(".png"):
                # Загрузка изображения с помощью PIL
                    img = Image.open(img_path)
                    # Изменение размера изображения
                    img = img.resize(target_size)
                    # Преобразование изображения в массив NumPy 
                    img_array = np.array(img)
                    # Преобразование изображения в одноканальный (grayscale)
                    img_array = img_array[:, :, -1]
                    # Добавление изображения и метки в датасет
                    dataset.append(img_array)
                    labels.append(folder_name)  # Используем название папки как метку

letters_num = {letter: labels.count(letter) for letter in set(labels)}

# Аугментация датасета
# Создание экземпляра ImageDataGenerator с параметрами аугментации
datagen = ImageDataGenerator(
    rotation_range=18,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest')

samples_num = 2048

for letter in letters_num:
    reference_indexes = [index for index, label in enumerate(labels) if label == letter]
    reference = np.array([dataset[index] for index in reference_indexes])
    reference = reference.reshape(-1, 28, 28, 1)
    
    # Генерация новых изображений      
    flow = datagen.flow(reference, batch_size=samples_num-letters_num[letter])
    new_images = next(flow)
    new_images = new_images.reshape(-1, 28, 28)

    # Добавление сгенерированного изображения в датасет
    for image in new_images:
        dataset.append(image)
        labels.append(letter)
    letters_num[letter] = samples_num

x = np.array(dataset)
y = np.array(labels)

# Стандартизация входных данных
x = x / 255
x = np.reshape(x, (len(x), 28, 28, 1))

label_encoder = LabelEncoder()
y_digit = label_encoder.fit_transform(y)
y_cat = to_categorical(y_digit, num_classes)

# Разбиение данных на обучающие и проверочные
x_train, x_test, y_train_cat, y_test_cat = train_test_split(x, y_cat, test_size=0.2, random_state=42)
x_train = x_train[:x_train.shape[0] // 64 * 64]
y_train_cat = y_train_cat[:x_train.shape[0] // 64 * 64]
x_test = x_test[:x_test.shape[0] // 64 * 64]
y_test_cat = y_test_cat[:x_test.shape[0] // 64 * 64]

np.savez('dataset.npz', x_train=x_train, y_train_cat=y_train_cat, x_test=x_test, y_test_cat=y_test_cat)