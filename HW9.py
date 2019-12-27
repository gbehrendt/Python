# ME:4150 Artificial Intelligence in Engineering
# HW9 - Data Augmentation (picture generation)
# by Gabriel Behrendt

import warnings
warnings.simplefilter("ignore")     #ignore warnings

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt

# Load the most beautiful picture in the world
img = load_img('./screw.jpg')
print('Original')
plt.imshow(img)
plt.show()

# Create object to transform data
data_generator = ImageDataGenerator(rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.3,
                                    zoom_range=0.3,
                                    horizontal_flip=True
                                    )

# Preprocess data
X = img_to_array(img)  
X = X.reshape((1,) + X.shape)  

# Apply transformation
i = 0
print('Transfomed')
for batch in data_generator.flow(X):
    i += 1
    plt.imshow(array_to_img(batch[0]))
    plt.show()
    if i % 8 == 0:  # Generate eight transformed pictures
        break  # To avoid generator to loop indefinitely
        
# Load the most beautiful picture in the world
img = load_img('./nails.jpg')
print('Original')
plt.imshow(img)
plt.show()

# Create object to transform data
data_generator = ImageDataGenerator(rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.3,
                                    zoom_range=0.3,
                                    horizontal_flip=True
                                    )

# Preprocess data
X = img_to_array(img)  
X = X.reshape((1,) + X.shape)  

# Apply transformation
i = 0
print('Transfomed')
for batch in data_generator.flow(X):
    i += 1
    plt.imshow(array_to_img(batch[0]))
    plt.show()
    if i % 8 == 0:  # Generate eight transformed pictures
        break  # To avoid generator to loop indefinitely