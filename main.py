#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, adjusted_rand_score, silhouette_score, rand_score, accuracy_score
from mlxtend.plotting import plot_confusion_matrix


from tf_som import SOM
from label_and_test import label, test


# hyper-parameters
train_data = 60000
label_data = int(train_data*0.01) # 1% of the training data for labeling
test_data = 10000

input_dim = 784 # input dimension because of the MNIST dataset (28x28)
map_wth = 5 # width of the output map
map_hgt = 20 # height of the output map
epochs = 10
learning_rate_0 = 1.0
learning_rate_T = 0.01
neigbourhood_radius_0 = 10.0
neigbourhood_radius_T = 0.01

class_nbr = 10 # number of classes

# GPU name
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError("GPU device not found!")
print('Found GPU at: {}'.format(device_name))


# importing dataset
(x_train_all, index_train_all), (x_test_all, index_test_all) = mnist.load_data()

# normalizing images -> As the pixel values range from 0 to 256, apart from 0 the range is 255. So dividing all the values by 255 will convert it to range from 0 to 1.
x_train_all = x_train_all.astype('float32') / 255.
x_test_all = x_test_all.astype('float32') / 255.

# transform the image into a single vector of 28x28 = 784 pixels
x_train_all = x_train_all.reshape((60000, 784))
x_test_all = x_test_all.reshape((10000, 784))

# constructing datasets of train and data
x_train = np.copy(x_train_all[:train_data,:])
y_train = np.copy(index_train_all[:train_data]) # this is not used since we are working with an semi-supervised dataset
x_label = np.copy(x_train_all[:label_data,:])
y_label = np.copy(index_train_all[:label_data])
x_test = np.copy(x_test_all[:test_data,:])
y_test = np.copy(index_test_all[:test_data])

# %%

print("\nHyper-parameters:" +
        f"\n - initial learning rate = {learning_rate_0}" +
        f"\n - final learning rate = {learning_rate_T}" +
        f"\n - initial neighbourhood radius = {neigbourhood_radius_0}" +
        f"\n - final neighbourhood radius = {neigbourhood_radius_T}")


#%%
# train the network
som = SOM(
        map_width = map_wth, 
        map_height = map_hgt, 
        input_dim = input_dim,
        initial_learning_rate = learning_rate_0,
        final_learning_rate = learning_rate_T,
        initial_neighbourhood_radius = neigbourhood_radius_0,
        final_neighbourhood_radius = neigbourhood_radius_T,
        epochs = epochs
)


# start_time = timeit.default_timer()
som.train(x_train)

#%%
# display neurons weights as mnist digits
def plot(weights):
    weights = weights.numpy()
    som_grid = plt.figure(figsize=(10, 10)) # width, height in inches
    for n in range(map_wth*map_hgt):
        image = weights[n].reshape([28,28]) # x_train[num] is the 784 normalized pixel values
        sub = som_grid.add_subplot(map_wth, map_hgt, n + 1)
        sub.set_axis_off()
        clr = sub.imshow(image, cmap = plt.get_cmap("jet"), interpolation = "nearest")


i = 0
for weight in som.saved_weights[0::10]:
    plot(weight)
    plt.savefig(f"plot_vid/som_weights_{i}.png", bbox_inches='tight', pad_inches=0)
    i = i + 1
#%%
# Supervised part: labeling the groups
weights = som.get_weights().numpy()

# label the network
neuron_label = label(class_nbr, weights, x_label, y_label)

# save labels in txt file
aux = neuron_label.reshape(map_wth,map_hgt)
file_object = open(f'labels_{map_wth}x{map_hgt}.txt', 'a')
for i in range(map_wth):
    file_object.write(str(aux[i])+'\n')
file_object.close()


# test the network
y_pred = test(weights, x_test, neuron_label)


# %%
# metrics
print("Accuracy Score = ", accuracy_score(y_test, y_pred))

print("Adjusted Rand Score = ", adjusted_rand_score(y_test, y_pred))

print("Rand Score = ", rand_score(y_test, y_pred))



#%%

conf_mat = confusion_matrix(y_test, y_pred)

fig, ax = plot_confusion_matrix(conf_mat=conf_mat,
                                show_normed=True,
                                show_absolute=False,
                                class_names=[0,1,2,3,4,5,6,7,8,9],
                                figsize=(8, 8))

fig.show()

#todo:


# %%
