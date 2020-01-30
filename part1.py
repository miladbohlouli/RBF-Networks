from data_read import *
from sklearn.model_selection import train_test_split

data_x, data_y = read_data()
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

num_train = x_train.shape[0]
num_classes = len(np.unique(data_y))
num_test = x_test.shape[0]
k = 700
lam = 0.1
models = list()
path = "models/1.2"

#   We must change the format of the y_train to one_hot
y_train_one_hot = np.zeros((num_train, num_classes)) - 1
y_train_one_hot[np.arange(num_train), np.array(y_train, dtype=np.int8)] = 1

# for cat in range(num_classes):
#     net = RBF_NN(k, lam, name="RBF_" + str(cat))
#     net.train(x_train, y_train_one_hot[:, cat], mode='typical')
#     print("trained model (%d/%d)"%(cat+1, num_classes))
#     models.append(net)


#   In this line we save all the models at once
# save_models(models, path)

file_names = []
for i in range(num_classes):
    file_names.append(os.path.join(path, "RBF_" + str(i)))

models = load_models(file_names)

results = np.zeros((num_test, num_classes))
for i in range(num_classes):
    results[:, i] = models[i].test(x_test).ravel()

labels = np.argmax(results, axis=1)

RBF_NN.evaluate(y_test, labels)

fig = plt.figure()
plt.title("test data shown with k:%d"%(k))
ax = fig.add_subplot(111, projection='3d')
for clus in np.unique(labels):
    data_clus = x_test[labels == clus]
    ax.scatter(data_clus[:, 0], data_clus[:, 1], data_clus[:, 2], s=3)
plt.show()