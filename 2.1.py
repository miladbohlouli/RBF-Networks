from data_read import *
from sklearn.model_selection import train_test_split

start = 2
k_range = 235
step = 1
lam = 0.1
noise = 0.8

path = "models/2.1.3"

data_x, data_y = generate_regression_data(part='3', num_data=1000, variance=noise)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33)

MSE_test = np.zeros(int((k_range-start)/step))
MSE_train = np.zeros(int((k_range-start)/step))
regression = []
predictions_test = []
predictions_train = []
for i, k in enumerate(range(0, k_range-start, step)):
    print("processing k value:(%d/%d)" % (k + start, k_range))
    regression.append(RBF_NN(k+start, lam, "regression", type='2'))
    regression[i].train(x_train, y_train)

    predictions_train.append(regression[i].test(x_train))
    predictions_test.append(regression[i].test(x_test))

    MSE_train[i] = MSE(predictions_train[i], y_train)
    MSE_test[i] = MSE(predictions_test[i], y_test)

index = np.argmin(MSE_test)
best_regressor = regression[index]
best_pred_test = predictions_test[index]
best_pred_train = predictions_train[index]
regression[index].save(path)

print("\n******** best MSE ***********")
print("MSE train: %.6f\nMSE test: %.6f"%(MSE_train[index], MSE_test[index]))
print("with k: %d" % (index * step + start))

#   plot the amount of the MSE of train
plt.figure()
plt.grid(True, alpha=0.4)
plt.plot(range(start, k_range, step), MSE_train, color='yellow')
plt.plot(range(start, k_range, step), MSE_test, color='red')
plt.title("MSE error for different values of k")
plt.ylabel("MSE error")
plt.xlabel("k")
plt.legend(("training MSE", "testing MSE"), loc=2)

#   ploting the output
plt.figure()
plt.title("Fitting the model on test data")
plt.grid(True, alpha=0.4)
plt.scatter(data_x, data_y, color='purple', s=3)
plt.plot(x_train[np.argsort(x_train)], best_pred_train[np.argsort(x_train)], color='yellow')
plt.legend(("predictions", "true values"), loc=2)
plt.show()
