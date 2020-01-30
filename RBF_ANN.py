import tensorflow as tf
from sklearn.cluster import KMeans
from data_read import *
import pickle
import tensorflow_probability as tfp

#   type = 1 when classification otherwise regression
class RBF_NN:
    def __init__(self, k=None, lam=None, name=None, type='1'):
        self.num_dimensions = None
        self.__sess = None
        self.__saver = None
        self.k = k
        self.lam = lam
        self.name = name
        self.type = type

    #   In this method x_train and y_train are numpy arrays, these will be feed to network
    def train(self, x_train, y_train, mode='typical'):

        num_train = len(x_train)
        tf.reset_default_graph()

        #   This is the mode when we have a classification problem
        if self.type == '1':
            self.num_dimensions = x_train.shape[1]

            #   running the k_means algorithm to find the centers of the x_train
            clustering = KMeans(self.k).fit(x_train)
            centers = clustering.cluster_centers_
            labels = clustering.labels_

        elif self.type == '2':
            self.num_dimensions = 1
            clustering = KMeans(self.k).fit(np.repeat(np.reshape(x_train, [num_train, 1]), axis=1, repeats=2))
            centers = clustering.cluster_centers_
            labels = clustering.labels_

        # Find the maximum distance between the centers to use in sigma calculation
        d_max = -1
        for center in centers:
            current_maximum = np.max(np.linalg.norm(center - centers, axis=1), axis=0)
            if current_maximum > d_max:
                d_max = current_maximum

        with tf.variable_scope("test", reuse=False):

            #   This part is the part that we define the model


            if mode == 'ellipsis':
                covariances = np.zeros((self.k, self.num_dimensions, self.num_dimensions))
                for i in range(self.k):
                    covariances[i] = (np.cov(np.transpose(x_train[labels == i])))
                covariances = tf.Variable(covariances, dtype=tf.float64)

            x = tf.placeholder(name="x_input", dtype=tf.float64, shape=[1, self.num_dimensions])
            y = tf.placeholder(name="y_input", dtype=tf.float64, shape=[])
            centers = tf.Variable(centers, dtype=tf.float64)
            sigma = tf.Variable(d_max / (2 * self.k) ** 0.5, name="sigma")

            if mode == 'typical':
                phi = tf.reshape(tf.math.exp(((-1 / (2 * sigma ** 2)) * tf.linalg.norm(x - centers, axis=1) ** 2)), shape=[self.k, 1])
            if mode == 'ellipsis':
                phi = []
                for i in range(self.k):
                    phi.append(tf.reshape(tf.math.exp(-0.5 * tf.matmul(tf.matmul(x - centers[i], tfp.math.pinv(covariances[i]))
                                                            , x - centers[i], transpose_b=True) / sigma), shape=[1]))
                phi = tf.convert_to_tensor(phi)

            p = tf.Variable(tf.eye(self.k, dtype=tf.float64) / self.lam)
            w = tf.get_variable(name="w", initializer=tf.zeros_initializer, shape=[self.k, 1], dtype=tf.float64)

            p_new = p - (tf.matmul(tf.matmul(tf.matmul(p, phi), phi, transpose_b=True), p)) \
                    / (1 + tf.matmul(tf.matmul(phi, p, transpose_a=True), phi))

            g = tf.matmul(p_new, phi)
            alpha = y - tf.matmul(w, phi, transpose_a=True)

            assign0 = tf.assign(p, p_new)
            assign1 = tf.assign(w, w + tf.matmul(g, alpha))

            tf.matmul(phi, w, transpose_a=True, name="output")

        init = tf.global_variables_initializer()

        self.__saver = tf.train.Saver()

        self.__sess = tf.Session()
        self.__sess.run(init)

        for i in range(num_train):
            if i % 300 == 0:
                print("completion percentage: %.2f"%(i*100 / num_train))
                # print(self.__sess.run(part3, feed_dict={self.x: np.reshape(x_train[i], [1, self.num_dimensions]), y: y_train[i]}))
            _, _ = self.__sess.run([assign0, assign1], feed_dict={"test/x_input:0": np.reshape(x_train[i], [1, self.num_dimensions]), y: y_train[i]})

    #   x_test and y_test will be numpy arrays that will be feed to the network
    def test(self, x_test):
        num_test = x_test.shape[0]
        results = np.zeros((num_test, 1))

        for i in range(num_test):
            results[i] = self.__sess.run("test/output:0", feed_dict={"test/x_input:0": np.reshape(x_test[i], [1, self.num_dimensions])})

        return results

    @staticmethod
    def evaluate(ground_truth, predicted_labels):

        num_classes = np.unique(ground_truth).__len__()
        confusion_matrix = np.zeros((num_classes, num_classes))
        recall = np.zeros(num_classes)
        precision = np.zeros(num_classes)
        f1_measure = np.zeros(num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                confusion_matrix[i, j] = np.sum(np.logical_and(predicted_labels == i, ground_truth == j))
            recall[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
            precision[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
            f1_measure[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

        accuracy = np.sum(ground_truth == predicted_labels) / ground_truth.shape[0]
        print("*********Confusion matrix*********")
        print(confusion_matrix)
        print("**************Precision**************")
        print(precision)
        print("average:%f" % (np.average(precision)))
        print("**************recall**************")
        print(recall)
        print("average:%f" % (np.average(recall)))
        print("**************f1_measure**************")
        print(f1_measure)
        print("average:%f" % (np.average(f1_measure)))
        print("*********Accuracy*********")
        print(accuracy)

        return confusion_matrix, precision, recall, f1_measure, accuracy

    def save(self, path):
        dictionary_data = {
            'num_dimensions': self.num_dimensions,
            'k': self.k,
            'lam': self.lam,
            'name': self.name,
        }

        with open(os.path.join(path,  self.name) + ".obj", 'wb') as file_fd:
            pickle.dump(dictionary_data, file_fd)
        self.__saver.save(self.__sess, os.path.join(path, self.name))

    def load(self, path):
        with open(path + ".obj", 'rb') as file_fd:
            dictionary_data = pickle.load(file_fd)
            self.num_dimensions = dictionary_data['num_dimensions']
            self.k = dictionary_data['k']
            self.lam = dictionary_data['lam']
            self.name = dictionary_data['name']

        self.__saver = tf.train.import_meta_graph(path + ".meta")

        sess = tf.Session()
        self.__saver.restore(sess, path)
        self.__sess = sess
