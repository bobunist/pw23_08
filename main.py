import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


def generate_ring_data(num_points):
    r1 = np.random.uniform(5, 10, num_points // 2)
    theta1 = np.random.uniform(0, 2 * np.pi, num_points // 2)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    r2 = np.random.uniform(0, 5, num_points // 2)
    theta2 = np.random.uniform(0, 2 * np.pi, num_points // 2)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2)

    x = np.vstack((np.concatenate((x1, x2)), np.concatenate((y1, y2)))).T
    y = np.array([0] * (num_points // 2) + [1] * (num_points // 2))

    return x, y


def generate_four_quadrant_data(num_points):
    x1 = np.random.uniform(-10, 0, num_points // 4)
    y1 = np.random.uniform(0, 10, num_points // 4)

    x2 = np.random.uniform(0, 10, num_points // 4)
    y2 = np.random.uniform(0, 10, num_points // 4)

    x3 = np.random.uniform(0, 10, num_points // 4)
    y3 = np.random.uniform(0, -10, num_points // 4)

    x4 = np.random.uniform(-10, 0, num_points // 4)
    y4 = np.random.uniform(-10, 0, num_points // 4)

    x = np.vstack((np.concatenate((x1, x2, x3, x4)), np.concatenate((y1, y2, y3, y4)))).T
    y = np.array([0] * (num_points // 4) + [1] * (num_points // 4) + [0] * (num_points // 4) + [1] * (num_points // 4))

    return x, y


def generate_parabola_data(num_points):
    x1 = 0.75 * np.random.uniform(-10, 10, num_points // 2)
    y1 = 0.33 * x1 ** 2 + np.random.normal(0, 1, num_points // 2) - 15
    x1 += 3

    x2 = 0.75 * np.random.uniform(-10, 10, num_points // 2)
    y2 = -0.33 * (x2 ** 2) + np.random.normal(0, 1, num_points // 2) + 15
    x2 -= 3

    x = np.vstack((np.concatenate((x1, x2)), np.concatenate((y1, y2)))).T
    y = np.array([0] * (num_points // 2) + [1] * (num_points // 2))

    return x, y


def plot_dataset(X, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='yellow', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='purple', label='Class 1')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()


data_sets = []
data_sets.append(generate_ring_data(np.random.randint(100, 200)))
data_sets.append(generate_four_quadrant_data(np.random.randint(100, 200)))
data_sets.append(generate_parabola_data(np.random.randint(100, 200)))


def split_data(x, y, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test


def train_test_knn(x_train, x_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    return y_pred, knn


def plot_accuracy(x_train, x_test, y_train, y_test, k_values):
    train_accuracy = []
    test_accuracy = []

    for k in k_values:
        y_pred, _ = train_test_knn(x_train, x_test, y_train, y_test, k)
        train_accuracy.append(accuracy_score(y_train, knn.predict(x_train)))
        test_accuracy.append(accuracy_score(y_test, y_pred))

    plt.plot(k_values, train_accuracy, label='Train Accuracy')
    plt.plot(k_values, test_accuracy, label='Test Accuracy')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. k')
    plt.show()


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, confusion


for i, dataset in enumerate(data_sets):
    x, y = dataset
    x_train, x_test, y_train, y_test = split_data(x, y, test_size=0.2)

    k_values = list(range(1, 9))
    print(f'Dataset {i + 1}')

    for k in k_values:
        y_pred, knn = train_test_knn(x_train, x_test, y_train, y_test, k)
        print(f'K={k}:')

        accuracy, precision, recall, f1, confusion = evaluate_model(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1-Score: {f1:.2f}')
        print('Confusion Matrix:')
        print(confusion)
        print('---')

    plot_accuracy(x_train, x_test, y_train, y_test, k_values)
