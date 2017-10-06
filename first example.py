from perceptron import Perceptron

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[1], [1], [0], [1]]

net = Perceptron(2, 4, 1)
net.train(x, y, epoch=1000)
for i, elem in enumerate(x):
    print(y[i][0], net.work(elem)[0])
