from perceptron import Perceptron

train_link = 'Test data\\Train data.txt'
test_input_link = 'Test data\\Test input.txt'

net = Perceptron(2, 4, 1)
net.save('Test data\\For saving.txt')
net = Perceptron(link='Test data\\For saving.txt')
net.train(error=0.01, link=train_link)
print(net.work(link=test_input_link))
