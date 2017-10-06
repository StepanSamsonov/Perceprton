# -*- coding: utf-8 -*-
import random
import math
import numpy as np


class Perceptron():
    def __init__(self, size_of_input=None, size_of_middle=None, size_of_output=None, coefficient=1, link=None):
        """
        :param size_of_input: int, neurons number on the first layer
        :param size_of_middle: int, neurons number on the second layer
        :param size_of_output: int, neurons number on the third layer
        :param coefficient: float, gradient value will be multiplied by this coefficient
        :param link: directory link, if given, Perceptron will be initialized by data which is kept there (see 'save' method)
        """
        if link:
            try:
                text = open(link)
            except BaseException:
                raise Perceptron.ArgumentError('Link is not correct')
            try:
                self._size_of_input = int(text.readline())
                self._size_of_middle = int(text.readline())
                self._size_of_output = int(text.readline())
                self._coefficient = float(text.readline())
                self._weight_input = []
                for i in range(self._size_of_input):
                    line = str(text.readline()).split(" ")
                    newline = []
                    for elem in line:
                        newline.append(float(elem))
                    self._weight_input.append(newline)
                self._weight_output = []
                for i in range(self._size_of_middle):
                    line = str(text.readline()).split(" ")
                    newline = []
                    for elem in line:
                        newline.append(float(elem))
                    self._weight_output.append(newline)
                print("Completed")
            except BaseException:
                raise Perceptron.InvalidData('Invalid data')
        else:
            try:
                self._size_of_input = int(size_of_input)
                self._size_of_middle = int(size_of_middle)
                self._size_of_output = int(size_of_output)
                self._coefficient = float(coefficient)
            except BaseException:
                raise Perceptron.ArgumentError('Invalid argument')
            self._weight_input = []
            for i in range(self._size_of_input):
                line = []
                for j in range(self._size_of_middle):
                    line.append(random.random())
                self._weight_input.append(line)
            self._weight_output = []
            for i in range(self._size_of_middle):
                line = []
                for j in range(self._size_of_output):
                    line.append(random.random())
                self._weight_output.append(line)

    def save(self, link):
        """
        Save Perceptron data. It can be used in __init__ function.
        :param link: directory link
        """
        try:
            text = open(link, "w")
        except BaseException:
            raise Perceptron.ArgumentError('Link is not correct')
        data = str(self._size_of_input) + '\n' + str(self._size_of_middle) + '\n' + str(
            self._size_of_output) + '\n' + str(self._coefficient) + '\n'
        for line in self._weight_input:
            newline = []
            for elem in line:
                newline.append(str(elem))
            line = " ".join(newline)
            data += str(line) + '\n'
        for line in self._weight_output:
            newline = []
            for elem in line:
                newline.append(str(elem))
            line = " ".join(newline)
            data += str(line) + '\n'
        text.write(data)
        text.close()

    def _f(self, x):
        return 1 / (1 + math.e ** (-x))

    def _pf(self, x):
        if x > 350 or x < -350:
            return 0
        else:
            return (math.e ** x) / (((math.e ** x) + 1) ** 2)

    def _work_input(self, link):
        data = open(link)
        input = str(data.readline()).split(" ")
        middle = []
        for i in range(self._size_of_middle):
            value = 0
            for j in range(self._size_of_input):
                value += int(input[j]) * self._weight_input[j][i]
            middle.append(self._f(value))
        return middle

    def _work_input_without_link(self, input):
        middle = []
        for i in range(self._size_of_middle):
            value = 0
            for j in range(self._size_of_input):
                value += int(input[j]) * self._weight_input[j][i]
            middle.append(self._f(value))
        return middle

    def _work_output(self, middle):
        output = []
        for i in range(self._size_of_output):
            value = 0
            for j in range(self._size_of_middle):
                value += middle[j] * self._weight_output[j][i]
            output.append(self._f(value))
        return output

    def _work_input_with_p(self, input):
        middle = []
        for i in range(self._size_of_middle):
            value = 0
            for j in range(self._size_of_input):
                value += int(input[j]) * self._weight_input[j][i]
            middle.append(self._pf(value))
        return middle

    def _work_output_with_p(self, middle):
        output = []
        for i in range(self._size_of_output):
            value = 0
            for j in range(self._size_of_middle):
                value += middle[j] * self._weight_output[j][i]
            output.append(self._pf(value))
        return output

    def work(self, inp_array=None, link=None):
        """
        Perform work of Perceptron. Only one of the parameters should be given.
        :param inp_array: list, input array
        :param link: directory link
        :return: array-answer of Perceptron
        """
        if link:
            try:
                data = open(link)
            except BaseException:
                raise Perceptron.ArgumentError('Link is not correct')
            try:
                input = str(data.readline()).split(" ")
                middle = []
                for i in range(self._size_of_middle):
                    value = 0
                    for j in range(self._size_of_input):
                        value += (float(input[j])) * self._weight_input[j][i]
                    middle.append(self._f(value))
                output = []
                for i in range(self._size_of_output):
                    value = 0
                    for j in range(self._size_of_middle):
                        value += middle[j] * self._weight_output[j][i]
                    output.append(str(self._f(value)))
            except BaseException:
                raise Perceptron.InvalidData('Invalid data')
            return output
        elif inp_array:
            middle = []
            try:
                for i in range(self._size_of_middle):
                    value = 0
                    for j in range(self._size_of_input):
                        value += (inp_array[j]) * self._weight_input[j][i]
                    middle.append(self._f(value))
                output = []
                for i in range(self._size_of_output):
                    value = 0
                    for j in range(self._size_of_middle):
                        value += middle[j] * self._weight_output[j][i]
                    output.append(self._f(value))
            except BaseException:
                raise Perceptron.InvalidData('Invalid data')
            return output

    def train(self, input_arr=None, output_arr=None, error=None, epoch=None, link=None):
        """
        Perform learning of Perceptron.
        Only one parameter should be given from pairs:
        input_arr, output_arr or link
        error or epoch
        :param input_arr: list, input train data
        :param output_arr: list, output train data
        :param error: float, if error of Perceptron will be less than this number, learning will be aborted
        :param epoch: int, iteration number on training data
        :param link: directory link, file must have format n*(input example + '\n' + output example + '\n')
        :return: list of error on each epoch
        """
        if not (error or epoch):
            raise Perceptron.ArgumentError("'error' or 'epoch' should be given")
        if not link and not (input_arr and output_arr):
            raise Perceptron.ArgumentError("'input_arr' and 'output_arr' should be given")
        if link:
            try:
                text = open(link)
            except BaseException:
                raise Perceptron.ArgumentError('Invalid link')
            data = text.readlines()
            sample = []
            i = 0
            try:
                while i < len(data) - 1:
                    learn = []
                    elem1 = []
                    elem2 = []
                    line = str(data[i]).split(" ")
                    for elem in line:
                        elem1.append(float(elem))
                    line = str(data[i + 1]).split(" ")
                    for elem in line:
                        elem2.append(float(elem))
                    learn.append(elem1)
                    learn.append(elem2)
                    sample.append(learn)
                    i += 2
            except BaseException:
                raise Perceptron.InvalidData('Invalid data')
        elif input_arr and output_arr:
            sample = []
            try:
                for i, _ in enumerate(input_arr):
                    sample.append([input_arr[i], output_arr[i]])
            except BaseException:
                raise Perceptron.InvalidData('Arrays have different sizes')

        def go_epoch():
            correct = np.array([0.0] * self._size_of_output)
            for learn in sample:
                for i in range(1):
                    delta_input = []
                    delta_output = []
                    middle = self._work_input_without_link(learn[0])
                    middle_with_p = self._work_input_with_p(learn[0])
                    output = self._work_output(middle)
                    output_with_p = self._work_output_with_p(middle)
                    for i in range(self._size_of_middle):
                        line = []
                        for j in range(self._size_of_output):
                            delta = 2 * (output[j] - learn[1][j]) * output_with_p[j] * middle[i]
                            line.append(delta)
                        delta_output.append(line)
                    for i in range(self._size_of_input):
                        line = []
                        for j in range(self._size_of_middle):
                            delta = 0
                            for k in range(self._size_of_output):
                                delta += 2 * (output[k] - learn[1][k]) * output_with_p[k] * self._weight_output[j][k]
                            delta = delta * middle_with_p[j] * learn[0][i]
                            line.append(delta)
                        delta_input.append(line)
                    for i in range(self._size_of_input):
                        for j in range(self._size_of_middle):
                            self._weight_input[i][j] -= self._coefficient * delta_input[i][j]
                    for i in range(self._size_of_middle):
                        for j in range(self._size_of_output):
                            self._weight_output[i][j] -= self._coefficient * delta_output[i][j]
                correct += (np.array(output) - np.array(learn[1])) ** 2
            correct = correct / len(learn)
            global wrong
            wrong = 0
            for elem in correct:
                wrong += abs(elem)
            wrong = wrong / len(correct)
            return wrong

        i = 1
        wrong = list()
        wrong.append(go_epoch())
        if not error:
            error = 0
        while wrong[-1] > error:
            if i == epoch:
                break
            wrong.append(go_epoch())
            i += 1
        return wrong

    class ArgumentError(Exception):
        pass

    class InvalidData(Exception):
        pass
