#!/usr/bin/env python3

import numpy as np
import argparse
import itertools as it
import copy
import stump


class Boosting (object):

    def __init__ (self, *values, inc_negative = True, inc_positive = True):
        self.__available = [
            stump.Decision(col, val) for val in values
                for col in range(data.shape[1] - 1)
        ]

        if inc_negative:
            self.__available.append(stump.Negative())

        if inc_positive:
            self.__available.append(stump.Positive())

        self.__labels = []
        self.__weights = []
        self.__equation = []
        self.__data = []

    def feed (self, data, positive = stump.positive, negative = stump.negative):
        self.__data = data
        self.__labels = data[ :, -1 ] == positive
        self.__weights = np.tile(1.0 / data.shape[0], ( data.shape[0], 1 ))
        self.__equation = []

    def iterate (self):
        min_error = float('inf')
        min_stump = None
        min_errors = None

        for j, stump in enumerate(self.__available):
            errors = stump.fit(self.__data, self.__labels)
            error = self.__weights[errors].sum()

            if error < min_error:
                min_error = error
                min_stump = stump
                min_errors = errors

        alpha = 0.5 * np.log((1 - min_error) / min_error)

        if alpha == 0.0:
            return False

        self.__weights *= np.exp(-alpha * map_errors(min_errors))
        self.__weights /= self.__weights.sum()

        self.__equation.append(( alpha, min_stump ))

        return min_error, min_stump

    def evaluate (self, inp, positive = stump.positive, negative = stump.negative):
        result = np.zeros(( inp.shape[0], 1 ), np.float)

        for alpha, stump in self.__equation:
            result += stump.evaluate(inp, alpha, -alpha)

        return result >= 0

    def accuracy (self, test):
        result = self.evaluate(test[ :, : -1 ])
        return (result == (test[ :, -1 ] == stump.positive)).sum() / result.shape[0]

    def __deepcopy__ (self, _):
        result = Boosting()
        result.__available = copy.deepcopy(self.__available)
        result.__labels = copy.deepcopy(self.__labels)
        result.__weights = copy.deepcopy(self.__weights)
        result.__equation = copy.deepcopy(self.__equation)
        result.__data = copy.deepcopy(self.__data)
        return result

    def __str__ (self):
        return ' + '.join('{:.3f}({})'.format(a, s) for a, s in self.equation)

    @property
    def equation (self):
        return self.__equation

def name_map (name):
    if name == 'positive':
        name = stump.positive
    elif name == 'negative':
        name = stump.negative

    return name

def map_errors (errors):
    result = np.tile(1, errors.shape)
    result[errors] = -1
    return result

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('input', type = str)
    argparser.add_argument('output', type = str, nargs = '?',
                           default = 'output.txt')
    argparser.add_argument('-max-it', type = int, default = 1000)
    argparser.add_argument('-no-negative', action = 'store_false',
                           dest = 'negative')
    argparser.add_argument('-no-positive', action = 'store_false',
                           dest = 'positive')
    argparser.add_argument('-no-shuffle', action = 'store_false',
                           dest = 'shuffle')
    argparser.add_argument('-k', type = int, default = 5)

    args = argparser.parse_args()

    data = None

    with open(args.input, 'r') as file:
        data = np.matrix([
            list(map(name_map, line.strip().split(','))) for line in file
        ])

    if args.shuffle:
        data = data[np.random.permutation(data.shape[0])]

    chunks = []

    part = 0
    part_size = data.shape[0] / args.k

    for i in range(args.k):
        next_part = part + part_size if i < (args.k - 1) else data.shape[0]
        chunks.append(data[ int(part) : int(next_part) ])
        part = next_part

    print('chunk sizes', ' '.join(str(x.shape[0]) for x in chunks))

    values = np.unique(data[ :, : -1 ].reshape(-1).A1)

    results = []

    max_accuracy = []

    print_step = max(1, np.ceil(args.max_it / 10))

    try:
        for i, test_data in enumerate(chunks):

            max_accuracy.append([])

            data = np.empty(( 0, test_data.shape[1] ), test_data.dtype)

            for j, chunk in enumerate(chunks):
                if j != i:
                    data = np.vstack(( data, chunk ))

            boosting = Boosting(*values, inc_negative = args.negative,
                                inc_positive = args.positive)
            boosting.feed(data)
            j = 0

            while boosting.iterate():
                accuracy = boosting.accuracy(test_data)

                while len(max_accuracy[i]) < j:
                    max_accuracy[i].append(max_accuracy[i][-1])

                if len(max_accuracy[i]) == j:
                    max_accuracy[i].append(accuracy)

                elif accuracy > max_accuracy[i][j]:
                    max_accuracy[i][j] = accuracy

                if not (j % print_step):
                    print('chunk = {} / {} , iteration = {} / {} , '
                          'accuracy = {:.3f}'.format(i + 1, len(chunks), j,
                                                     args.max_it, accuracy))

                j += 1

                if j >= args.max_it:
                    break

    except KeyboardInterrupt:
        print()

    max_len = max(len(ma) for ma in max_accuracy)

    for ma in max_accuracy:
        ma.extend(ma[-1] for _ in range(max_len - len(ma)))

    max_accuracy = np.array(max_accuracy).max(0)

    with open(args.output, 'w') as file:
        print(' '.join(map(str, max_accuracy)), file = file)
        print('data saved to {}'.format(args.output))
