# !/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import random
import pprint
import pandas as pd
from jano.jano import Jano
from datetime import datetime

class Test_Jano(unittest.TestCase):
    """
        Test_Jano Class: tests all possible combination of walk, and walks
        given a defined dataset.
    """

    @staticmethod
    def generate_dataframe(size):
        """
            Generates a dataframe from a defined size with a
            single row for each value."
        """
        n = size
        today = datetime.now()
        dataframe = pd.DataFrame({'date':   [pd.to_datetime(Jano.move(today, x)) for x in range(n)],
                                  'target': [random.randrange(0, 50) for x in range(0, n)],
                                  'attrib': [random.randrange(0, 50) for x in range(0, n)]
                                  })
        return dataframe

    def setUp(self):
        self.dataframe =  Test_Jano.generate_dataframe(size=1000)
        self.jano = Jano(self.dataframe)
        self.train_days = 1
        self.gap = 1
        self.test_days = 1
        self.target  = 'target'
        self.train_date_attrib = 'date'
        self.test_date_attrib  = 'date'
        self.begin = 0
        self.shift = 1
        self.jano.mask(train_days=self.train_days,
                       gap=self.gap,
                       test_days=self.test_days,
                       target=self.target,
                       train_date_attrib=self.train_date_attrib,
                       test_date_attrib =self.test_date_attrib)
        self.iterations = [10, 100, 350]

    def test_mask_attribs(self):
        """
            Test the defined mask for a given Jano instance.
        """

        attributes = ['dataframe', 'train_date_attrib', 'test_date_attrib',
                      'train_days', 'gap', 'test_days', 'target', 'dataframe_min_day',
                      'dataframe_max_day']

        for attrib in attributes:
            assert(hasattr(self.jano, attrib) == True)

        assert(self.jano.train_days == self.train_days)
        assert(self.jano.gap == self.gap)
        assert(self.jano.test_days == self.test_days)
        assert(self.jano.target == self.target)
        assert(self.jano.train_date_attrib == self.train_date_attrib)
        assert(self.jano.test_date_attrib == self.test_date_attrib)
        assert(self.jano.dataframe_min_day == self.dataframe['date'].min())
        assert(self.jano.dataframe_max_day == self.dataframe['date'].max())

    def test_mask_attribs_lenght(self):
        """
            Test lenght of all mask attributes.
        """
        assert(len(self.jano.__dict__.keys()) == 9)

    def test_mask_attributes_types(self):
        """
            Test all of mask attribute types.
        """
        assert(type(self.jano.train_days) == int)
        assert(type(self.jano.gap) == int)
        assert(type(self.jano.test_days) == int)
        assert(type(self.jano.target) == str)
        assert(type(self.jano.train_date_attrib) == str)
        assert(type(self.jano.test_date_attrib) == str)
        assert(type(self.jano.dataframe_min_day) == type(self.dataframe['date'].min()))
        assert(type(self.jano.dataframe_max_day) == type(self.dataframe['date'].max()))

    def __test_dataframe_lenght(self):
        # Testeamos que la cantidad de datos de testeo y entrenamiento no sea 0:
        assert (self.jano.X_train_len != 0)
        assert (self.jano.X_test_len != 0)
        # Testeamos que exista la misma cantidad de casos entre X_train e y_train:
        assert (self.jano.X_train_len == self.jano.y_train_len)
        # Test1eamos que exista la misma cantidad de casos entre X_test e y_test:
        assert (self.jano.X_test_len == self.jano.y_test_len)

        # pprint.pprint(self.jano.summary())

    def test_begin_walk_one(self):
        """
            Test begin parameter on walk_one method
        """
        for begin in range(0,10):

            X_train, X_test, y_train, y_test = self.jano.walk_one(begin=begin,
                                                                  shift=self.shift)
            #print(X_train['date'].min(), self.dataframe['date'].min(), self.jano.move(self.dataframe['date'].min(), begin), begin)
            assert(X_train['date'].min() == self.jano.move(self.dataframe['date'].min(), begin))
            self.__test_dataframe_lenght()

    def test_walk(self):
        """
            Test all possible combinations given a dataset, warnings and errors.
        """
        aux = 0

        for iteration in self.iterations:
            for X_train, X_test, y_train, y_test in self.jano.walk(begin=self.begin,
                                                                   iterations=iteration,
                                                                   shift=self.shift):

                self.__test_dataframe_lenght()

                aux += 1

                # Check dates movement is ok:
                if self.begin == 0:
                    train_start_date = pd.to_datetime(self.jano.summary()['train_start_date'])
                    frame_train_start_date = pd.to_datetime(self.dataframe['date'].min())
                    train_end_date = self.jano.move(train_start_date, 1)
                    expected_train_end_date = self.jano.move(frame_train_start_date,self.begin)

                pprint.pprint(self.jano.summary())

if __name__ == '__main__':
    unittest.main()
