# !/usr/bin/python
# -*- coding: utf-8 -*-


import unittest
import random
import pprint
import pandas as pd
from ...jano import Jano
from datetime import datetime

class Test_Jano(unittest.TestCase):
    """
        Test_Jano Class tests all possible combination of walk, and walks
        given a defined dataset. This class tests dataframes limits and
        data partitions that can not be made given a defined mask.
    """

    @staticmethod
    def generate_dataframe(size):
        """ Generates a dataframe from a defined size with a
            single row for each value."""
        n = size
        today = datetime.now()
        dataframe = pd.DataFrame({'date': [pd.to_datetime(Jano.move(today, x)) for x in range(n)],
                                  'TARGET':   [random.randrange(0, 50) for x in range(0, n)],
                                  'DATE_ATTRIB': [random.randrange(0, 50) for x in range(0, n)],
                                  'DATE_ATTRIB': [random.randrange(0, 50) for x in range(0, n)]
                                  })
        return dataframe

    def setUp(self):
        self.dataframe =  Test_Jano.generate_dataframe(size=1000)
        self.jano = Jano(self.dataframe)
        self.train_days = 1
        self.gap = 1
        self.test_days = 1
        self.target  = 'TARGET'
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
        """ Test the defined mask for a given Jano instance."""

        attributes = ['dataframe', 'train_date_attrib', 'test_date_attrib',
                      'train_days', 'gap', 'test_days', 'target', 'dataframe_min_day',
                      'dataframe_max_day']

        for attrib in attributes:
            assert(hasattr(self.jano, attrib) == True)

        assert(self.jano.train_days ==self.train_days)
        assert(self.jano.gap == self.gap)
        assert(self.jano.test_days == self.test_days)
        assert(self.jano.target == self.target)
        assert(self.jano.train_date_attrib == self.train_date_attrib)
        assert(self.jano.test_date_attrib == self.test_date_attrib)
        assert(self.jano.dataframe_min_day == self.dataframe['date'].min())
        assert(self.jano.dataframe_max_day == self.dataframe['date'].max())

    def test_total_mask_attribs(self):
        """ Test total lenght of mask attributes"""
        assert(len(self.jano.__dict__.keys()) == 9)

    def test_walk_one(self):
        """ Test all combination for one walk given a defined dataframe."""
        print('Test walk_one...')

        X_train, X_test, y_train, y_test = self.jano.walk_one(begin=self.begin,
                                                              shift=self.shift)

        pprint.pprint(self.jano.summary())

        # Testeamos que tenga una sola iteracion:
        assert(self.jano.iteration == 1)

        if self.begin == 1:
            # Testeamos que corresponda el start_date con la primer fecha del dataframe:
            # print(jano.train_start_date, dataframe['date'].min())
            pass
        elif self.begin == 0:
            assert(self.jano.train_start_date == self.jano.move(self.jano.train_start_date, self.begin))
        else:
            pass

        # Testeamos que la cantidad de datos de testeo y entrenamiento no sea 0:
        assert(self.jano.X_train_len != 0)
        assert(self.jano.X_test_len != 0)
        # Testeamos que exista la misma cantidad de casos entre X_train e y_train:
        assert(self.jano.X_train_len == self.jano.y_train_len)
        # Testeamos que exista la misma cantidad de casos entre X_test e y_test:
        assert(self.jano.X_test_len == self.jano.y_test_len)

    def test_walk(self):
        """
            Test all possible combinations given a dataset, warnings and errors.
        """

        aux = 0

        print('Test walk...')
        for iteration in self.iterations:
            for X_train, X_test, y_train, y_test in self.jano.walk(begin=self.begin,
                                                                   iterations=iteration,
                                                                   shift=self.shift):


                # Check iterations parameter:
                for frame_size in [self.jano.X_train_len,
                                   self.jano.X_test_len,
                                   self.jano.y_train_len,
                                   self.jano.y_test_len]:

                    assert (frame_size > 0)

            #assert (self.jano.summary()['iteration'] == aux)
                aux += 1

                # Check dates movement is ok:
                if self.begin == 0:
                    train_start_date = pd.to_datetime(self.jano.summary()['train_start_date'])
                    frame_train_start_date = pd.to_datetime(self.dataframe['date'].min())
                    train_end_date = self.jano.move(train_start_date, 1)
                    expected_train_end_date = self.jano.move(frame_train_start_date,self.begin)

                    #assert(train_end_date == expected_train_end_date)

                pprint.pprint(self.jano.summary())


if __name__ == '__main__':
    unittest.main()

