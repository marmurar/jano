# !/usr/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
from datetime import timedelta
import warnings
import logging
import pandas as pd

class Jano:
    def __init__(self, dataframe, logs=False):
        """t
        Class Jano: A pandas dataframe time split iterator
        for train&test time-dependant predictive models.


        :param dataframe: A defined pandas dataframe.
        :param logs: bool flag defined logs behaviour.
        """

        if self.__check_object_param(dataframe):
            self.dataframe = dataframe

        if logs == True:
            logging.basicConfig(level=logging.INFO)
            logging.info('Logging...')
        else:
            pass
            # TODD: Desactivar el logger
    @staticmethod
    def move(date, days):
        """
            A staticmethod to go add days given a datetime.
            Example: move(monday, 1) -> Tuesday.
        """

        if not isinstance(date, datetime):
            raise ValueError('date must be type datetime')

        shifted_date = date + timedelta(days=days)

        return shifted_date

    def get_mask(self):
        """
            Get jano atirbutes for the defined mask.

            :return: A dict with defined atributes
        """

        return {
                'train_days': self.train_days,
                'gap': self.gap,
                'test_days': self.test_days,
                'target': self.target
                }

    def summary(self):
        """
            Summary information about time splitting.

            :return: A dict with jano atributes defining time splits.
        """
        results = {}
        results['dataframe_lenght'] =  len(self.dataframe)
        results['train_start_date'] = str(self.train_start_date)
        results['test_start_date']  = str(self.test_start_date)
        results['train_end_date']   = str(self.train_end_date)
        results['test_end_date']    = str(self.test_end_date)
        results['train_days']  = self.train_days
        results['gap'] = self.gap
        results['test_days'] =  self.test_days
        results['target'] = self.target
        results['iteration'] =  self.iteration
        results['dataframe_min_day'] = self.dataframe_min_day
        results['dataframe_max_day'] = self.dataframe_max_day

        return results

    def __check_attrib(self, attrib):
        """
            Checks that the defined attributes belong to the
            defined dataframe.

            :param attrib:
            :return:
        """

        try:
            attrib in self.dataframe.columns.tolist()
        except Exception as e:
            raise(e)

        return True

    def __check_object_param(self, dataframe):

        if type(dataframe) == pd.core.frame.DataFrame:
            return True
        raise TypeError('dataframe parameter type must be: pd.core.frame.DataFrame')

    def __check_dataframes_consistency(self, X_train, X_test, y_train, y_test):
        """
            Check if X,y over train and test dataframes have same lenght.

            :param X_train:
            :param X_test:
            :param y_train:
            :param y_test:
            :return: bool
        """

        frame_lenghts = {
                         'X_train': len(X_train),
                         'y_train': len(y_train),
                         'X_test':  len(X_test),
                         'y_train': len(y_test)
                         }

        for framelen in frame_lenghts:
            if frame_lenghts[framelen] == 0:
                raise Exception('Failed to apply mask, dataframe has no information')

        return True

    def __define_train_start_date(self, begin):
        """
            Defines the starting date from which we will start slicing our dataframe.

            Parameters
            ----------
            begin : str

        """
        # case: None
        if begin == None:
            self.train_start_date = self.dataframe_min_day
            logging.info('Train starts at date: ' + str(self.train_start_date))
        # case: int
        elif isinstance(begin, int) == True:
            if begin > 0:
                self.train_start_date = self.__class__.move(self.dataframe_min_day , days= begin)
                logging.info('__define_train_start_date: Train starts at date: ' + str(self.train_start_date))
            elif begin == 0:
                self.train_start_date = self.__class__.move(self.dataframe_min_day, days=begin)
            else:
                raise ValueError('begin value must be greater than 0')
        # case: datetime
        elif isinstance(begin, datetime) == True:
            #TODO: No funciona con datetime por el tema de los minutos
            if begin < self.dataframe_min_day:
                raise ValueError('Starting date is older that min date from dataframe')
            self.train_start_date = begin
            logging.info('__define_train_start_date: Train starts dat date: ' + str(self.train_start_date))
        else:
            raise ValueError('begin parameter type must be None, int or datetime')

    def __split_time(self, begin, shift):

        if begin == 0:
            self.train_start_date = self.dataframe_min_day
            self.train_end_date   = self.__class__.move(self.train_start_date,days=self.train_days)
            self.test_start_date  = self.__class__.move(self.train_end_date,days=self.gap)
            self.test_end_date    = self.__class__.move(self.test_start_date,days=self.test_days)
        else:
            self.train_start_date = self.__class__.move(self.dataframe_min_day,days=begin + shift)
            self.train_end_date   = self.__class__.move(self.train_start_date,days= self.train_days)
            self.test_start_date  = self.__class__.move(self.train_end_date,days=self.gap)
            self.test_end_date    = self.__class__.move(self.test_start_date,days=self.test_days)

    def __get_train_test_iterator(self, begin, iterations, shift):
        """
            Iterator method used to move dates along the dataframe
            splitting time with a defined mask.

            :param begin: the starting date.
            :param iterations:
            :param shift:
            :return: iterator
        """
        aux = 0
        while aux < iterations:
            self.iteration = aux

            if aux == 0:
                self.__define_train_start_date(begin)
                self.train_end_date   = self.__class__.move(self.train_start_date, days= self.train_days)
                self.test_start_date  = self.__class__.move(self.train_end_date,   days= self.gap)
                self.test_end_date    = self.__class__.move(self.test_start_date,  days= self.test_days)
                self.gap_day_start    = self.__class__.move(self.train_end_date,   days= 1)
                self.gap_day_end      = self.__class__.move(self.test_start_date,  days= 1)

                if self.test_end_date > self.dataframe_max_day:
                    warnings.warn('Last available date is ' + str(self.dataframe_max_day) + ' and test goes up to ' + str(self.test_end_date))
            else:
                self.train_start_date = self.__class__.move(self.train_start_date, days= 1 + shift)
                self.train_end_date   = self.__class__.move(self.train_start_date, days= self.train_days)
                self.test_start_date  = self.__class__.move(self.train_end_date,   days= self.gap)
                self.test_end_date    = self.__class__.move(self.test_start_date,  days= self.test_days)
                self.gap_day_start    = self.__class__.move(self.train_end_date,   days= 1)
                self.gap_day_end      = self.__class__.move(self.test_start_date,  days= 1)

                if self.test_end_date > self.dataframe_max_day:
                    warnings.warn('Last available date is ' + str(self.dataframe_max_day) + ' and test goes up to ' + str(self.test_end_date))

            yield self.__get_train_test_dataframes()

            aux +=1

    def __get_train_test_dataframes(self):
        """
            Similar to train_test_split returns four dataframes X_train, X_test, y_train, y_test
            spliting train and test from dates defined from Jano mask.

            :return: dataframe
        """

        # Filter mask:
        X_train = self.dataframe[(self.dataframe[self.train_date_attrib] >= self.train_start_date) &
                                 (self.dataframe[self.train_date_attrib] <= self.train_end_date)]
        # Define target for Train:
        y_train = X_train[self.target]
        # Filter target:
        X_train = X_train[X_train.columns.difference([self.target])]
        # Filter test mask:
        X_test  = self.dataframe[(self.dataframe[self.test_date_attrib] >= self.test_start_date) & (self.dataframe[self.test_date_attrib] <= self.test_end_date)]
        # Define target for y_test:
        y_test  = X_test[self.target]
        # Filter target for test set:
        X_test  = X_test[X_test.columns.difference([self.target])]

        self.X_train_len =  len(X_train)
        self.X_test_len  =  len(X_test)
        self.y_train_len =  len(y_train)
        self.y_test_len  =  len(y_test)

        if self.__check_dataframes_consistency(X_train, X_test, y_train, y_test):
            pass

        return X_train, X_test, y_train, y_test

    def __check_splitting_limit(self):
        """
         Warns if during any time split Jano reaches the limit of the
         defined dataframe.
        """

        if self.test_end_date > self.dataframe_max_day:
            warnings.warn('Last available date is ' + str(self.dataframe_max_day) + ' and test goes up to ' + str(
                self.test_end_date))

    def mask(self, train_days, gap, test_days, target, train_date_attrib=None, test_date_attrib=None):
        """
            A mask defines the rule for which Jano will slice time when walking.
            It basically consist on the quantity of days the class will use
            in each iteration.

            Parameters
            ----------
            train_days : str
                The name of the animal
            gap : str
                The sound the animal makes
            test_days : int, optional
                The number of legs the animal (default is 4)
            target : int, optional
                The number of legs the animal (default is 4)
            train_date_attrib : int, optional
                The number of legs the animal (default is 4)
        """
        if self.__check_attrib(train_date_attrib) and self.__check_attrib(test_date_attrib):
            self.train_date_attrib = train_date_attrib
            self.test_date_attrib  =  test_date_attrib
        # TODO: Redefinir los limites del dataframe segun al attrib por el cual cortamos train y test.
        self.train_days = train_days
        self.gap =  gap
        self.test_days =  test_days
        self.target = target
        self.dataframe_min_day = self.dataframe[self.train_date_attrib].min()
        self.dataframe_max_day = self.dataframe[self.train_date_attrib].max()

    def walk_one(self, begin, shift):
        """
            When Jano walks, splits time along a dataframe using a mask.
            This method is meant to be used whenever you want to iterate
            troughout several time splits.

            Parameters
            ----------
            begin : defines the beggining of the data set provided.
                    Can be defined as an int or a datetime. When
                    defined as 0 begin will start at the min day.
                    When defined as datetime, beign will start on
                    that given datetime.
            shift : defines how dates will be shifted within
                    each time iterations.
            iterations : defines the quantity of walks. How many iterations
                    of the given mask and step will Jano walk.
        """

        if not isinstance(shift, int):
            raise TypeError('shift parameter must be type int')
        if shift < 0:
            raise ValueError('shift parameter must be equal or greater than 0')
        if not isinstance(begin, int):
            raise TypeError('shift parameter must be type int')

        self.iteration = 1

        self.__define_train_start_date(begin)
        self.__split_time(begin, shift)
        self.__check_splitting_limit()

        return self.__get_train_test_dataframes()

    def walk(self, begin, iterations, shift):
        """
            When Jano walks, splits time along a dataframe using a mask.
            This method is meant to be used whenever you want to iterate
            one time only.

            Parameters
            ----------
            begin : defines the beggining of the data set provided.
                    Can be defined as an int or a datetime. When
                    defined as 0 begin will start at the min day.
                    When defined as datetime, beign will start on
                    that given datetime.
            shift : defines how dates will be shifted within
                    each time iterations.
            iterations : defines the quantity of walks. How many iterations
                    of the given mask and step will Jano walk.
        """

        if not isinstance(iterations, int):
            raise TypeError('iterations parameter must be type int')
        if iterations <= 0:
            raise ValueError('iterations parameter must be equal or greater than 0')
        if not isinstance(shift, int):
            raise TypeError('shift parameter must be type int')
        if shift < 0:
            raise ValueError('shift parameter must be equal or greater than 0')

        return self.__get_train_test_iterator(begin, iterations, shift)

