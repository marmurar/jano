
# Jano

*God of beginnins, time and trasitions... *

---------------------------------------------------------------

## What is Jano ?

__Jano is a time slicer designed to train and test time correlated machine learning models.__ Jano operates by "walking" along pandas dataframes with at least one time variable. Users can think of Jano as an iteration over a dtaframe of sklearn.model_selection.TimeSeriesSplit where a few features are addes such as: definning training size iteration over time, test size, a definen gap of time between train and test, etc...

__Jano was essentially designed to test how will a defined model will behaive over time based on your disposable trasactional data.__ On the other hand tryes to tackle some of the following questions: How much data should be used in train and test to make robust predictions over time ?When the model should be re trained ?, How long will the model maintain performance ?, Do distribution attributes change over time ?, Does my target distirbution change over time ?

##  What is a mask ?

A mask is defined by the users and simply defines how would you like to iterate over a defined dataframe, check this example: 


```python
 import pandas as pd

df = pd.DataFrame('date':['01-01-2020', '02-01-2020', '03-01-2020',
                          '04-01-2020', '05-01-2020', '06-01-2020',
                          '07-01-2020', '08-01-2020', '09-01-2020'],
                  'attrib':[9,4,2,3,4,5,6,1,2,4]
                  'target':[0,1,2,3,4,5,6,7,8,9])
```


```python
import jano as jano

jano = Jano(df)

# Define a jano mask:
jano.mask(train_days = 8, 
          gap = 1, 
          test_days = 1, 
          target = 'target', 
          train_date_attrib = 'date')
```

__In this example Jano uses 8 days to train, tests with 1 day and leaves 1 day as a gap from the end of the train until the start of the test period.__ If you want to iterate over a dataframe with the defined mask then you want to "walk" over a dataframe, check te following example...

!['basic walk usage'](jano)

## How to "walk" with Jano ?

__Jano "walks" over dataframes slicing time and defining where train and test begins and ends.__ It walks along an user defined mask with the condition to iterate over time. We'l the above dataframe to make a simple example training and testing with 2 and 1 day, leaving no days between train and test.


```python
# Re-define the mask:
jano.mask(train_days = 1, 
          gap = 0, 
          test_days = 1, 
          target = 'target', 
          train_date_attrib = 'date')
```


```python
for X_train, X_test, y_train, y_test in jano.walk(begin=0, iterations=3, shift=0):
    <train your model...>
    <predict...>
    <check results !>
```

For the above example we get four splitted dataframes(X_train, X_test, y_train, y_test) splitted from the first day of the dataframe (since parameter "begin" is 0 which is the first day) and each for iterarion will return a new dataframe.

__This is exactly what we did !__

!['basic walk usage'](jano_walk.gif)

---

### Author:

Marcos Manuel Muraro
