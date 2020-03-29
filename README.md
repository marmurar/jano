
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

In this example Jano uses 8 days to train, tests with 1 day and leaves 1 day as a gap from the end of the train until the start of the test period. 

### How to "walk" with Jano ?

Jano "walks" over dataframes slicing time and defining where train and test begins and ends. It walks along an user defined mask with the condition to iterate over time. Use jano.walk_one() method make only one slice and jano.walk() when 
