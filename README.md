
# Jano

*God of beginnins, time and trasitions... *

---------------------------------------------------------------

### What is Jano ?

__Jano is a pandas dataframe time slicer designed designed to train and test time dependant machine learning models.__ Basically it needs a dataframe with at
least one time variable to iterate over and slice time in several dataframes. Users can think of Jano as an extention of sklearn.model_selection.TimeSeriesSplit.
Jano tackles some of the following questions: 
    1. How much data should be used in train and test to make robust predictions over time ? 
    2. When the model should be re trained ?
    3. How long will the model maintain performance ?
    4. Do attributes distirbutions change over time ?
    5. Does my target change over time ?
    
    Etc...

### How to use Jano ?

Jano "walks" over dataframes slicing time and defining where train and test begins and ends. It walks along an user defined mask with the condition to iterate over time. Use jano.walk_one() method make only one slice and jano.walk() when 

![alt text](jano.jpg)
