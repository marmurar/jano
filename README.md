
# Jano

*God of beginnins, time and trasitions... *

---------------------------------------------------------------

### What is Jano ?

__Jano is a pandas dataframe time slicer designed designed to train and test time dependant machine learning models.__ Basically it needs a dataframe with at
least one time variable to iterate over and slice time in several dataframes. Users can think of Jano as an extention of sklearn.model_selection.TimeSeriesSplit.
Jano tackles some of the following questions: 
    * How much data should i use in train and test to make robust predictions over time ? 
    * When should i re train the model ?
    * How longer can i manitain model performance over time ?
