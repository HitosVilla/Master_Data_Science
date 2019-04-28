```python
%pylab inline
import pandas as pd

from importlib import reload # Reload Library
reload(pd)

import os
os.getcwd() #get current directory
```

# NUMPY
## Generate Random Data
```python
np.linspace(0, 19, 5) # 5 numbers between 0 and 19
np.arange(0,10,2)     # Even numbers between 0 and 10
np.random.normal(size=5) #5 random numbers from a Normal Distribution
np.random.randint(10, size=15) #15 natural numbers less than 10
np.random.shuffle(a) #Shuffle a list o maxtrix

import random
random.randrange(0, 100, 7) #return one number between 0 and 100 multiple of 7
```

## Strings
```python
sentence = 'abcdefghijklmnñopqrstuvwxyz'
sentence[2::3] #from second character until the end taking 3 by 3
sentence[::-1] #reverse
```

## Be careful when copying
```python
a = [2,3] ; b = [3,2]
a = b[:]   # different objects
a = b.copy # different objects
a = b      # reference the same object, if one is updated then the other is also updated
```


# PANDAS
## Dummies
Convert categorical variable into dummy/indicator variables
```python
def createDummies(df, var_name):
    dummy = pd.get_dummies(df[var_name], prefix=var_name)
    df = df.drop(var_name, axis = 1)
    df = pd.concat([df, dummy ], axis = 1)
    return df
```

##Prints
```python
name ='hitos'
birth = 1990
print('{} will reach 100 years in {}'.format(name, birth))
print(name, 'will reach 100 years in', birth + 100)
message = '%s will reach 100 years in %s' %(name,birth + 100)
print(message)
print(name + ' will reach 100 years in ' + str(birth + 100))
print('\t tabulator \n line break') 
print('-'*70) # print 70 times "-"
```

## Functions
```python
## Functions
def function_name(variable=(0,0)): # Default value (0,0)
    'Function Description, help(function_name) returns the text in here'
    return ....
    
fun1() or fun2() or fun3() = the first returned value different than None
```
