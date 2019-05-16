# SparkSQL and DataFrames

Dataset(data with type) vs DataFrame (rows of data)

__Advantage__
* custom memory management (project Tungsten),
* optimized execution plans (Catalyst optimizer)

### Important clasess

* __pyspark.sql.SparkSession__ Main entry point for DataFrame and SQL functionality.
```
from pyspark.sql import SparkSession
session = SparkSession.builder.getOrCreate()  # Create session
session.sparkContext.getConf().getAll()       # All config
session = SparkSession.builder.config('someoption.key','somevalue').getOrCreate() #Update config
```

* __pyspark.sql.DataFrame__ A distributed collection of data grouped into named columns.
```
df = session.createDataFrame(rows)
df = session.createDataFrame(rdd)
df = session.createDataFrame(zip(ids,positions),schema=[id,position])
df = session.read.csv('txtfile.csv') # read.json and read.parquet
df = session.sql('SELECT * FROM csv.`txtfile.csv` where _c3 = "..." ')

df.printSchema()

df.select('id').show()
df.filter(df['id']>5) or df.where(df['id']>5) 

df.take(5) # show rows
df.show(5) # show a table

df.crosstab('col1', 'col2') # pivot table

df.groupBy('col').mean('col2).show() 
df.groupBy('col').agg(functions.mean('col2').alias('pepe'),
                      functions.stddev_pop('col2')
                      
df.join(df1, on='id', how='left').show()      # By default how is = 'inner'
df.join(df1, on=df['id']==df1['id2']).show()  # column name is diferent in tables
df.join(df1, on=['id', 'location']).show()    # join by two columns
```

* __pyspark.sql.Column__ A column expression in a DataFrame.
```
# Adding columns, DataFrame is inmutable, so we create a new one
df2 = df.withColumn('anewcol',df['id']*10).withColumn('anewcol2',df['id']*10)
```
* __pyspark.sql.Row__ A row of data in a DataFrame.
```
from pyspark.sql import Row
rows = [Row(id=id_, position=postition_) for id_,postition_ in zip(ids,positions)]

```

* `pyspark.sql.GroupedData` Aggregation methods, returned by DataFrame.groupBy().
* `pyspark.sql.DataFrameNaFunctions` Methods for handling missing data (null values).
* `pyspark.sql.DataFrameStatFunctions` Methods for statistics functionality.
* __pyspark.sql.functions__ List of built-in functions available for DataFrame, each function expect a column.
```
from pyspark.sql import functions
df.select('id',functions.sqrt(df['id'])).show(3)          # Do Not use Python functions.

uds_log1p = functions.udf(math.log1p, types.FloatType())  # Create a User Define Fucntion to use Python functions
functions.udf(lambda x: x+x)

df3 = df.select('id','position',uds_log1p('id'))

# UDF are faster in Scala than in pySpark




* __pyspark.sql.types__ List of data types available.
```
from pyspark.sql import types
# Define schema
fields = [types.StructField('id', types.IntegerType()),types.StructField('position', types.StringType())]
my_schema = types.StructType(fields)
session.createDataFrame(zip(ids,positions),schema=my_schema)
```
* `pyspark.sql.Window` For working with window functions.


### Summary statistics

https://databricks.com/blog/2015/06/02/statistical-and-mathematical-functions-with-dataframes-in-spark.html



# RDDs
```Spark
my_rdd = sc.parallelize(input_list) # Distribution
my_rdd2 = my_rdd.map(funtion_name)  # Not compute
my_rdd2.collect()                   # Compute

my_rdd.take(3) # compute the 3 firts elements and show them
```

* __Transformations__ produce an RDD. map, filter, reduceByKey. sc.parallelize and sc.textFile
* __Actions__ are implemented as methods on an RDD, and return an object non RDD. reduce, collect, take, takeOrdered, and count.
```spark
my_first_rdd\
    .filter(funtion_name)\      # Trasformation
    .map(lambda n:1)\           # Trasformation
    .reduce(lambda a,b: a+b)    # Action
    
my_rdd.takeOrdered(10, lambda x: -x) # Reverse order
```
Persists
```spark
from pyspark import StorageLevel
StorageLevel.MEMORY_AND_DISK
non_s_cake.persist()
#usedisk, usememory, useOffheap, deserielized, replication 
non_s_cake.getStorageLevel()

# you can use also my_rdd.cache() to persists 
```
Partitions
```
my_rdd = sc.parallelize(input_list, num_partitions) 
my_rdd.getNumPartitions()
my_rdd.glom().collect()   # Show how data is distributed on each partition
```
Pair RDDs
```
pair_rdd = my_rdd.map(lambda element: (element,1))
count_elements = pair_rdd.reduceByKey(lambda v1,v2: v1+v2)
count_elements.collect()

# if RDD is like (key, (value1,..., valuen)), you can work with mapValues
my_rdd.mapValues(lambda v: v[0]-v[n-1])
```
Text File
```
# Read file
s_lines = sc.textFile('txtfile_name.txt')

# lower case
words = s_lines.map(lambda line: line.lower())\
               .flatMap(lambda line: line.split()) # flatMap put all toguether in the same line and remove empty elements

# Remove empty words
words = s_lines.map(lambda line: line.lower())\
               .flatMap(lambda line: line.split())
               
               
```

