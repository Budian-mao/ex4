from pyspark import SparkContext
def map_func(x):
    s = x.split(",")
    total_loan = round(float(s[2]))
    intervalmin = (total_loan // 1000)*1000
    intervalmax = intervalmin + 1000
    interval = "("+str(intervalmin)+","+str(intervalmax)+")"
    return (interval,1)
sc=SparkContext()
lines = sc.textFile("train_data.csv").filter(lambda line: not line.startswith("loan_id")).map(lambda x:map_func(x)).cache()
result = lines.reduceByKey(lambda x,y:x+y).collect()
print(result)
sc.stop()