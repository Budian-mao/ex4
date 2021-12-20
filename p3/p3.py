#第三题
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .enableHiveSupport().getOrCreate()

df=spark.read.options(header='True',inferSchema='True').csv("train_data.csv")

for i in range(0,40000,1000):
    print(str(i)+","+str(i+1000)+":"+str(df.filter(df.total_loan.between(i,i+1000)).count()))

import csv
import pandas

#第三题第一问
total_num=df.count()
df1 = df.groupBy('employer_type').count().toPandas()
df1["count"]=df1["count"]/total_num
df1.to_csv("3-1.csv",index=0,header=0,float_format="%.4f")

#第三题第二问
df1=df.withColumn("total_money",df.year_of_loan*df.monthly_payment*12-df.total_loan).select("user_id","total_money").toPandas().to_csv("3-2.csv",index=0,header=0,float_format="%.4f")

#第三题第三问
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
#1. 将dataframe里面的work_year转化为int型
def cal_work_year(work_year):
    if work_year == None:
        return 0
    elif '<' in work_year:
        return 1
    else:
        year = work_year.split(' ')[0]
        year = year.split('+')[0]
        return int(year)


# 自定义函数
udf_cal_work_year = udf(cal_work_year, IntegerType())
# 完成任务三
df3 = df.withColumn('new_work_year', udf_cal_work_year(df.work_year))
df3 = df3.select(df3.user_id, df3.censor_status, df3.new_work_year).filter(df3.new_work_year > 5).toPandas().to_csv("3-3.csv",index=0,header=0,float_format="%.4f")
