import pyspark.sql.types as typ
import pyspark.ml.classification as cl
import pyspark.ml.evaluation as ev
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark import SparkContext
from pyspark.sql import SQLContext
if __name__ == '__main__':
    sc = SparkContext.getOrCreate()
    spark = SQLContext(sc)
    spark_df = spark.read.format("csv").option('header', 'true').option('inferScheme', 'true').load("train_data.csv")
    spark_df = spark_df.drop('loan_id').drop('user_id')
    spark_df = spark_df.fillna("0")
    StrLabel=["class","sub_class","work_type","issue_date","employer_type","industry","earlies_credit_mon","work_year"]
    for item in StrLabel:
        indexer=StringIndexer(inputCol=item,outputCol="%sIndex"%item)
        spark_df=indexer.fit(spark_df).transform(spark_df)
        spark_df=spark_df.drop(item)
    for item in spark_df.columns:
        spark_df=spark_df.withColumn(item,spark_df[item].cast(typ.DoubleType()))

    cols=spark_df.columns
    cols.remove("is_default")
    ass = VectorAssembler(inputCols=cols, outputCol="features")
    spark_df=ass.transform(spark_df)

    model_df = spark_df.select(["features","is_default"])

    train_df,test_df=model_df.randomSplit([0.8,0.2],seed=541)
    #随机森林
    rf = cl.RandomForestClassifier(labelCol="is_default", numTrees=128, maxDepth=9).fit(train_df)
    res = rf.transform(test_df)
    rf_auc = ev.BinaryClassificationEvaluator(labelCol="is_default").evaluate(res)
    print("随机森林预测准确率为：%f"%rf_auc)
    # 逻辑回归
    log_reg = cl.LogisticRegression(labelCol='is_default').fit(train_df)
    res = log_reg.transform(test_df)
    log_reg_auc = ev.BinaryClassificationEvaluator(labelCol="is_default").evaluate(res)
    print("逻辑回归：%f" % log_reg_auc)

    # 决策树
    DTC = cl.DecisionTreeClassifier(labelCol='is_default').fit(train_df)
    res = DTC.transform(test_df)
    DTC_auc = ev.BinaryClassificationEvaluator(labelCol="is_default").evaluate(res)
    print("决策树：%f" % DTC_auc)
    # 支持向量机
    SVM = cl.LinearSVC(labelCol='is_default').fit(train_df)
    res = SVM.transform(test_df)
    SVM_auc = ev.BinaryClassificationEvaluator(labelCol="is_default").evaluate(res)
    print("支持向量机：%f" % SVM_auc)