import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashSet
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator,BinaryClassificationEvaluator}

object predict {

      case class Data(loan_id:String,user_id:String,work_type:Int,employer_type:Int,industry:Int,house_exist:Int,house_loan_status:Int,censor_status:Int,marriage:Int,offsprings:Int,label:Int)
      def main(args: Array[String]) {
            if (args.length != 2) {
                  System.err.println("Usage: <data path> <report path>")
                  System.exit(1)
            }
            val conf = new SparkConf().setAppName("DefaultForecast")
            val sc = new SparkContext(conf)
            val spark=  SparkSession.builder().getOrCreate()
            import spark.implicits._
            val rdd = sc.textFile(args(0)).map(x=>x.split(",")).repartition(1)
            var data = rdd.map(x=>Data(x(0),x(1),x(2).toInt,x(3).toInt,x(4).toInt,x(5).toInt,x(6).toInt,x(7).toInt,x(8).toInt,x(9).toInt,x(10).toInt)).toDF()
            val assemble = new VectorAssembler().setInputCols(Array("work_type","employer_type","industry","house_exist", "house_loan_status","censor_status","marriage", "offsprings")).setOutputCol("features")
            data = assemble.transform(data)
            val Array(trainData,testData) = data.randomSplit(Array(0.8,0.2))
            val classifier: DecisionTreeClassifier = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features").setMaxBins(16).setImpurity("gini").setSeed(10)
            val dtcModel: DecisionTreeClassificationModel = classifier.fit(trainData)
            val treetrainPre = dtcModel.transform(trainData)
            // 预测分析
            val treetestPre = dtcModel.transform(testData)
            val acc = new MulticlassClassificationEvaluator().setMetricName("accuracy")
            val rtn = new ArrayBuffer[String]()
            rtn += "accuracy    :" + acc.evaluate(treetestPre)
            val rddrtn = sc.parallelize(rtn)
            sc.stop()

      }
}