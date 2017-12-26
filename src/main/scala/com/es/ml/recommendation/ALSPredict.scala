package com.es.ml.recommendation

import com.es.ml.recommendation.ALSTrain.Rating
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import org.apache.spark.sql.SparkSession

/**
  * Created by zhangw on 2017/12/19.
  * 协同过滤 CollaborativeFiltering 预测
  */
object ALSPredict {

  /** 命令行参数 */
  case class Params(test_data: String = "", //测试数据路径
                    model_path: String = "", //模型路径
                    predict_out: String = "", //预测结果保存路径
                    appname: String = "CollaborativeFiltering_Predict"
                   )

  def main(args: Array[String]) {
    if (args.length < 4) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("CollaborativeFiltering_Predict") {
      head("CollaborativeFiltering_Predict: 协同过滤预测.")
      opt[String]("test_data")
        .required()
        .text("测试数据路径")
        .action((x, c) => c.copy(test_data = x))
      opt[String]("model_path")
        .required()
        .text("模型路径")
        .action((x, c) => c.copy(model_path = x))
      opt[String]("appname")
        .required()
        .text("appname")
        .action((x, c) => c.copy(appname = x))
      opt[String]("predict_out")
        .required()
        .text("预测结果保存路径")
        .action((x, c) => c.copy(predict_out = x))
    }
    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }

  }

  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)

  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

  def run(p: Params): Unit = {
    val spark = SparkSession.builder.appName(p.appname).getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._
    val model = ALSModel.load(p.model_path) //加载模型
    val testdata = spark.read.textFile(p.test_data).map(parseRating).toDF() //预测数据

    val result = model.transform(testdata)
    val predictionAndLabels = result.select("prediction", "rating").
      map(row => {
        val predict = row.getAs[Double]("prediction")
        val label = row.getAs[Double]("rating")
        s"${predict} ${label}"
      })

    predictionAndLabels.rdd.saveAsTextFile(p.predict_out) //保存预测结果
    sc.stop()
  }

}
