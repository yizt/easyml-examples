package com.es.ml.regression
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.AFTSurvivalRegression
import org.apache.spark.ml.regression.AFTSurvivalRegressionModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
/**
  * Created by zhangw on 2017/12/19.
  *   * SurvivalRegression 生存回归
  */
object SurvivalRegressionPredict {
  /** 命令行参数 */
  case class Params(test_data: String = "", //测试数据路径
                    model_path: String = "", //模型路径
                    predict_out: String = "", //预测结果保存路径
                    delimiter:String=",",//分隔符
                    appname: String = "SurvivalRegression_Predict"
                   )

  def main(args: Array[String]) {
    if (args.length < 4) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("SurvivalRegression_Predict") {
      head("SurvivalRegression_Predict:生存回归预测.")
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
      opt[String]("delimiter")
        .required()
        .text("分隔符")
        .action((x, c) => c.copy(delimiter = x))
    }
    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }

  }
  def run(p:Params): Unit = {
    val spark = SparkSession.builder.appName(p.appname).getOrCreate()
    val sc = spark.sparkContext

    import spark.implicits._
    val inputRDD = sc.textFile(p.test_data)

    val inputRDDTuple =  inputRDD.map(line => {
      val arr=line.split(p.delimiter).map(_.toDouble)
      val vec:Array[Double]= new Array(arr.length-2)
      Array.copy(arr,2,vec,0,arr.length-2)
      (arr(0),arr(1),Vectors.dense(vec))
    })

    val testdata=spark.createDataFrame(inputRDDTuple).toDF("label", "censor", "features")


    val model = AFTSurvivalRegressionModel.load(p.model_path) //加载模型

    //预测数据
    val result =  model.transform(testdata)
    val predictionAndLabels = result.select("prediction", "label").
      map(row => {
        val predict = row.getAs[Float]("prediction")
        val label = row.getAs[Double]("label")
        s"${predict} ${label}"
      })
    sc.stop()
  }
}
