package com.es.ml.regression

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.AFTSurvivalRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
/**
  * Created by zhangw on 2017/12/19.
  * SurvivalRegression 生存回归
  */
object SurvivalRegressionTrain {
  /** 命令行参数 */
  case class Params(train_data: String = "", //训练数据路径
                    model_out: String = "",  //模型保存路径
                    appname: String = "SurvivalRegression_Train",
                    quantile_probabilities_1: Double = 0.3 , //
                    quantile_probabilities_2: Double = 0.6 , //
                    quantiles_col:String="quantiles",
                    delimiter:String=","//分隔符
                   )
  def main(args: Array[String]) {
    if (args.length < 6) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("SurvivalRegression_Train") {
      head("SurvivalRegression_Train: 生存回归训练.")
      opt[String]("train_data")
        .required()
        .text("训练数据路径")
        .action((x, c) => c.copy(train_data = x))
      opt[String]("model_out")
        .required()
        .text("模型保存路径")
        .action((x, c) => c.copy(model_out = x))
      opt[String]("appname")
        .required()
        .text("appname")
        .action((x, c) => c.copy(appname = x))
      opt[Double]("quantile_probabilities_1")
        .required()
        .text("quantile_probabilities_1")
        .action((x, c) => c.copy(quantile_probabilities_1 = x))
      opt[Double]("quantile_probabilities_2")
        .required()
        .text("quantile_probabilities_2")
        .action((x, c) => c.copy(quantile_probabilities_2 = x))
      opt[String]("quantiles_col")
        .required()
        .text("quantiles_col")
        .action((x, c) => c.copy(quantiles_col = x))
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

    val inputRDD = sc.textFile(p.train_data)

    val inputRDDTuple =  inputRDD.map(line => {
      val arr=line.split(p.delimiter).map(_.toDouble)
      val vec:Array[Double]= new Array(arr.length-2)
      Array.copy(arr,2,vec,0,arr.length-2)
      (arr(0),arr(1),Vectors.dense(vec))
    })

    val training=spark.createDataFrame(inputRDDTuple).toDF("label", "censor", "features")

    val quantileProbabilities = Array(p.quantile_probabilities_1, p.quantile_probabilities_2)
    val aft = new AFTSurvivalRegression()
      .setQuantileProbabilities(quantileProbabilities)
      .setQuantilesCol(p.quantiles_col)

    // Fit the model
    val model = aft.fit(training)

    // Print the coefficients, intercept and scale parameter for AFT survival regression
    println(s"Coefficients: ${model.coefficients}")
    println(s"Intercept: ${model.intercept}")
    println(s"Scale: ${model.scale}")
    model.transform(training).show(false)

    model.save(p.model_out) //保存模型
    sc.stop()
  }
}
