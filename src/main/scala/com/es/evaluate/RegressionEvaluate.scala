package com.es.evaluate

import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import org.apache.spark.mllib.evaluation.RegressionMetrics
/**
  * Created by zhangw on 2017/12/26.
  * 回归模型评估
  */
object RegressionEvaluate{
  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,text格式
                    appName: String = "RegressionEvaluate"
                   )

  def main(args: Array[String]) {
    if (args.length < 1) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("RegressionEvaluate") {
      head("RegressionEvaluate:.")
      opt[String]("input")
        .required()
        .text("输入数据")
        .action((x, c) => c.copy(input = x))

      opt[String]("appName")
        .required()
        .text("appName")
        .action((x, c) => c.copy(appName = x))
    }
    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }

  }

  def run(p: Params): Unit = {
    val conf = new SparkConf().setAppName(p.appName)
    val sc = new SparkContext(conf)

    val input = sc.textFile(p.input).map(elem=>{
      val arrs=elem.split(" ")
      (arrs(0).toDouble,arrs(1).toDouble)
    })

    val metrics = new RegressionMetrics(input)

    // Squared error
    println(s"MSE = ${metrics.meanSquaredError}")
    println(s"RMSE = ${metrics.rootMeanSquaredError}")

    // R-squared
    println(s"R-squared = ${metrics.r2}")

    // Mean absolute error
    println(s"MAE = ${metrics.meanAbsoluteError}")

    // Explained variance
    println(s"Explained variance = ${metrics.explainedVariance}")

    sc.stop()
  }


}