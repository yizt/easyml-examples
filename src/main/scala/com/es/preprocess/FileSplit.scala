package com.es.preprocess

import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by mick.yi on 2017/12/20.
  * 分割文件
  */
object FileSplit {

  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,text格式
                    outputA: String = "", //输出数据A,text格式
                    outputB: String = "", //输出数据B,text格式
                    ratio: Double = 0.5d, //分割比例
                    appName: String = "FileSplit"
                   )

  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("FileSplit") {
      head("FileSplit:.")
      opt[String]("input")
        .required()
        .text("输入数据")
        .action((x, c) => c.copy(input = x))
      opt[String]("outputA")
        .required()
        .text("输出数据A,text格式")
        .action((x, c) => c.copy(outputA = x))
      opt[String]("outputB")
        .required()
        .text("输出数据B,text格式")
        .action((x, c) => c.copy(outputB = x))
      opt[String]("appName")
        .required()
        .text("appName")
        .action((x, c) => c.copy(appName = x))
      opt[Double]("ratio")
        .required()
        .text("分割比例")
        .action((x, c) => c.copy(ratio = x))
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

    val input = sc.textFile(p.input)

    val outputA = input.sample(false, p.ratio)
    val outputB = input.subtract(outputA)
    //保存结果
    outputA.saveAsTextFile(p.outputA)
    outputB.saveAsTextFile(p.outputB)

    sc.stop()
  }

}
