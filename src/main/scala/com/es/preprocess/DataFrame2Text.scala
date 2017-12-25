package com.es.preprocess

import com.es.util.DataFrameUtil
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by mick.yi on 2017/12/20.
  * DataFrame转为Text
  */
object DataFrame2Text {

  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,parquet格式
                    output: String = "", //输出数据,text格式
                    delemiter: String = " ", //列分隔符，默认空格
                    resultCols: String = "", //输出结果保留的列,默认全部输出
                    appName: String = "DataFrame2Text"
                   )

  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("DataFrame2Text") {
      head("DataFrame2Text:.")
      opt[String]("input")
        .required()
        .text("输入数据")
        .action((x, c) => c.copy(input = x))
      opt[String]("delemiter")
        .required()
        .text("列分隔符，默认空格")
        .action((x, c) => c.copy(delemiter = x))
      opt[String]("output")
        .required()
        .text("输出数据")
        .action((x, c) => c.copy(output = x))
      opt[String]("appName")
        .required()
        .text("appName")
        .action((x, c) => c.copy(appName = x))
      opt[String]("resultCols")
        .required()
        .text("输出结果保留的列")
        .action((x, c) => c.copy(resultCols = x))
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
    val sqlContext = new SQLContext(sc)
    import sqlContext.sparkSession.implicits._
    val inputDF = sqlContext.read.parquet(p.input)

    //保存结果
    val resultDF = DataFrameUtil.select(inputDF, p.resultCols) //只保存选择的列
    resultDF.map(_.mkString(" ")).rdd.saveAsTextFile(p.output)
    sc.stop()
  }

}
