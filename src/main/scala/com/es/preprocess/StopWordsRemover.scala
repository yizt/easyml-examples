package com.es.preprocess

import com.es.util.DataFrameUtil
import org.apache.spark.ml.feature.{StopWordsRemover}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by mick.yi on 2017/12/19.
  * 停用词去除
  */
object StopWordsRemover {

  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,parquet格式
                    stopWordPath: String = "", //停用词数据存放路径
                    output: String = "", //输出数据,parquet格式
                    inputCol: String = "", //处理列名
                    outputCol: String = "", //去除停用词后的结果保存列名
                    resultCols: String = "", //输出结果保留的列,默认全部输出
                    appName: String = "StopWordsRemover"
                   )

  def main(args: Array[String]) {
    if (args.length < 5) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("StopWordsRemover") {
      head("StopWordsRemover:.")
      opt[String]("input")
        .required()
        .text("输入数据")
        .action((x, c) => c.copy(input = x))
      opt[String]("stopWordPath")
        .required()
        .text("停用词数据存放路径")
        .action((x, c) => c.copy(stopWordPath = x))
      opt[String]("output")
        .required()
        .text("输出数据")
        .action((x, c) => c.copy(output = x))
      opt[String]("appName")
        .required()
        .text("appName")
        .action((x, c) => c.copy(appName = x))
      opt[String]("inputCol")
        .required()
        .text("处理列名")
        .action((x, c) => c.copy(inputCol = x))
      opt[String]("outputCol")
        .required()
        .text("去除停用词后的结果保存列名")
        .action((x, c) => c.copy(outputCol = x))
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

    val inputDF = sqlContext.read.parquet(p.input)
    //加载停用词(默认一个词一行
    val stopWords = sc.textFile(p.stopWordPath).collect()
    //定义分词器
    val remover = new StopWordsRemover().
      setInputCol(p.inputCol).
      setOutputCol(p.outputCol).
      setStopWords(stopWords)

    //转换数据
    val outputDF = remover.transform(inputDF)

    //保存结果
    val resultDF = DataFrameUtil.select(outputDF, p.resultCols) //只保存选择的列
    resultDF.write.parquet(p.output)

    sc.stop()
  }
}
