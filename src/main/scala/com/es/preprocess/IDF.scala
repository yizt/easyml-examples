package com.es.preprocess

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by mick.yi on 2017/12/20.
  * 自定义IDF统计,输出默认为 word idf两列,空格分隔，保存为文本格式
  */
object IDF {

  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,parquet格式
                    output: String = "", //输出数据,text格式
                    inputCol: String = "", //句子列
                    appName: String = "IDF"
                   )

  def main(args: Array[String]) {
    if (args.length < 5) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("IDF") {
      head("IDF:.")
      opt[String]("input")
        .required()
        .text("输入数据")
        .action((x, c) => c.copy(input = x))
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
        .text("句子列")
        .action((x, c) => c.copy(inputCol = x))
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
    import sqlContext.sparkSession.implicits._
    val textRDD = inputDF.select(p.inputCol).
      map(_.getAs[String](p.inputCol)).
      map(_.split(" "))
    textRDD.cache()

    val docNum = textRDD.count()

    val idfRDD = textRDD.
      flatMap(x => x).map((_, 1))
      .rdd.reduceByKey(_ + _).map { case (word, wordNum) => {
      $"${word} ${Math.log(docNum.toDouble / wordNum)}" //求idf值
    }
    }
    //保存idf结果
    idfRDD.saveAsTextFile(p.output)

    sc.stop()
  }
}
