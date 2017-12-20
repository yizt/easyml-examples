package com.es.preprocess

import com.es.util.DataFrameUtil
import com.huaban.analysis.jieba.{JiebaSegmenter}
import com.huaban.analysis.jieba.JiebaSegmenter.SegMode
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by mick.yi on 2017/12/20.
  * 中文分词(目前用jieba)
  */
object WordSegment {

  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,parquet格式
                    output: String = "", //输出数据,parquet格式
                    inputCol: String = "", //分词列
                    outputCol: String = "", //分词结果输出列
                    resultCols: String = "", //输出结果保留的列,默认全部输出
                    appName: String = "Tokenizer"
                   )

  def main(args: Array[String]) {
    if (args.length < 5) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("Tokenizer") {
      head("Tokenizer:.")
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
        .text("分词列")
        .action((x, c) => c.copy(inputCol = x))
      opt[String]("outputCol")
        .required()
        .text("分词结果输出列")
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

    //注册自定义函数分词
    import org.apache.spark.sql.functions._
    sqlContext.udf.register("segment", (text: String) => tokenize(text))
    val outputDF = inputDF.withColumn(p.outputCol, callUDF("segment", col(p.inputCol)))

    //保存结果
    val resultDF = DataFrameUtil.select(outputDF, p.resultCols) //只保存选择的列
    resultDF.write.parquet(p.output)

    sc.stop()
  }

  //jieba分词
  private lazy val jiebaSeg = new JiebaSegmenter()

  def tokenize(text: String): String = {
    val toks = jiebaSeg.process(text, SegMode.INDEX);
    import scala.collection.JavaConversions._
    toks.map(_.word).mkString(" ")
  }

}
