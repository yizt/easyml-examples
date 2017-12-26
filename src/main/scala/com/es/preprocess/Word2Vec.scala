package com.es.preprocess

import com.es.util.DataFrameUtil
import org.apache.spark.ml.feature.{Word2VecModel => W2VModel}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by mick.yi on 2017/12/18.
  * word2vec词向量模型
  */
object Word2Vec {

  /** 命令行参数 */
  case class Params(input: String = "", //输输入数据,parquet格式
                    modelPath: String = "", //模型保存路径
                    output: String = "", //输出数据,parquet格式
                    wordCol: String = "", // word列
                    vectorCol: String = "", // 词向量输出列
                    resultCols: String = "", //输出结果保留的列,默认全部输出
                    appName: String = "Word2Vec"
                   )

  def main(args: Array[String]) {
    if (args.length < 5) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("Word2Vec") {
      head("Word2Vec:.")
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
      opt[String]("wordCol")
        .required()
        .text("word列")
        .action((x, c) => c.copy(wordCol = x))
      opt[String]("vectorCol")
        .required()
        .text("词向量输出列")
        .action((x, c) => c.copy(vectorCol = x))
      opt[String]("resultCols")
        .required()
        .text("输出结果保留的列")
        .action((x, c) => c.copy(resultCols = x))
      opt[String]("modelPath")
        .required()
        .text("模型保存路径")
        .action((x, c) => c.copy(modelPath = x))

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

    //输入数据
    val inputDF = sqlContext.read.parquet(p.input)
    //word2vec模型
    val model = W2VModel.load(p.modelPath) //.load(sc,p.modelPath)
    //词向量
    val wordVectors = model.getVectors

    //左连接词向量
    val outputDF=inputDF.join(wordVectors,inputDF(p.wordCol) === wordVectors("word"),"left")

    //保存结果
    val cols=if (p.resultCols == null || "".equals(p.resultCols.trim))
      outputDF.schema.map(_.name) :+ p.vectorCol mkString ","  //去除wordVectors中的word列
    else
      p.resultCols

    val resultDF = DataFrameUtil.select(outputDF, cols) //只保存选择的列
    resultDF.write.parquet(p.output)

    sc.stop()
  }
}
