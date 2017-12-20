package com.es.preprocess

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
  case class Params(data: String = "", //数据
                    modelPath: String = "", //模型保存路径
                    appName: String = "Word2Vec"
                   )

  def main(args: Array[String]) {
    if (args.length < 6) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("Word2VecTrain") {
      head("Word2VecTrain:.")
      opt[String]("data")
        .required()
        .text("训练语料路径")
        .action((x, c) => c.copy(data = x))
      opt[String]("modelPath")
        .required()
        .text("模型保存路径")
        .action((x, c) => c.copy(modelPath = x))
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
    val sqlContext = new SQLContext(sc)


    val model = W2VModel.load(p.modelPath) //.load(sc,p.modelPath)

    model.getVectors
    //model.transform()

    sc.stop()
  }
}
