package com.es.preprocess


import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import scopt.OptionParser

/**
  * Created by mick.yi on 2017/12/18.
  * Word2Vec训练
  */
object Word2VecTrain {

  /** 命令行参数 */
  case class Params(corpus: String = "", //训练数据
                    modelPath: String = "", //模型保存路径
                    vectorSize: Int = 200, //词向量维度
                    windowSize: Int = 5, //窗口大小
                    maxIter: Int = 100, //最大迭代次数
                    stepSize: Float = 0.025f, //学习率
                    numPartitions: Int = 1, //分区数
                    minCount: Int = 5, //进入word2vec的最小词频数
                    maxSentenceLength: Int = 20, //句子最大长度
                    appName: String = "Word2VecTrain"

                   )

  def main(args: Array[String]) {
    if (args.length < 6) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("Word2VecTrain") {
      head("Word2VecTrain:.")
      opt[String]("corpus")
        .required()
        .text("训练语料路径")
        .action((x, c) => c.copy(corpus = x))
      opt[String]("modelPath")
        .required()
        .text("模型保存路径")
        .action((x, c) => c.copy(modelPath = x))
      opt[String]("appName")
        .required()
        .text("appName")
        .action((x, c) => c.copy(appName = x))
      opt[Int]("vectorSize")
        .required()
        .text("词向量维度")
        .action((x, c) => c.copy(vectorSize = x))
      opt[Int]("windowSize")
        .required()
        .text("窗口大小")
        .action((x, c) => c.copy(windowSize = x))

      opt[Int]("maxIter")
        .required()
        .text("迭代次数")
        .action((x, c) => c.copy(maxIter = x))
      opt[Float]("stepSize")
        .required()
        .text("学习率")
        .action((x, c) => c.copy(stepSize = x))
      opt[Int]("numPartitions")
        .required()
        .text("分区数")
        .action((x, c) => c.copy(numPartitions = x))
      opt[Int]("minCount")
        .required()
        .text("进入word2vec的最小词频数")
        .action((x, c) => c.copy(minCount = x))
      opt[Int]("maxSentenceLength")
        .required()
        .text("句子最大长度")
        .action((x, c) => c.copy(maxSentenceLength = x))
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
    import sqlContext.implicits._

    val data = sc.textFile(p.corpus).map(_.split(" ").toSeq).toDF("text")


    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(p.vectorSize)
      .setWindowSize(p.windowSize)
      .setMinCount(p.minCount)
      .setMaxIter(p.maxIter)
      .setNumPartitions(p.numPartitions)
      .setMaxSentenceLength(p.maxSentenceLength)

    val model = word2Vec.fit(data)

    model.save(p.modelPath)

    sc.stop()
  }

}
