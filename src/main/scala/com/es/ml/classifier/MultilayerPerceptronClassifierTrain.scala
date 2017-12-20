package com.es.ml.classifier

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
/**
  * Created by zhangw on 2017/12/19.
  * MultilayerPerceptronClassifier 	多层感知机
  */
object MultilayerPerceptronClassifierTrain {
  /** 命令行参数 */
  case class Params(train_data: String = "", //训练数据路径
                    model_out: String = "",  //模型保存路径
                    appname: String = "MultilayerPerceptronClassifier_Train",
                    max_iter: Int = 100,  //迭代次数
                    blocksize:Int=128,
                    seed:Long=1234L,
                    layers_1:Int=4,
                    layers_2:Int=5,
                    layers_3:Int=4,
                    layers_4:Int=3
                   )
  def main(args: Array[String]) {
    if (args.length < 6) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("MultilayerPerceptronClassifier_Train") {
      head("MultilayerPerceptronClassifier_Train: 广义线性回归训练.")
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
      opt[Int]("max_iter")
        .required()
        .text("迭代次数")
        .action((x, c) => c.copy(max_iter = x))
      opt[Int]("blocksize")
        .required()
        .text("块大小")
        .action((x, c) => c.copy(blocksize = x))
      opt[Int]("seed")
        .required()
        .text("种子")
        .action((x, c) => c.copy(seed = x))
      opt[Int]("layers_1")
        .required()
        .text("layers_1")
        .action((x, c) => c.copy(layers_1 = x))
      opt[Int]("layers_2")
        .required()
        .text("layers_2")
        .action((x, c) => c.copy(layers_2 = x))
      opt[Int]("layers_3")
        .required()
        .text("layers_3")
        .action((x, c) => c.copy(layers_3 = x))
      opt[Int]("layers_4")
        .required()
        .text("layers_4")
        .action((x, c) => c.copy(layers_4 = x))
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

    val training = spark.read.format("libsvm").load(p.train_data) //加载数据

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](p.layers_1, p.layers_2, p.layers_3, p.layers_4)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
        .setLayers(layers)
      .setBlockSize(p.blocksize)
      .setSeed(p.seed)
      .setMaxIter(p.max_iter)

    // train the model
    val model = trainer.fit(training)

    model.save(p.model_out) //保存模型
    sc.stop()
  }
}
