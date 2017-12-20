package com.es.ml.classifier

import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by zhangw on 2017/12/18.
  * NaiveBayes 训练
  */

object NaiveBayesTrain {
  /** 命令行参数 */
  case class Params(train_data: String = "", //训练数据路径
                    model_out: String = "",  //模型保存路径
                    appname: String = "NaiveBayes_Train",
                    lambda: Double = 1.0, //
                    model_type:String="multinomial"//模型类型
                   )
  def main(args: Array[String]) {
    if (args.length < 5) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("NaiveBayes_Train") {
      head("NaiveBayes_Train: 朴素贝叶斯训练.")
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
      opt[Double]("lambda")
        .required()
        .text("lambda")
        .action((x, c) => c.copy(lambda = x))
      opt[String]("model_type")
        .required()
        .text("模型类型")
        .action((x, c) => c.copy(model_type = x))
    }

    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }

  }

  def run(p:Params): Unit = {
    val conf = new SparkConf().setAppName(p.appname)
    val sc = new SparkContext(conf)

    val training = MLUtils.loadLibSVMFile(sc,p.train_data) //加载数据
    val model = NaiveBayes.train(training, lambda = p.lambda, modelType = p.model_type)

    model.save(sc,p.model_out) //保存模型
    sc.stop()
  }

}
