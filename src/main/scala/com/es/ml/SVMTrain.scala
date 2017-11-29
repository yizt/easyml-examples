package com.es.ml

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by mick.yi on 2017/11/28.
  * svm 训练
  */
object SVMTrain {
  /** 命令行参数 */
  case class Params(train_data: String = "", //训练数据路径
                    model_out: String = "",  //模型保存路径
                    appname: String = "SVM_Train",
                    num_iterations: Int = 100  //训练迭代次数
                   )
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("Svm_Train") {
      head("Svm_Train: 支持向量机训练.")
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
      opt[Int]("num_iterations")
        .required()
        .text("迭代次数")
        .action((x, c) => c.copy(num_iterations = x))
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
    val model = SVMWithSGD.train(training, p.num_iterations) //训练
    model.save(sc,p.model_out) //保存模型
    sc.stop()
  }
}
