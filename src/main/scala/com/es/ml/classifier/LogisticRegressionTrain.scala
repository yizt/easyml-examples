package com.es.ml.classifier

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
/**
  * Created by zhangw on 2017/12/18.
  * LogisticRegression 训练
  */
object LogisticRegressionTrain {
  /** 命令行参数 */
  case class Params(train_data: String = "", //训练数据路径
                      model_out: String = "",  //模型保存路径
                    appname: String = "LogisticRegression_Train",
                   num_classes:Int=10//类别数
                   )
  def main(args: Array[String]) {
    if (args.length < 4) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("LogisticRegression_Train") {
      head("LogisticRegression_Train: 逻辑回归训练.")
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
      opt[Int]("num_classes")
        .required()
        .text("类别数")
        .action((x, c) => c.copy(num_classes = x))
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
    val model = new LogisticRegressionWithLBFGS()
        .setNumClasses(p.num_classes)
        .run(training)
    model.save(sc,p.model_out) //保存模型
    sc.stop()
  }
}
