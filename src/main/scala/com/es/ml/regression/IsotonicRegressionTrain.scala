package com.es.ml.regression

import org.apache.spark.mllib.regression.{IsotonicRegression, IsotonicRegressionModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
/**
  * Created by zhangw on 2017/12/19.
  * IsotonicRegression 训练
  */
object IsotonicRegressionTrain {
  /** 命令行参数 */
  case class Params(train_data: String = "", //训练数据路径
                    model_out: String = "",  //模型保存路径
                    appname: String = "IsotonicRegression_Train",
                    isotonic: Boolean = true  //
                   )
  def main(args: Array[String]) {
    if (args.length < 4) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("IsotonicRegression_Train") {
      head("IsotonicRegression_Train: 保序回归训练.")
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
      opt[Boolean]("isotonic")
        .required()
        .text("isotonic")
        .action((x, c) => c.copy(isotonic = x))
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

    val data = MLUtils.loadLibSVMFile(sc,p.train_data).cache() //加载数据
    val training = data.map { labeledPoint =>
        (labeledPoint.label, labeledPoint.features(0), 1.0)
      }
    val model = new IsotonicRegression().setIsotonic(p.isotonic).run(training)

    model.save(sc,p.model_out) //保存模型
    sc.stop()
  }
}
