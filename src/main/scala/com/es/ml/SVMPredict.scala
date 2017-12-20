package com.es.ml

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by mick.yi on 2017/11/28.
  * svm预测
  */
object SVMPredict {

  /** 命令行参数 */
  case class Params(test_data: String = "", //测试数据路径
                    model_path: String = "", //模型路径
                    predict_out: String = "", //预测结果保存路径
                    appname: String = "Svm_Predict"
                   )

  def main(args: Array[String]) {
    if (args.length < 6) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("Svm_Predict") {
      head("Svm_Train: 支持向量机预测.")
      opt[String]("test_data")
        .required()
        .text("测试数据路径")
        .action((x, c) => c.copy(test_data = x))
      opt[String]("model_path")
        .required()
        .text("模型路径")
        .action((x, c) => c.copy(model_path = x))
      opt[String]("appname")
        .required()
        .text("appname")
        .action((x, c) => c.copy(appname = x))
      opt[String]("predict_out")
        .required()
        .text("预测结果保存路径")
        .action((x, c) => c.copy(predict_out = x))
    }
    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }

  }

  def run(p: Params): Unit = {
    val conf = new SparkConf().setAppName(p.appname)
    val sc = new SparkContext(conf)
    val model = SVMModel.load(sc, p.model_path) //加载模型
    val testdata = MLUtils.loadLibSVMFile(sc, p.test_data) //加载数据
    //预测数据
    val scoreAndLabels = testdata.map { point =>
        val score = model.predict(point.features)
        (score, point.label)
      }
    scoreAndLabels.saveAsTextFile(p.predict_out) //保存预测结果
    sc.stop()
  }
}
