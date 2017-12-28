package com.es.ml.regression

import org.apache.spark.mllib.regression.{IsotonicRegression, IsotonicRegressionModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
/**
  * Created by zhangw on 2017/12/19.
  * IsotonicRegression 预测
  */
object IsotonicRegressionPredict {
  /** 命令行参数 */
  case class Params(test_data: String = "", //测试数据路径
                    model_path: String = "", //模型路径
                    predict_out: String = "", //预测结果保存路径
                    appname: String = "IsotonicRegression_Predict"
                   )

  def main(args: Array[String]) {
    if (args.length < 4) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("IsotonicRegression_Predict") {
      head("IsotonicRegression_Predict: 保序回归预测.")
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
  def run(p:Params): Unit = {
    val conf = new SparkConf().setAppName(p.appname)
    val sc = new SparkContext(conf)
    val model = IsotonicRegressionModel.load(sc, p.model_path) //加载模型
    val data = MLUtils.loadLibSVMFile(sc,p.test_data) //加载数据
    val testdata = data.map { labeledPoint =>
        (labeledPoint.label, labeledPoint.features(0), 1.0)
      }
    //预测数据
    val labelAndPreds = testdata.map { point =>
        val predictedLabel = model.predict(point._2)
      s"${predictedLabel} ${point._1}"
      }

    labelAndPreds.saveAsTextFile(p.predict_out)//保存预测结果
    sc.stop()
  }
}
