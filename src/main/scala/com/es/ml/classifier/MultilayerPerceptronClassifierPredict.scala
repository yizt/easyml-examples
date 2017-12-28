package com.es.ml.classifier

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
/**
  * Created by zhangw on 2017/12/19.
  * MultilayerPerceptronClassifier 	多层感知机
  */
object MultilayerPerceptronClassifierPredict {
  /** 命令行参数 */
  case class Params(test_data: String = "", //测试数据路径
                    model_path: String = "", //模型路径
                    predict_out: String = "", //预测结果保存路径
                    appname: String = "MultilayerPerceptronClassifier_Predict"
                   )

  def main(args: Array[String]) {
    if (args.length < 4) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("MultilayerPerceptronClassifier_Predict") {
      head("MultilayerPerceptronClassifier_Predict:生存回归预测.")
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
    val spark = SparkSession.builder.appName(p.appname).getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._
    val testdata = spark.read.format("libsvm").load(p.test_data) //加载数据

    val model = MultilayerPerceptronClassificationModel.load(p.model_path) //加载模型

    //预测数据
    val result =  model.transform(testdata)

    val predictionAndLabels = result.select("prediction", "label").
      map(row => {
        val predict = row.getAs[Float]("prediction")
        val label = row.getAs[Double]("label")
        s"${predict} ${label}"
      })
    predictionAndLabels.write.save(p.predict_out)//保存预测结果

    sc.stop()
  }

}
