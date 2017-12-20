package com.es.ml.regression

import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
/**
  * Created by zhangw on 2017/12/19.
  * GeneralizedLinearRegression 训练 广义线性回归
  */
object GeneralizedLinearRegressionTrain {
  /** 命令行参数 */
  case class Params(train_data: String = "", //训练数据路径
                    model_out: String = "",  //模型保存路径
                    appname: String = "GeneralizedLinearRegression_Train",
                    family: String = "gaussian", //
                    link: String="identity",//
                    max_iter:Int=10,//迭代次数
                    reg_param:Double=0.3//
                   )
  def main(args: Array[String]) {
    if (args.length < 7) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("GeneralizedLinearRegression_Train") {
      head("GeneralizedLinearRegression_Train: 广义线性回归训练.")
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
      opt[String]("family")
        .required()
        .text("family")
        .action((x, c) => c.copy(family = x))
      opt[String]("link")
        .required()
        .text("link")
        .action((x, c) => c.copy(link = x))
      opt[Int]("max_iter")
        .required()
        .text("迭代次数")
        .action((x, c) => c.copy(max_iter = x))
      opt[Double]("reg_param")
        .required()
        .text("reg_param")
        .action((x, c) => c.copy(reg_param = x))
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

    val glr = new GeneralizedLinearRegression()
      .setFamily(p.family)
      .setLink(p.link)
      .setMaxIter(p.max_iter)
      .setRegParam(p.reg_param)

    // Fit the model
    val model = glr.fit(training)

    model.save(p.model_out) //保存模型
    sc.stop()
  }
}
