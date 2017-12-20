package com.es.ml.regression
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.AFTSurvivalRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
/**
  * Created by zhangw on 2017/12/19.
  * SurvivalRegression 生存回归
  */
object SurvivalRegressionTrain {
  /** 命令行参数 */
  case class Params(train_data: String = "", //训练数据路径
                    model_out: String = "",  //模型保存路径
                    appname: String = "SurvivalRegression_Train",
                    quantile_probabilities_1: Double = 100 , //训练迭代次数
                    quantile_probabilities_2: Double = 100 , //训练迭代次数
                    quantiles_col:String="quantiles"
                   )
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("SurvivalRegression_Train") {
      head("SurvivalRegression_Train: 生存回归训练.")
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
      opt[Double]("quantile_probabilities_1")
        .required()
        .text("quantile_probabilities_1")
        .action((x, c) => c.copy(quantile_probabilities_1 = x))
      opt[Double]("quantile_probabilities_2")
        .required()
        .text("quantile_probabilities_2")
        .action((x, c) => c.copy(quantile_probabilities_2 = x))
      opt[String]("quantiles_col")
        .required()
        .text("quantiles_col")
        .action((x, c) => c.copy(quantiles_col = x))
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

    /*val training = spark.createDataFrame(Seq(
      (1.218, 1.0, Vectors.dense(1.560, -0.605)),
      (2.949, 0.0, Vectors.dense(0.346, 2.158)),
      (3.627, 0.0, Vectors.dense(1.380, 0.231)),
      (0.273, 1.0, Vectors.dense(0.520, 1.151)),
      (4.199, 0.0, Vectors.dense(0.795, -0.226))
    )).toDF("label", "censor", "features")*/

    val quantileProbabilities = Array(p.quantile_probabilities_1, p.quantile_probabilities_2)
    val aft = new AFTSurvivalRegression()
      .setQuantileProbabilities(quantileProbabilities)
      .setQuantilesCol(p.quantiles_col)

    // Fit the model
    val model = aft.fit(training)

    // Print the coefficients, intercept and scale parameter for AFT survival regression
    println(s"Coefficients: ${model.coefficients}")
    println(s"Intercept: ${model.intercept}")
    println(s"Scale: ${model.scale}")
    model.transform(training).show(false)

    model.save(p.model_out) //保存模型
    sc.stop()
  }
}
