package com.es.ml.regression
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by zhangw on 2017/12/18.
  * GradientBoostedTreesRegression 训练
  */
object GradientBoostedTreesRegressionTrain {
  /** 命令行参数 */
  case class Params(train_data: String = "", //训练数据路径
                    model_out: String = "",  //模型保存路径
                    appname: String = "GradientBoostedTreesRegression_Train",
                    default_params: String = "Classification", //默认参数
                    num_iterations:Int=3, //迭代次数
                    num_classes:Int=2, //类别数
                    max_depth:Int=5 //最大深度
                   )
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("GradientBoostedTreesRegression_Train") {
      head("GradientBoostedTreesRegression_Train: 梯度提升树回归训练.")
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
      opt[String]("default_params")
        .required()
        .text("默认参数")
        .action((x, c) => c.copy(default_params = x))
      opt[Int]("num_classes")
        .required()
        .text("类别数")
        .action((x, c) => c.copy(num_classes = x))
      opt[Int]("max_depth")
        .required()
        .text("最大深度")
        .action((x, c) => c.copy(max_depth = x))
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
    val boostingStrategy = BoostingStrategy.defaultParams(p.default_params)
     boostingStrategy.numIterations = p.num_iterations // Note: Use more iterations in practice.
     boostingStrategy.treeStrategy.maxDepth = p.max_depth
    // Empty categoricalFeaturesInfo indicates all features are continuous.
     boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val training = MLUtils.loadLibSVMFile(sc,p.train_data) //加载数据
    val model = GradientBoostedTrees.train(training, boostingStrategy)

    model.save(sc,p.model_out) //保存模型
    sc.stop()
  }
}
