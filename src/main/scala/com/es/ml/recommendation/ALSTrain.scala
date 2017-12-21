package com.es.ml.recommendation

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import org.apache.spark.sql.SparkSession
/**
  * Created by zhangw on 2017/12/19.
  * 协同过滤 Collaborative Filtering 训练
  */
object ALSTrain {
  /** 命令行参数 */
  case class Params(data: String = "", //测试数据路径
                    model_out: String = "", //结果保存路径
                    appname: String = "CollaborativeFiltering_Train",
                    max_iter:Int=5 ,//迭代次数
                    reg_param:Double=0.01,//
                    user_col:String="userId",//
                    item_col:String="movieId",//
                    rating_col:String="rating"//
                   )
  def main(args: Array[String]) {
    if (args.length < 8) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("CollaborativeFiltering_Train") {
      head("CollaborativeFiltering_Train: 协同过滤.")
      opt[String]("data")
        .required()
        .text("测试数据路径")
        .action((x, c) => c.copy(data = x))
      opt[String]("appname")
        .required()
        .text("appname")
        .action((x, c) => c.copy(appname = x))
      opt[String]("model_out")
        .required()
        .text("模型保存路径")
        .action((x, c) => c.copy(model_out = x))
      opt[Int]("max_iter")
        .required()
        .text("迭代次数")
        .action((x, c) => c.copy(max_iter = x))
      opt[Double]("reg_param")
        .required()
        .text("reg_param")
        .action((x, c) => c.copy(reg_param = x))
      opt[String]("user_col")
        .required()
        .text("user_col")
        .action((x, c) => c.copy(user_col = x))
      opt[String]("item_col")
        .required()
        .text("item_col")
        .action((x, c) => c.copy(item_col = x))
      opt[String]("rating_col")
        .required()
        .text("rating_col")
        .action((x, c) => c.copy(rating_col = x))
    }
    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }

  }

  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

  def run(p:Params): Unit = {
    val spark = SparkSession.builder.appName(p.appname).getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._
    // Load the edges as a graph
    val training = spark.read.textFile(p.data).map(parseRating).toDF()
    val als = new ALS()
      .setMaxIter(p.max_iter)
      .setRegParam(p.reg_param)
      .setUserCol(p.user_col)
      .setItemCol(p.item_col)
      .setRatingCol(p.rating_col)
    val model = als.fit(training)
    //保存模型
    model.save(p.model_out)
    sc.stop()
  }

}
