package com.es.ml
import org.apache.spark.graphx.GraphLoader
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
/**
  * Created by zhangw on 2017/12/19.
  */
object PageRank {
  /** 命令行参数 */
  case class Params(data: String = "", //测试数据路径
                    data_2: String = "", //测试数据路径
                    output: String = "", //结果保存路径
                    tol:Double=0.0001,
                    delimiter:String=",",//分隔符
                    appname: String = "PageRank"
                   )
  def main(args: Array[String]) {
    if (args.length < 3) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("PageRank") {
      head("PageRank: PageRank.")
      opt[String]("data")
        .required()
        .text("测试数据路径")
        .action((x, c) => c.copy(data = x))
      opt[String]("data_2")
        .required()
        .text("测试数据路径")
        .action((x, c) => c.copy(data_2 = x))
      opt[String]("appname")
        .required()
        .text("appname")
        .action((x, c) => c.copy(appname = x))
      opt[String]("output")
        .required()
        .text("结果保存路径")
        .action((x, c) => c.copy(output = x))
      opt[String]("delimiter")
        .required()
        .text("分隔符")
        .action((x, c) => c.copy(delimiter = x))
      opt[Double]("tol")
        .required()
        .text("tol")
        .action((x, c) => c.copy(tol = x))
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
    // Load the edges as a graph
    val data = GraphLoader.edgeListFile(sc,p.data)
    // Run PageRank
    val ranks = data.pageRank(p.tol).vertices

    // Join the ranks with the usernames
    val users = sc.textFile(p.data_2).map { line =>
      val fields = line.split(p.delimiter)
      (fields(0).toLong, fields(1))
    }
    val ranksByUsername = users.join(ranks).map {
      case (id, (username, rank)) => (username, rank)
    }
    // Print the result
    println(ranksByUsername.collect().mkString("\n"))
    ranksByUsername.saveAsTextFile(p.output)
    //打印所有中心点
    sc.stop()
  }



}
