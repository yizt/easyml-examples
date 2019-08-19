package com.es.preprocess

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by zhangw on 2018/1/2.
  * 文本转为ArrayType的dataframe : Int，Array[String]
  */
object Text2ArrayDF {
  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,text格式
                    output: String = "", //输出数据,parquet格式
                    delemiter: String = " ", //列分隔符，默认空格
                    colNames: String = "", //列名,逗号分隔
                    appName: String = "Text2ArrayDF"
                   )

  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("Text2ArrayDF") {
      head("Text2ArrayDF:.")
      opt[String]("input")
        .required()
        .text("输入数据")
        .action((x, c) => c.copy(input = x))
      opt[String]("delemiter")
        .required()
        .text("列分隔符，默认空格")
        .action((x, c) => c.copy(delemiter = x))
      opt[String]("output")
        .required()
        .text("输出数据")
        .action((x, c) => c.copy(output = x))
      opt[String]("appName")
        .required()
        .text("appName")
        .action((x, c) => c.copy(appName = x))
      opt[String]("colNames")
        .required()
        .text("列名，逗号分隔")
        .action((x, c) => c.copy(colNames = x))
    }
    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }

  }

  def run(p: Params): Unit = {
    val spark = SparkSession.builder.appName(p.appName).getOrCreate()
    val sc = spark.sparkContext
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    val inputRDD = sc.textFile(p.input)
    val arr:RDD[Array[String]]=inputRDD.map(_.split(p.delemiter))

    val inputRDDTuple =  inputRDD.map(line => {
      val arr=line.split(p.delemiter)
      val vec:Array[String]= new Array(arr.length-1)
      Array.copy(arr,1,vec,0,arr.length-1)
      (arr(0).toInt,vec)
    })

    val cols=p.colNames.split(",")

    val inputDF=spark.createDataFrame(inputRDDTuple).toDF(cols(0), cols(1))

    //保存结果
    inputDF.write.parquet(p.output)
    sc.stop()
  }



}
