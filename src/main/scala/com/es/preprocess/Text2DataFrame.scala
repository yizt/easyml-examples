package com.es.preprocess

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by mick.yi on 2017/12/20.
  * 文本转dataframe
  */
object Text2DataFrame {

  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,text格式
                    output: String = "", //输出数据,parquet格式
                    delemiter: String = " ", //列分隔符，默认空格
                    colNames: String = "", //列名，逗号分隔
                    appName: String = "DataFrame2Text"
                   )

  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("Text2DataFrame") {
      head("Text2DataFrame:.")
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
    val conf = new SparkConf().setAppName(p.appName)
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    val inputRDD = sc.textFile(p.input)
    val colNames=p.colNames.split(",") //列名逗号分隔
    println(s"p.delemiter:${p.delemiter}")
    val inputRDDTuple =  inputRDD.map(line => {
      val colVals = strToFixLengthArray(line,p.delemiter,colNames.length)
      //数组转为tuple
      colVals match {
        case Array(a) => Tuple1(a)
        case Array(a, b) => (a, b)
        case Array(a, b, c) => (a, b, c)
        case Array(a, b, c, d) => (a, b, c, d)
        case Array(a, b, c, d, e) => (a, b, c, d, e)
        case Array(a, b, c, d, e, f) => (a, b, c, d, e, f)
      }
    })

    val inputDF=colNames.length match {
      case 1 => inputRDDTuple.asInstanceOf[RDD[Tuple1[String]]].toDF(colNames:_*)
      case 2 => inputRDDTuple.asInstanceOf[RDD[Tuple2[String,String]]].toDF(colNames:_*)
      case 3 => inputRDDTuple.asInstanceOf[RDD[Tuple3[String,String,String]]].toDF(colNames:_*)
      case 4 => inputRDDTuple.asInstanceOf[RDD[Tuple4[String,String,String,String]]].toDF(colNames:_*)
      case 5 => inputRDDTuple.asInstanceOf[RDD[Tuple5[String,String,String,String,String]]].toDF(colNames:_*)
      case 6 => inputRDDTuple.asInstanceOf[RDD[Tuple6[String,String,String,String,String,String]]].toDF(colNames:_*)
    }


    //保存结果
    inputDF.write.parquet(p.output)
    sc.stop()
  }

  /**
    * 将字符串转为固定的长度，多余的元素合并为最后一个元素
    * @param text
    * @param delemiter
    * @param arrLength
    * @return
    */
  private def strToFixLengthArray(text:String,delemiter:String,arrLength:Int):Array[String]={
    val arr=text.split(delemiter)
    if(arrLength==1)
      Array(text)
    else if(arr.length<=arrLength)
      arr
    else{
      val lastElem=for (i <- arrLength-1 until arr.length) yield arr(i)
      val preElems=for (i <- 0 until arrLength-1) yield arr(i)
      val res=preElems.:+(lastElem.mkString(delemiter))
      res.toArray
    }
  }
}
