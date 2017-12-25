package com.es.preprocess

import com.es.util.DataFrameUtil
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types.{LongType, StructField}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by mick.yi on 2017/12/20.
  * 特征索引
  */
object FeatureIndex {

  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,parquet格式
                    featurePath: String = "", //特征保存路径
                    output: String = "", //输出数据,parquet格式
                    inputCol: String = "", //特征文本列
                    outputCol: String = "", //结果输出列
                    resultCols: String = "", //输出结果保留的列,默认全部输出
                    appName: String = "FeatureIndex"
                   )

  def main(args: Array[String]) {
    if (args.length < 5) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("FeatureIndex") {
      head("FeatureIndex:.")
      opt[String]("input")
        .required()
        .text("输入数据")
        .action((x, c) => c.copy(input = x))
      opt[String]("featurePath")
        .required()
        .text("特征保存路径")
        .action((x, c) => c.copy(featurePath = x))
      opt[String]("output")
        .required()
        .text("输出数据")
        .action((x, c) => c.copy(output = x))
      opt[String]("appName")
        .required()
        .text("appName")
        .action((x, c) => c.copy(appName = x))
      opt[String]("inputCol")
        .required()
        .text("分词列")
        .action((x, c) => c.copy(inputCol = x))
      opt[String]("outputCol")
        .required()
        .text("分词结果输出列")
        .action((x, c) => c.copy(outputCol = x))
      opt[String]("resultCols")
        .required()
        .text("输出结果保留的列")
        .action((x, c) => c.copy(resultCols = x))
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
    val inputDF = sqlContext.read.parquet(p.input)
    //加载特征文本
    val featureIdxMap = sc.textFile(p.featurePath).distinct().collect().
      zipWithIndex.toMap.
      map{case(word,idx)=>(word,idx+1)}  //改为1-基准索引

    //增加索引列id
    val newDF = if (!inputDF.schema.fieldNames.contains("id")) {
      val schema = inputDF.schema.add(StructField("id", LongType, true))
      val newRdd = inputDF.rdd.zipWithIndex().map { case (row, id) => {
        Row.merge(row, Row.fromTuple(Tuple1(id)))
      }
      }
      sqlContext.createDataFrame(newRdd, schema)
    } else
      inputDF

    //特征名转特征索引
    val dataDF = newDF.select("id", p.inputCol).map(row => {
      val rowIdx = row.getAs[Long]("id") //行号
      val text = row.getAs[String](p.inputCol)
      val newText = text.split(" ").map { feature => {
        val arr = feature.split(":")
        val featureName = arr(0)
        val feafureVal = if(arr.length<=1||"".equals(arr(1))) 0d else arr(1).toDouble
        val featureIndex = featureIdxMap.getOrElse(featureName, -1)
        (featureIndex , feafureVal)
      }
      }.filter{case(featureIndex , feafureVal)=>{
        featureIndex >= 0
      }}.sortBy(_._1).   //顺序
        map{case(featureIndex , feafureVal)=>{
        featureIndex + ":" + feafureVal //从 特征名:特征值 转为 特征索引号:特征值
      }}.
        mkString(" ")
      (rowIdx, newText)
    }).toDF("id", p.outputCol)


    //转换数据
    val outputDF = dataDF.join(newDF, "id")

    //保存结果
    val resultDF = DataFrameUtil.select(outputDF, p.resultCols) //只保存选择的列
    resultDF.write.parquet(p.output)

    sc.stop()
  }

}
