package com.es.preprocess

import com.es.util.DataFrameUtil
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types.{StructType, StructField, LongType};

import scopt.OptionParser

/**
  * Created by mick.yi on 2017/12/20.
  * 普通的tfidf
  */
object WordTFIDF {

  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,parquet格式
                    idfPath: String = "", //词的逆文档词频保存路径
                    output: String = "", //输出数据,parquet格式
                    inputCol: String = "", //分词后文本列
                    outputCol: String = "", //结果输出列
                    resultCols: String = "", //输出结果保留的列,默认全部输出
                    appName: String = "WordTFIDF"
                   )

  def main(args: Array[String]) {
    if (args.length < 5) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("WordTFIDF") {
      head("WordTFIDF:.")
      opt[String]("input")
        .required()
        .text("输入数据")
        .action((x, c) => c.copy(input = x))
      opt[String]("idfPath")
        .required()
        .text("词的逆文档词频保存路径")
        .action((x, c) => c.copy(idfPath = x))
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
    //加载idf
    val idfDF = sc.textFile(p.idfPath).
      map(_.split(" ")).
      map(arr => (arr(0), arr(1).toDouble)).toDF("word", "idf")

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

    //列转行
    val dataRDD = newDF.select("id", p.inputCol).map(row => {
      val rowIdx = row.getAs[Long]("id") //行号
      val text = row.getAs[String](p.inputCol)
      text.split(" ").distinct. //去重
        zipWithIndex.map { case (word, colIdx) => {
        (word, colIdx, rowIdx) //词，列索引，行索引
      }
      }
    }).flatMap(x => x)

    //计算词频TF
    val tfDF = dataRDD.map { case (word, colIdx, rowIdx) => (word, 1) }.
      rdd.
      reduceByKey(_ + _).
      toDF("word", "tf")

    //计算tfidf
    val wordTfidfDF = dataRDD.toDF("word", "colIdx", "id").
      join(idfDF, "word").
      join(tfDF, "word").rdd.map(row => {
      val tf = row.getAs[Int]("tf").toDouble
      val idf = row.getAs[Double]("idf")
      val word = row.getAs[String]("word")
      val colIdx = row.getAs[Int]("colIdx")
      val rowIdx = row.getAs[Long]("id")
      val tfidf = tf * idf //tfidf值
      (rowIdx, List.apply((word + ":" + tfidf, colIdx)))
    }).reduceByKey(_ ::: _).
      map { case (id, list) => {
        val str = list.sortBy(_._2).map(_._1).mkString(" ") //保持原有的列顺序，空格合并所有的word:tfidf
        (id, str)
      }
      }.toDF("id", p.outputCol)

    //转换数据
    val outputDF = wordTfidfDF.join(newDF, "id")

    //保存结果
    val resultDF = DataFrameUtil.select(outputDF, p.resultCols) //只保存选择的列
    resultDF.write.parquet(p.output)

    sc.stop()
  }


}
