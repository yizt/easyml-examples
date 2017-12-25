package com.es.util

import org.apache.spark.sql.DataFrame

/**
  * Created by mick.yi on 2017/12/19.
  * 处理DataFrame
  */
object DataFrameUtil {
  /**
    *
    * @param inputDF    输入DataFrame
    * @param selectCols 选择的列，逗号分隔
    * @return
    */
  def select(inputDF: DataFrame, selectCols: String): DataFrame = {
    if (selectCols == null || "".equals(selectCols.trim))
      inputDF
    else {
      val cols = selectCols.split(",")
      val head = cols(0) //第一列
      val tail = cols.toList.tail //剩余的列
      if(cols.length<=1)
        inputDF.select(head)
      else
        inputDF.select(head, tail: _*)
    }
  }

}
