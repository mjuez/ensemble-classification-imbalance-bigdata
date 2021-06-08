/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ubu.admirable.util

import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.feature.{ VectorAssembler, StringIndexer }
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.DoubleType

object LibSVMReader {
  
  /**
    * Reads a data set from a libSVM file.
    *
    * @param path The path where the libSVM file is stored.
    * @param session The Spark session.
    * @return a Spark Dataset.
    */
  def libSVMToML(path: String, session: SparkSession): Dataset[_] = {
    
    val libsvmData = session
      .read
      .format("libsvm")
      .option("inferSchema", "true")
      .load(path)

    new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(libsvmData)
      .transform(libsvmData)
      .drop("label")
      .withColumnRenamed("indexedLabel", "label")
  }
  
}