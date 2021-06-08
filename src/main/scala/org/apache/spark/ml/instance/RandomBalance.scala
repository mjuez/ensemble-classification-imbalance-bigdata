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

package org.apache.spark.ml.instance

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.param.ParamValidators
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.sql.functions.rand
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.ml.param.shared.HasLabelCol
import org.apache.spark.sql.functions._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.util.LabelUtils
import scala.util.Random
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.knn.KNN

class RandomBalance(override val uid: String) extends Transformer
  with KNNParams with HasLabelCol with HasFeaturesCol 
  with HasSeed with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("rb2"))

  def setK(value: Int): this.type = set(k, value)
  def setSeed(value: Long): this.type = set(seed, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val rng = new Random($(seed))
    
    val classCounts = LabelUtils.getLabelsInfo(dataset, $(labelCol))

    require(classCounts.size == 2, 
      "Random Balance requires a binary classification data set")

    val (minorityLabel, minoritySize) = classCounts(0)
    val (majorityLabel, majoritySize) = classCounts(1)
    val size = minoritySize + majoritySize

    val newMajoritySize = rng.nextInt(size.toInt - 3) + 2.0 // random between 2 and size-2 (inclusive)
    val newMinoritySize = size - newMajoritySize

    if (newMajoritySize != majoritySize) {
      val newMajorityFrac = newMajoritySize / majoritySize.toDouble
      val newMinorityFrac = newMinoritySize / minoritySize.toDouble
      val majorityDF = dataset.filter(col($(labelCol)) === majorityLabel).toDF
      val minorityDF = dataset.filter(col($(labelCol)) === minorityLabel).toDF
      if (newMinorityFrac > 1){
        val newMajo = majorityDF
          .select($(labelCol), $(featuresCol))
          .sample(false, newMajorityFrac, rng.nextLong)
        val newMino = minorityDF
          .select($(labelCol), $(featuresCol))
          .sample(true, newMinorityFrac, rng.nextLong)
        newMajo.union(newMino)
        /*val newDF = newMajo.union(minorityDF.select($(labelCol), $(featuresCol))).cache
        val percOver = ((newMinorityFrac - 1) * 100).ceil.toInt
        val smote = new SMOTE()
          .setK($(k))
          .setPercOver(percOver)
          .setSeed(rng.nextLong)
          .setToMajo(false)
        val ranbalDF = smote.transform(newDF)
        newDF.unpersist
        ranbalDF*/
      }else{
        val newMino = minorityDF
          .select($(labelCol), $(featuresCol))
          .sample(false, newMinorityFrac, rng.nextLong)
        val newMajo = majorityDF
          .select($(labelCol), $(featuresCol))
          .sample(true, newMajorityFrac, rng.nextLong)
        newMino.union(newMajo)
        /*val newDF = newMino.union(majorityDF.select($(labelCol), $(featuresCol))).cache
        val percOver = ((newMajorityFrac - 1) * 100).ceil.toInt
        val smote = new SMOTE()
          .setK($(k))
          .setPercOver(percOver)
          .setSeed(rng.nextLong)
          .setToMajo(true)
        val ranbalDF = smote.transform(newDF)
        newDF.unpersist
        ranbalDF*/
      }
    }else{
      dataset.select($(labelCol), $(featuresCol)).toDF
    }
  }

  override def transformSchema(schema: StructType): StructType = schema

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  // private def resampleData(df: DataFrame, fraction: Double, isMajo: Boolean, rng: Random): DataFrame = {
  //   if(fraction > 1){
  //     val percOver = ((fraction - 1) * 100).ceil.toInt
  //       val smote = new SMOTE()
  //         .setK($(k))
  //         .setPercOver(percOver)
  //         .setSeed(rng.nextLong)
  //       smote.transform(df)
  //     // if(!isMajo){
  //     //   val percOver = ((fraction - 1) * 100).ceil.toInt
  //     //   val smote = new SMOTE()
  //     //     .setK($(k))
  //     //     .setPercOver(percOver)
  //     //     .setSeed(rng.nextLong)
  //     //   smote.transform(df)
  //     // }else{
  //     //   // basic oversampling
  //     //   df.select($(labelCol), $(featuresCol)).sample(true, fraction, rng.nextLong)
  //     // }
  //   }else{
  //     // basic undersampling
  //     df.select($(labelCol), $(featuresCol)).sample(false, fraction, rng.nextLong)
  //   }
  // }
}