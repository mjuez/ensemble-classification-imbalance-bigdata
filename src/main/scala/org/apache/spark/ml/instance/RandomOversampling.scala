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
import org.apache.spark.ml.util.{Identifiable, DefaultParamsWritable}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.param.shared.HasLabelCol
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * Random Oversampling Transformer.
 * Oversamples the minority class the percentage of instances that is required.
 *
 * @author Álvar Arnaiz-González
 * @param uid  All types inheriting from `Identifiable` require a `uid`.
 *             This includes Transformers, Estimators, and Models.
 */
class RandomOversampling(override val uid: String) extends Transformer 
  with RandomOversamplingParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("rostr"))

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /** @group setParam */
  def setPercOver(value: Int): this.type = set(percOver, value)

  /**
   * Performs Random Oversampling on the minority class.
   * 
   * @param inputDS Input dataset to oversamples.
   */
  override def transform(inputDS: Dataset[_]): DataFrame = {
    import inputDS.sparkSession.implicits._

    // Find out the minority class.
    val classes = inputDS.groupBy(col($(labelCol))).count.sort(asc("count")).
                          collect.map(r => (r.getAs[Double](0), r.getAs[Long](1)))

    val (minClass, minNum) = classes(0)
    val (majClass, majNum) = classes(classes.size - 1)

    val minorityDF = inputDS.filter(col($(labelCol)) === minClass).toDF

    var fraction = 0.0
    // Perc. overs. equal to 0 means to balance minority and majority classes
    if ($(percOver) == 0)
      fraction = (majNum - minNum).toFloat / minNum.toFloat
    else
      fraction = $(percOver).toFloat / 100

    inputDS.toDF.union(minorityDF.sample(true, fraction, $(seed)))
  }

  /**
   * The schema of the output Dataset is the same as the input one.
   * 
   * @param schema Input schema.
   */
  override def transformSchema(schema: StructType): StructType = schema

  /**
   * Creates a copy of this instance.
   * 
   * @param extra  Param values which will overwrite Params in the copy.
   */
  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}

trait RandomOversamplingParams extends Params 
  with HasLabelCol with HasFeaturesCol with HasSeed {
  
  /** @group param */
  final val percOver: IntParam = new IntParam(this, "percOver", 
    "Oversampling percentage, 0 means balance the classes.",
    ParamValidators.gtEq(0))

  setDefault(percOver -> 100)

  /** @group getParam */
  final def getPercOver: Int = $(percOver)
}
