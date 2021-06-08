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
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, DataFrame, Column}
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.param.shared.HasLabelCol
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.param.ParamValidators
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.ml.param.Param
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import scala.util.Random
import org.apache.spark.ml.feature.MinMaxScalerModel
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.param.ParamMap
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.util.LabelUtils
import org.apache.spark.ml.param.BooleanParam

class SMOTE(override val uid: String) extends Transformer
  with KNNParams with SMOTEParams with HasLabelCol 
  with HasFeaturesCol with HasSeed with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("smote"))

  def setK(value: Int): this.type = set(k, value)
  def setPercOver(value: Int): this.type = set(percOver, value)
  def setToMajo(value: Boolean): this.type = set(toMajo, value)
  def setSeed(value: Long): this.type = set(seed, value)

  override def transform(ds: Dataset[_]): DataFrame = {
    // el dataset tiene que venir normalizado
    val session = ds.sparkSession
    val rnd = new Random($(seed))

    val classCounts = LabelUtils.getLabelsInfo(ds, $(labelCol))

    val inputSize = ds.count

    val scalerModel = new MinMaxScaler()
      .setInputCol($(featuresCol))
      .setOutputCol("fs")
      .setMin(0).setMax(1)
      .fit(ds)

    val normDS = scalerModel
      .transform(ds)
      .drop($(featuresCol))
      .withColumnRenamed("fs", $(featuresCol))
      .cache

    val (minorityLabel, minoritySize) = classCounts(0)
    val (majorityLabel, majoritySize) = classCounts(1)

    val majorityDF = normDS.filter(col($(labelCol)) === majorityLabel).toDF
    val minorityDF = normDS.filter(col($(labelCol)) === minorityLabel).toDF
    normDS.unpersist

    val fraction = if ($(percOver) == 0)
      (majoritySize - minoritySize).toFloat / minoritySize.toFloat
    else
      $(percOver).toFloat / 100

    val creationFactor = fraction.ceil.toInt

    val df = if(!ds.columns.contains("neighbors")){
      if($(toMajo)){
        new KNN()
        .setTopTreeSize(majorityDF.count().toInt / 500)
        .setK($(k))
        .setAuxCols(Array($(featuresCol)))
        .fit(majorityDF)
        .transform(majorityDF)
      }else{
        new KNN()
        .setTopTreeSize(minorityDF.count().toInt / 500)
        .setK($(k))
        .setAuxCols(Array($(featuresCol)))
        .fit(minorityDF)
        .transform(minorityDF)
      }
    }else{
      if($(toMajo)){
        majorityDF
      }else{
        minorityDF
      }
    }
    
    val synthSamples = df.rdd.flatMap {
      case Row(label: Double, currentF: Vector, neighborsIter: Iterable[_]) =>
      val neighbors = neighborsIter.asInstanceOf[Iterable[Row]].toList
      (0 to (creationFactor - 1)).map{ case(_) =>
        val randomIndex = rnd.nextInt(neighbors.size)
        val neighbour = neighbors(randomIndex).get(0).asInstanceOf[Vector].asBreeze
        val currentFBreeze = currentF.asBreeze
        val difference = (neighbour - currentFBreeze) * rnd.nextDouble
        val synthSample = Vectors.fromBreeze(currentFBreeze + difference)
        Row.fromSeq(Seq(label, synthSample))
      }
    }.cache

    val sampleFrac = fraction / creationFactor
    val synthSamplesDF = session
      .createDataFrame(synthSamples, ds.select($(labelCol), $(featuresCol)).schema)
      .sample(false, sampleFrac, rnd.nextLong)
    ds.select($(labelCol), $(featuresCol)).toDF.union(denormalize(synthSamplesDF, scalerModel))
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

  /**
   * De-normalize the dataset previously normalized by a MinMaxScaler.
   * 
   * @param dataset Dataset to de-normalize.
   * @param minmax MinMaxScaler used previously to normalize.
   * @param originalMax Vector of maximum values per attribute.
   * @param originalMin Vector of minimum values per attribute.
   * @return De-normalized dataframe.
   */
  private def denormalize(df: DataFrame, scaler: MinMaxScalerModel): DataFrame = {

    val numFeatures = scaler.originalMax.size
    val scale = 1

    // transformed value for constant cols
    val minArray = scaler.originalMin.toArray

    val scaleArray = Array.tabulate(numFeatures) { 
      i => val range = scaler.originalMax(i) - scaler.originalMin(i)
      // scaleArray(i) == 0 iff i-th col is constant (range == 0)
      if (range != 0) range else 0.0
    }

    val transformer = udf { vector: Vector =>
      // 0 in sparse vector will probably be rescaled to non-zero
      val values = vector.toArray
      var i = 0
      while (i < numFeatures) {
        if (!values(i).isNaN) {
          if (scaleArray(i) != 0) {
            values(i) = values(i) * scaleArray(i) + minArray(i)
          }
          else {
            // scaleArray(i) == 0 means i-th col is constant
            values(i) = scaler.originalMin(i)
          }
        }
        i += 1
      }
      Vectors.dense(values).compressed
    }

    // Denormalize the features column and overwrite it.
    df.withColumn($(featuresCol), transformer(col($(featuresCol))))
  }

}

trait KNNParams extends Params {
  
  /** @group param */
  final val k: IntParam = new IntParam(this, "k", 
    "Number of nearest neighbours, preferably an odd number.",
    ParamValidators.gtEq(3))

  setDefault(k -> 5)

  /** @group getParam */
  final def getK: Int = $(k)

}

trait SMOTEParams extends Params {
  
  /** @group param */
  final val percOver: IntParam = new IntParam(this, "percOver", 
    "Oversampling percentage.", ParamValidators.gtEq(0))

  /** @group param */
  final val toMajo: BooleanParam = new BooleanParam(this, "toMajo", 
    "If oversampling majority class.")

  setDefault(percOver -> 100, toMajo -> false)

  /** @group getParam */
  final def getPercOver: Int = $(percOver)

  /** @group getParam */
  final def getToMajo: Boolean = $(toMajo)

}