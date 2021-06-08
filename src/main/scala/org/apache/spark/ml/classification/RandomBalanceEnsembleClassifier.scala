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

package org.apache.spark.ml.classification

import org.apache.spark.annotation.Since
import org.apache.spark.ml.tree.RandomForestClassifierParams
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.util.LabelUtils
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.sql.functions._
import scala.util.Random
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.ml.param.ParamValidators
import org.apache.spark.ml.param.IntParam
import org.apache.spark.sql.Row
import org.apache.spark.ml.instance.KNNParams
import org.apache.spark.ml.feature.MinMaxScaler
import java.{util => ju}
import org.apache.spark.ml.instance.RandomBalance
import org.apache.spark.ml.feature.MinMaxScalerModel
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.knn.KNN

@Since("2.4.5")
class RandomBalanceEnsembleClassifier @Since("2.4.5") (
    @Since("2.4.5") override val uid: String)
  extends AbstractImbalanceClassifier with KNNParams {

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  @Since("2.4.5")
  def this() = this(Identifiable.randomUID("rbec"))

  override def copy(extra: ParamMap): RandomBalanceEnsembleClassifier = defaultCopy(extra)

  @Since("2.4.5")
  override def train(dataset: Dataset[_]): RandomForestClassificationModel = {
    val rand = new Random($(seed))
  
    val classCounts = LabelUtils.getLabelsInfo(dataset, $(labelCol))
    require(classCounts.size == 2, "Random Balance requires a binary classification data set")

    val numFeatures = dataset.select(col($(featuresCol))).first match {
      case Row(features: Vector) => features.size
    }

    val trees = (1 to $(numResamples)).flatMap( rs => {
      // in weka bagging does bootstrapping with replacement (true)
      val bootstrap = dataset.sample(true, $(subsamplingRate), rand.nextLong()).cache()
      val resampledData = resample(bootstrap, rand).toDF
      bootstrap.unpersist()

      val rfc = new RandomForestClassifier()
        .setLabelCol($(labelCol))
        .setFeaturesCol($(featuresCol))
        .setNumTrees($(numTrees))
        .setSubsamplingRate(1.0)
        .setMaxDepth($(maxDepth))
        .setMaxBins($(maxBins))
        .setMinInstancesPerNode($(minInstancesPerNode))
        .setMinInfoGain($(minInfoGain))
        .setCheckpointInterval($(checkpointInterval))
        .setSeed(rand.nextLong())
        .setMaxMemoryInMB($(maxMemoryInMB))
        .setCacheNodeIds($(cacheNodeIds))
        .setFeatureSubsetStrategy($(featureSubsetStrategy))
        .fit(resampledData)

      resampledData.unpersist()
      val dT = ju.Calendar.getInstance()
      System.out.println("[" + dT.getTime() + "] Trained resample: " + rs)
      rfc.trees
    }).toArray
    
    new RandomForestClassificationModel(trees, numFeatures, 2)
  }

  override def resample(dataset: Dataset[_], rand: Random): Dataset[_] = {
    val rb = new RandomBalance()
      .setK($(k))
      .setSeed(rand.nextLong())

    rb.transform(dataset)
  }

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

  private def unionWithDifferentCols(df1: DataFrame, df2: DataFrame): DataFrame = {
    val cols1 = df1.columns.toSet
    val cols2 = df2.columns.toSet
    val total = cols1 ++ cols2 // union

    def expr(myCols: Set[String], allCols: Set[String]) = {
      allCols.toList.map(x => x match {
        case x if myCols.contains(x) => col(x)
        case _ => lit(null).as(x)
      })
    }

    df1.select(expr(cols1, total):_*).union(df2.select(expr(cols2, total):_*))
  }

}