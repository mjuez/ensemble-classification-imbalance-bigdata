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

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._

import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.ImbalanceGradientBoostedTrees
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.tree.model.{GradientBoostedTreesModel => OldGBTModel}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.param.ParamValidators

/**
 * Gradient-Boosted Trees (GBTs) (http://en.wikipedia.org/wiki/Gradient_boosting)
 * learning algorithm for classification.
 * It supports binary labels, as well as both continuous and categorical features.
 *
 * The implementation is based upon: J.H. Friedman. "Stochastic Gradient Boosting." 1999.
 *
 * Notes on Gradient Boosting vs. TreeBoost:
 *  - This implementation is for Stochastic Gradient Boosting, not for TreeBoost.
 *  - Both algorithms learn tree ensembles by minimizing loss functions.
 *  - TreeBoost (Friedman, 1999) additionally modifies the outputs at tree leaf nodes
 *    based on the loss function, whereas the original gradient boosting method does not.
 *  - We expect to implement TreeBoost in the future:
 *    [https://issues.apache.org/jira/browse/SPARK-4240]
 *
 * @note Multiclass labels are not currently supported.
 */
@Since("1.4.0")
class ImbalanceGBTClassifier @Since("1.4.0") (
    @Since("1.4.0") override val uid: String)
  extends ProbabilisticClassifier[Vector, ImbalanceGBTClassifier, GBTClassificationModel]
  with GBTClassifierParams with ImbalanceGBTClassifierParams with DefaultParamsWritable with Logging {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("gbtc"))

  // Override parameter setters from parent trait for Java API compatibility.

  // Parameters from TreeClassifierParams:

  /** @group setParam */
  @Since("1.4.0")
  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  @Since("1.4.0")
  def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  @Since("1.4.0")
  def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /** @group setParam */
  @Since("3.0.0")
  def setMinWeightFractionPerNode(value: Double): this.type = set(minWeightFractionPerNode, value)

  /** @group setParam */
  @Since("1.4.0")
  def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)

  /**
   * Specifies how often to checkpoint the cached node IDs.
   * E.g. 10 means that the cache will get checkpointed every 10 iterations.
   * This is only used if cacheNodeIds is true and if the checkpoint directory is set in
   * [[org.apache.spark.SparkContext]].
   * Must be at least 1.
   * (default = 10)
   * @group setParam
   */
  @Since("1.4.0")
  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /**
   * The impurity setting is ignored for GBT models.
   * Individual trees are built using impurity "Variance."
   *
   * @group setParam
   */
  @Since("1.4.0")
  def setImpurity(value: String): this.type = {
    logWarning("GBTClassifier.setImpurity should NOT be used")
    this
  }

  // Parameters from TreeEnsembleParams:

  /** @group setParam */
  @Since("1.4.0")
  def setSubsamplingRate(value: Double): this.type = set(subsamplingRate, value)

  /** @group setParam */
  @Since("1.4.0")
  def setSeed(value: Long): this.type = set(seed, value)

  // Parameters from GBTParams:

  /** @group setParam */
  @Since("1.4.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  @Since("1.4.0")
  def setStepSize(value: Double): this.type = set(stepSize, value)

  /** @group setParam */
  @Since("2.3.0")
  def setFeatureSubsetStrategy(value: String): this.type =
    set(featureSubsetStrategy, value)

  // Parameters from GBTClassifierParams:

  /** @group setParam */
  @Since("1.4.0")
  def setLossType(value: String): this.type = set(lossType, value)

  /** @group setParam */
  @Since("2.4.0")
  def setValidationIndicatorCol(value: String): this.type = {
    set(validationIndicatorCol, value)
  }

  /**
   * Sets the value of param [[weightCol]].
   * If this is not set or empty, we treat all instance weights as 1.0.
   * By default the weightCol is not set, so all instances have weight 1.0.
   *
   * @group setParam
   */
  @Since("3.0.0")
  def setWeightCol(value: String): this.type = set(weightCol, value)

  @Since("3.0.0")
  def setRandomUnderSampling(value: Boolean): this.type = set(randomUnderSampling, value)

  @Since("3.0.0")
  def setPercSampling(value: Int): this.type = set(percSampling, value)

  override protected def train(
      dataset: Dataset[_]): GBTClassificationModel = instrumented { instr =>
    val withValidation = isDefined(validationIndicatorCol) && $(validationIndicatorCol).nonEmpty

    val validateInstance = (instance: Instance) => {
      val label = instance.label
      require(label == 0 || label == 1, s"GBTClassifier was given" +
        s" dataset with invalid label $label.  Labels must be in {0,1}; note that" +
        s" GBTClassifier currently only supports binary classification.")
    }

    val (trainDataset, validationDataset) = if (withValidation) {
      (extractInstances(dataset.filter(not(col($(validationIndicatorCol)))), validateInstance),
        extractInstances(dataset.filter(col($(validationIndicatorCol))), validateInstance))
    } else {
      (extractInstances(dataset, validateInstance), null)
    }

    val numClasses = 2
    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, labelCol, weightCol, featuresCol, predictionCol, leafCol,
      impurity, lossType, maxDepth, maxBins, maxIter, maxMemoryInMB, minInfoGain,
      minInstancesPerNode, minWeightFractionPerNode, seed, stepSize, subsamplingRate, cacheNodeIds,
      checkpointInterval, featureSubsetStrategy, validationIndicatorCol, validationTol, 
      randomUnderSampling, percSampling)
    instr.logNumClasses(numClasses)

    val categoricalFeatures = MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val boostingStrategy = super.getOldBoostingStrategy(categoricalFeatures, OldAlgo.Classification)
    val (baseLearners, learnerWeights) = if (withValidation) {
      ImbalanceGradientBoostedTrees.runWithValidation(trainDataset, validationDataset, boostingStrategy,
        $(seed), $(featureSubsetStrategy), $(randomUnderSampling), $(percSampling), Some(instr))
    } else {
      ImbalanceGradientBoostedTrees.run(trainDataset, boostingStrategy, $(seed), $(featureSubsetStrategy),
        $(randomUnderSampling), $(percSampling), Some(instr))
    }
    baseLearners.foreach(copyValues(_))

    val numFeatures = baseLearners.head.numFeatures
    instr.logNumFeatures(numFeatures)

    new GBTClassificationModel(uid, baseLearners, learnerWeights, numFeatures)
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): ImbalanceGBTClassifier = defaultCopy(extra)
}

@Since("1.4.0")
object ImbalanceGBTClassifier extends DefaultParamsReadable[ImbalanceGBTClassifier] {

  /** Accessor for supported loss settings: logistic */
  @Since("1.4.0")
  final val supportedLossTypes: Array[String] = GBTClassifierParams.supportedLossTypes

  @Since("2.0.0")
  override def load(path: String): ImbalanceGBTClassifier = super.load(path)

}

private[ml] trait ImbalanceGBTClassifierParams extends PredictorParams {

  /**
   * Whether RUS or ROS is applied.
   * 
   * @group param
   */
  val randomUnderSampling: Param[Boolean] = new BooleanParam(this, "randomUnderSampling", 
    "Whether RUS is applied (true) or ROS (false).")

  setDefault(randomUnderSampling -> true)

  /** @group getParam */
  def getRandomUnderSampling: Boolean = $(randomUnderSampling)

  /**
   * Whether RUS or ROS is applied.
   * 
   * @group param
   */
  val percSampling: IntParam = new IntParam(this, "percSampling", 
    "Under/Oversampling percentage: 0 means balance the classes.",
    ParamValidators.gtEq(0))

  setDefault(percSampling -> 0)

  /** @group getParam */
  def getPercSampling: Int = $(percSampling)

}