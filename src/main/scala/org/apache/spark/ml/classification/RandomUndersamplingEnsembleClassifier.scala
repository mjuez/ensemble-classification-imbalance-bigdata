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
import org.apache.spark.ml.instance.RandomUndersamplingParams
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset
import scala.util.Random
import org.apache.spark.ml.instance.RandomUndersampling

@Since("2.4.5")
class RandomUndersamplingEnsembleClassifier @Since("2.4.5") (
    @Since("2.4.5") override val uid: String)
  extends AbstractImbalanceClassifier with RandomUndersamplingParams {

  /** @group setParam */
  def setPercUnder(value: Int): this.type = set(percUnder, value)

  @Since("2.4.5")
  def this() = this(Identifiable.randomUID("rusec"))

  override def copy(extra: ParamMap): RandomUndersamplingEnsembleClassifier = defaultCopy(extra)

  override def resample(dataset: Dataset[_], rand: Random): Dataset[_] = {
    val rus = new RandomUndersampling()
      .setPercUnder($(percUnder))
      .setSeed(rand.nextLong())

    rus.transform(dataset)
  }

}