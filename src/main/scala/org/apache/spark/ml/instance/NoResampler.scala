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
import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.param.ParamMap

class NoResampler(override val uid: String) extends Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("nor"))

  override def transform(inputDS: Dataset[_]): DataFrame = {
    return inputDS.toDF
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