/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

package ai.djl.timeseries.transform;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.TestRequirements;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.timefeature.TimeFeature;
import ai.djl.timeseries.transform.convert.Convert;
import ai.djl.timeseries.transform.feature.Feature;
import ai.djl.timeseries.transform.field.Field;
import ai.djl.timeseries.transform.split.Split;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

public class TransformTest {

    @BeforeClass
    public void setUp() {
        // TODO: Remove this once we support PyTorch support for timeseries extension
        TestRequirements.notArm();
    }

    @Test
    public void testVstackFeatures() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            TimeSeriesData input = new TimeSeriesData(10);
            input.setField(FieldName.FEAT_STATIC_REAL, manager.create(new Shape(1, 2, 3)));
            input.setField(FieldName.FEAT_STATIC_CAT, manager.create(new Shape(2, 2, 3)));
            FieldName[] inputFields = {FieldName.FEAT_STATIC_REAL, FieldName.FEAT_STATIC_CAT};
            Convert.vstackFeatures(FieldName.FEAT_TIME, inputFields, input);
            Assert.assertTrue(input.contains(FieldName.FEAT_TIME.name()));
            Assert.assertEquals(input.get(FieldName.FEAT_TIME).getShape(), new Shape(3, 2, 3));
        }
    }

    @Test
    public void testAddObservedValuesIndicator() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            TimeSeriesData input = new TimeSeriesData(10);
            NDArray target = manager.zeros(new Shape(2, 3));
            target.set(new NDIndex("0, 0"), Float.NaN);
            input.setField(FieldName.TARGET, target);

            Feature.addObservedValuesIndicator(
                    manager, FieldName.TARGET, FieldName.FEAT_TIME, input);
            Assert.assertTrue(input.contains(FieldName.FEAT_TIME.name()));
            Assert.assertEquals(
                    input.get(FieldName.TARGET).get("0, 0").toFloatArray()[0], 0.f, 1e-3f);
            Assert.assertEquals(
                    input.get(FieldName.FEAT_TIME).get("0, 0").toFloatArray()[0], 0.f, 1e-3f);
        }
    }

    @Test
    public void testAddTimeFeature() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            TimeSeriesData input = new TimeSeriesData(10);
            List<BiFunction<NDManager, List<LocalDateTime>, NDArray>> timeFeatures =
                    TimeFeature.timeFeaturesFromFreqStr("D");

            input.setStartTime(LocalDateTime.parse("2011-01-29T00:00"));
            input.setField(FieldName.TARGET, manager.ones(new Shape(32, 48)));
            Feature.addTimeFeature(
                    manager,
                    FieldName.START,
                    FieldName.TARGET,
                    FieldName.FEAT_AGE,
                    timeFeatures,
                    28,
                    "D",
                    input);

            NDArray array = input.get(FieldName.FEAT_AGE);
            Assert.assertEquals(array.getShape(), new Shape(3, 76));
            Assert.assertEquals(array.get("0, 0").toFloatArray()[0], 0.3333f, 1e-4f);
            Assert.assertEquals(array.get("1, 0").toFloatArray()[0], 0.4333f, 1e-4f);
            Assert.assertEquals(array.get("2, 0").toFloatArray()[0], -0.4233f, 1e-4f);
        }
    }

    @Test
    public void testAddAgeFeature() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            TimeSeriesData input = new TimeSeriesData(10);

            input.setField(FieldName.TARGET, manager.ones(new Shape(32, 48)));
            Feature.addAgeFeature(manager, FieldName.TARGET, FieldName.FEAT_AGE, 28, input);

            NDArray array = input.get(FieldName.FEAT_AGE);
            Assert.assertEquals(array.get("0, 0").toFloatArray()[0], 0.30103f, 1e-5f);
            Assert.assertEquals(array.getShape(), new Shape(1, 76));
        }
    }

    @Test
    public void testRemoveFields() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            TimeSeriesData input = new TimeSeriesData(10);

            NDArray array = manager.ones(new Shape(2, 3));
            input.setField(FieldName.TARGET, array);
            input.setField(FieldName.FEAT_AGE, array);
            input.setField(FieldName.FEAT_TIME, array);

            List<FieldName> removeFields = Arrays.asList(FieldName.FEAT_AGE, FieldName.FEAT_TIME);
            Field.removeFields(removeFields, input);

            Assert.assertTrue(input.contains(FieldName.TARGET.name()));
            Assert.assertFalse(input.contains(FieldName.FEAT_AGE.name()));
            Assert.assertFalse(input.contains(FieldName.FEAT_TIME.name()));
        }
    }

    @Test
    public void testSetFields() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            TimeSeriesData input = new TimeSeriesData(10);

            NDArray array = manager.ones(new Shape(2, 3));
            Field.setField(FieldName.TARGET, array, input);

            Assert.assertTrue(input.contains(FieldName.TARGET.name()));
            Assert.assertEquals(input.get(FieldName.TARGET), array);
        }
    }

    @Test
    public void testAsArray() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            TimeSeriesData input = new TimeSeriesData(10);

            NDArray array = manager.ones(new Shape(2, 3));
            input.setField(FieldName.FEAT_STATIC_CAT, array);

            Convert.asArray(FieldName.FEAT_STATIC_CAT, 2, input);
            Assert.assertTrue(input.contains(FieldName.FEAT_STATIC_CAT.name()));
            Assert.assertEquals(
                    input.get(FieldName.FEAT_STATIC_CAT).getDataType(), DataType.FLOAT32);

            Convert.asArray(FieldName.FEAT_STATIC_CAT, 2, DataType.INT32, input);
            Assert.assertTrue(input.contains(FieldName.FEAT_STATIC_CAT.name()));
            Assert.assertEquals(input.get(FieldName.FEAT_STATIC_CAT).getDataType(), DataType.INT32);
        }
    }

    @Test
    public void testSelectFields() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            TimeSeriesData input = new TimeSeriesData(10);

            NDArray array = manager.ones(new Shape(1));
            NDArray array1 = manager.ones(new Shape(2));
            NDArray array2 = manager.ones(new Shape(3));
            input.setField(FieldName.TARGET, array);
            input.setField(FieldName.FEAT_AGE, array1);
            input.setField(FieldName.FEAT_TIME, array2);

            String[] fields = {FieldName.FEAT_TIME.name(), FieldName.FEAT_AGE.name()};
            input = Field.selectField(fields, input);

            Assert.assertEquals(input.size(), 2);
            Assert.assertEquals(input.valueAt(0).getShape(), new Shape(3));
            Assert.assertEquals(input.valueAt(1).getShape(), new Shape(2));
        }
    }

    @Test
    public void testInstanceSplit() {
        try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            TimeSeriesData input = new TimeSeriesData(10);

            NDArray array = manager.ones(new Shape(32, 48));
            input.setField(FieldName.TARGET, array);
            input.setField(FieldName.FEAT_TIME, manager.ones(new Shape(32, 700)));
            input.setField(FieldName.OBSERVED_VALUES, manager.ones(new Shape(32, 400)));
            input.setStartTime(LocalDateTime.parse("2011-01-29T00:00"));

            Split.instanceSplit(
                    manager,
                    FieldName.TARGET,
                    FieldName.IS_PAD,
                    FieldName.START,
                    FieldName.FORECAST_START,
                    PredictionSplitSampler.newTestSplitSampler(),
                    30,
                    28,
                    new FieldName[] {FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES},
                    0f,
                    input);

            Assert.assertTrue(input.contains("PAST_" + FieldName.FEAT_TIME));
            Assert.assertTrue(input.contains("FUTURE_" + FieldName.FEAT_TIME));
            Assert.assertTrue(input.contains("PAST_" + FieldName.OBSERVED_VALUES));
            Assert.assertTrue(input.contains("FUTURE_" + FieldName.OBSERVED_VALUES));
            Assert.assertTrue(input.contains("PAST_" + FieldName.TARGET));
            Assert.assertTrue(input.contains("FUTURE_" + FieldName.TARGET));

            Assert.assertEquals(
                    input.get("PAST_" + FieldName.FEAT_TIME).getShape(), new Shape(30, 32));
            Assert.assertEquals(
                    input.get("FUTURE_" + FieldName.FEAT_TIME).getShape(), new Shape(28, 32));
            Assert.assertEquals(
                    input.get("PAST_" + FieldName.OBSERVED_VALUES).getShape(), new Shape(30, 32));
            Assert.assertEquals(
                    input.get("FUTURE_" + FieldName.OBSERVED_VALUES).getShape(), new Shape(28, 32));
            Assert.assertEquals(
                    input.get("PAST_" + FieldName.TARGET).getShape(), new Shape(30, 32));
            Assert.assertEquals(
                    input.get("FUTURE_" + FieldName.TARGET).getShape(), new Shape(0, 32));
            Assert.assertEquals(
                    input.getForecastStartTime(), LocalDateTime.parse("2011-03-18T00:00"));
        }
    }
}
