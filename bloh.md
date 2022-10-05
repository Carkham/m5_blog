### DJL timeseries package applied on M5forecasting dataset

![img](https://cdn-images-1.medium.com/max/1200/1*vfVcF-ZaC_WbTt_LWlXH0w.png)

The timeseries package contains components and tools for building time series models and inferring on pretrained models using DJL.

### M5 Forecasting

#### Introduction

M5 Forecasting competition is one of the [Makridakis competitions](https://en.wikipedia.org/wiki/Makridakis_Competitions) which aims to provide a better understanding and advancement of forecasting methodology by comparing the performance of different methods in achieving accurate predictions, estimating the levels of uncertainty, avoiding costly mistakes, and applying optimal forecasting practices.

#### Goal

We need to **forecast future sales** at Walmart based on hierarchical sales in the states of California, Texas, and Wisconsin. We are tasked to estimate the accuracy of our predictions with the metric: **Weighted Root Mean Squared Scaled Error** (RMSSE), which is derived from the [Mean Absolute Scaled Error (MASE)](https://en.wikipedia.org/wiki/Mean_absolute_scaled_error) that was designed to be scale invariant and symmetric.

#### Data

We will make predictions on **42,840 hierarchical time-series data**. “Hierarchical” here means that we can aggregate the data from different perspectives, including item level, department level, product category level, and state level. Also for each item, we can access information about its price, promotions, and holidays. As well as sales data from Jan 2011 all the way to June 2016.

**Notes**: Since **most of the time series data contain zero values**, we **aggregate sales data every seven days** to train and predict at a coarser granularity, considering the performance of the model. **After downloading the dataset from M5 Forecasting competition, you can use our** [**script**](https://gist.github.com/Carkham/a5162c9298bc51fec648a458a3437008) **to get coarse-grained data. This script will create “weekly_xxx.csv” files representing weekly data in the dataset directory you specify.**

#### DeepAR model

DeepAR forecasting algorithm is a supervised learning algorithm for forecasting scalar (one-dimensional) time series using recurrent neural networks (RNN).

Unlike traditional time series forecasting models, DeepAR estimates the future probability distribution of time series based on the past. In retail businesses, probabilistic demand forecasting is critical to delivering the right inventory at the right time and in the right place.

Therefore, we choose the sales data set in the real scene as an example to describe how to use the timeseries package for forecasting

#### Metrics

We use the following metrics to evaluate the performance of the DeepAR model in the M5 Forecasting competition.

```
> [INFO ] - metric: Coverage[0.99]:    0.92
> [INFO ] - metric: Coverage[0.67]:    0.51
> [INFO ] - metric: abs_target_sum:    1224665.00
> [INFO ] - metric: abs_target_mean:   10.04
> [INFO ] - metric: NRMSE: 0.84
> [INFO ] - metric: RMSE:  8.40
> [INFO ] - metric: RMSSE: 1.00
> [INFO ] - metric: abs_error: 14.47
> [INFO ] - metric: QuantileLoss[0.67]:    18.23
> [INFO ] - metric: QuantileLoss[0.99]:    103.07
> [INFO ] - metric: QuantileLoss[0.50]:    9.49
> [INFO ] - metric: QuantileLoss[0.95]:    66.69
> [INFO ] - metric: Coverage[0.95]:    0.87
> [INFO ] - metric: Coverage[0.50]:    0.33
> [INFO ] - metric: MSE:   70.64
```

As you can see, our pretrained model has some effect on the data prediction of item value. And some metrics can basically meet expectations. For example, **RMSSE**, which is a measure of the relative error between the predicted value and the actual value. 1.00 means that the model can reflect the changes of the time series data to a certain extent.

### Run the M5 Forecasting inference example

#### Setup

To get started with DJL time series package, add the following code snippet defining the necessary dependencies to your `build.gradle` file.

```
plugins {
  id 'java'
}
repositories {                           
  mavenCentral()
}
dependencies {
  implementation "org.apache.logging.log4j:log4j-slf4j-impl:2.17.1"
  implementation platform("ai.djl:bom:0.16.0")
  implementation "ai.djl:api"
  implementation "ai.djl.timeseries"
  runtimeOnly "ai.djl.mxnet:mxnet-engine"
  runtimeOnly "ai.djl.mxnet:mxnet-model-zoo"
}
```

#### Define your dataset

In order to realize the preprocessing of time series data, we define the `TimeSeriesData` as the input of the Translator, which is used to store the feature fields and perform corresponding transformations.

So for your own dataset, you need to customize the way you get the data and put it into `TimeSeriesData` as the input to the translator.

For M5 dataset we have:

```
class M5Dataset implements Iterable<NDList>, Iterator<NDList> {
      
    // coarse-grained data
    private static String fileName = "weekly_sales_train_evaluation.csv";

    private NDManager manager;
    private List<Feature> target;
    private List<CSVRecord> csvRecords;
    private long size;
    private long current;

    M5Dataset(Builder builder) {
        manager = builder.manager;
        target = builder.target;
        prepare(builder);
        size = csvRecords.size();
    }

    /** Load data into CSVRecords */
    private void prepare(Builder builder) throws IOException {
        // csvUrl represents the url path to your local data
        URL csvUrl = builder.root.resolve(fileName).toUri().toURL();
        Reader reader = new InputStreamReader(
            new BufferedInputStream(csvUrl.openStream()), StandardCharsets.UTF_8);
        CSVParser csvParser = new CSVParser(reader, builder.csvFormat);
        csvRecords = csvParser.getRecords();
    }

    @Override
    public boolean hasNext() {
        return current < size;
    }

    @Override
    public NDList next() {
        NDList data = getRowFeatures(manager, current, target);
        current++;
        return data;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Get string data of selected cell from index row in CSV file and create NDArray to save  */
    private NDList getRowFeatures(NDManager manager, long index, List<Feature> selected) {
        DynamicBuffer bb = new DynamicBuffer();
        // Feature is the djl classs used to convert strings into their corresponding data
        for (Feature feature : selected) {
            String name = feature.getName();
            String value = getCell(index, name);
            feature.getFeaturizer().featurize(bb, value);
        }
        FloatBuffer buf = bb.getBuffer();
        return new NDList(manager.create(buf, new Shape(bb.getLength())));
    }

    private String getCell(long rowIndex, String featureName) {
        CSVRecord record = csvRecords.get(Math.toIntExact(rowIndex));
        return record.get(featureName);
    }

    @Override
    public Iterator<NDList> iterator() {
        return this;
    }

    public static final class Builder {

        NDManager manager;
        List<Feature> target;
        CSVFormat csvFormat;
        Path root;

        Builder(Path root, NDManager manager) {
            this.root = root;
            this.manager = manager;
            csvFormat =
                CSVFormat.DEFAULT
                .builder()
                .setHeader()
                .setSkipHeaderRecord(true)
                .setIgnoreHeaderCase(true)
                .setTrim(true)
                .build();
            target = new ArrayList<>();
            for (int i = 1; i <= 277; i++) {
                target.add(new Feature("w_" + i, true));
            }
        }

        public M5Dataset build() {
            return new M5Dataset(this);
        }
    }
}
```

#### Prepare dataset

#### Set your own dataset path.

```
Path m5ForecastFile = Paths.get("/YOUR PATH/m5-forecasting-accuracy");
NDManager manager = NDManager.newBaseManager();
M5Dataset dataset = M5Dataset.builder().setManager(manager).setRoot(m5ForecastFile).build();
```

#### Config your translator

> *The inference workflow is combined with input preprocessing, model forward, and output post-processing. DJL encapsulates input and output processing into the translator and uses Predictor to do the model forward.*

`DeepARTranslator` provides support for data preprocessing and postprocessing for probabilistic prediction models. Referring to GluonTS, our translator can perform corresponding preprocessing on `TimeseriesData` containing data according to different parameters to obtain the input of the network model. And post-processing the output of the network to get the prediction result.

For DeepAR models, you must set the following arguments.

```
Logger logger = LoggerFactory.getLogger(TimeSeriesDemo.class);
String freq = "W";
int predictionLength = 4;
LocalDateTime startTime = LocalDateTime.parse("2011-01-29T00:00");

Map<String, Object> arguments = new ConcurrentHashMap<>();

arguments.put("prediction_length", predictionLength);
arguments.put("freq", freq); // The predicted frequency contains units and values

// Parameters from DeepAR in GluonTS
arguments.put("use_" + FieldName.FEAT_DYNAMIC_REAL.name().toLowerCase(), false); 
arguments.put("use_" + FieldName.FEAT_STATIC_CAT.name().toLowerCase(), false);
arguments.put("use_" + FieldName.FEAT_STATIC_REAL.name().toLowerCase(), false);
```

For any other GluonTS model, you can quickly develop your own translator using the classes in `transform` modules (etc. `TransformerTranslator`).

#### Load your own model from the local file system

At this step, you need to construct the `Criteria` API, which is used as search criteria to look for a ZooModel. In this application, you can customize your local pretrained model path (local directory or an archive file containing .`params` and `symbol.json`.) with .`optModelPath()`. The following code snippet loads the model with the file path: `/YOUR PATH/deepar.zip` .

```
DeepARTranslator translator = DeepARTranslator.builder(arguments).build();
Criteria<TimeSeriesData, Forecast> criteria =
        Criteria.builder()
                .setTypes(TimeSeriesData.class, Forecast.class)
                .optModelPath(Paths.get("/YOUR PATH/deepar.zip"}))
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();
```

#### Inference

Now, you are ready to used the model bundled with the translator created above to run inference.

Since we need to generate features based on dates and make predictions with reference to the context, for each `TimeSeriesData` you must set the values of its `**StartTime**` and `**TARGET**` fields.

```
try (ZooModel<TimeSeriesData, Forecast> model = criteria.loadModel();
             Predictor<TimeSeriesData, Forecast> predictor = model.newPredictor()) {
    data = dataset.next();
    NDArray array = data.singletonOrThrow();
    TimeSeriesData input = new TimeSeriesData(10);
    input.setStartTime(startTime); // start time of prediction
    input.setField(FieldName.TARGET, array); // target value through whole context length
    Forecast forecast = predictor.predict(input);
    saveResult(forecast); // save result and plot it with python.
    }
}
```

#### Results

The `Forecast` are objects that contain all the sample paths in the form of `NDArray` with dimension `(numSamples, predictionLength)`, the start date of the forecast. You can access all these information by simply invoking the corresponding function.

You can summarize the sample paths by computing, including the mean and quantile, for each step in the prediction window.

```
logger.info("Mean of the prediction windows:\n" + forecast.mean().toDebugString());
logger.info("0.5-quantile(Median) of the prediction windows:\n" + forecast.quantile("0.5").toDebugString());
> [INFO ] - Mean of the prediction windows:
> ND: (4) cpu() float32
> [5.97, 6.1 , 5.9 , 6.11]
>
> [INFO ] - 0.5-quantile(Median) of the prediction windows:
> ND: (4) cpu() float32
> [6., 5., 5., 6.]
```

We visualize the forecast result with mean, prediction intervals, etc.

![img](https://cdn-images-1.medium.com/max/1200/1*hiLSY5NrkTZ_Sf6pexMgyA.jpeg)

#### Metrics

Here we compute aggregate performance metrics in the source code

Click [here](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/DeepARTimeSeries.java) to see the **full source code**.

# Train your first time series model

In this section we will take you through the creation and training of our time series model - DeepAR

## Define your probability distribution

In timeseries package, we do not just predict the target value, but the target probability distribution.
The probability distribution provides the benefit that it can reflect the possibility of the target value in different intervals, which is of greater significance for real production scenarios. Therefore, before you train any of your timing models, you need to define a probability `DistributionOutput` for it as the predicted output.

```java
DistributionOutput distributionOutput = new NegativeBinomialOutput();
```

Here we consider that sales are more in line with the negtive binomial distribution.

## Construct your model

As with Translator you need to set some hyperparameters, including the frequency, length of the prediction even number of layers for a neural network, etc.

```java
String freq = "W";
int predictionLength = 4;
DeepARNetwork getDeepARModel(DistributionOutput distributionOutput, boolean training) {
    DeepARNetwork.Builder builder = DeepARNetwork.builder()
        .setCardinality(cardinality)
        .setFreq(freq)
        .setPredictionLength(predictionLength)
        .optDistrOutput(distributionOutput)
        .optUseFeatStaticCat(true);
    // This is because the network will output different content during training and inference
    return training ? builder.buildTrainingNetwork() : builder.buildPredictionNetwork();
```

After we set up the configuration, we can build a DeepAR Network Block and set it back to model using setBlock

```java
Model model = Model.newInstance("deepar");
DeepARNetwork trainingNetwork = getDeepARModel(distributionOutput, true);
model.setBlock(trainingNetwork);
```

## prepare dataset

For `TimeseriesDataset`, you need to specify the pre-processing transformations, which includes steps such as time-series feature generation and transformation that need to be applied to `TimeSeriesData` before feeding the input to the model.

Different models have different pre-processing requirements, so we can derive the corresponding transformations based on the previously obtained network models and obtain some necessary parameters.

```java
List<TimeSeriesTransform> trainingTransformation = trainingNetwork.createTrainingTransformation(manager);
int contextLength = trainingNetwork.getContextLength();
```

You can construct a M5Forecast builder with your own specifications.

```java
M5Forecast getDataset(
    List<TimeSeriesTransform> transformation,
    Repository repository,
    int contextLength,
    Dataset.Usage usage) {
    // In order to create a TimeSeriesDataset, you must specify the transformation of the data
    // preprocessing
    M5Forecast.Builder builder =
        M5Forecast.builder()
        .optUsage(usage)
        .setRepository(repository)
        .setTransformation(transformation)
        .setContextLength(contextLength)
        .setSampling(32, usage == Dataset.Usage.TRAIN);

    int maxWeek = usage == Dataset.Usage.TRAIN ? 273 : 277;
    for (int i = 1; i <= maxWeek; i++) {
        builder.addFeature("w_" + i, FieldName.TARGET);
    }

    // this is the static category feature
    M5Forecast m5Forecast =
        builder.addFeature("state_id", FieldName.FEAT_STATIC_CAT)
        .addFeature("store_id", FieldName.FEAT_STATIC_CAT)
        .addFeature("cat_id", FieldName.FEAT_STATIC_CAT)
        .addFeature("dept_id", FieldName.FEAT_STATIC_CAT)
        .addFeature("item_id", FieldName.FEAT_STATIC_CAT)
        .addFieldFeature(
        FieldName.START,
        // A Featurizer convert String to LocalDateTime
        new Feature("date", TimeFeaturizers.getConstantTimeFeaturizer(startTime)))
        .build();
    m5Forecast.prepare(new ProgressBar());
    return m5Forecast;
}
```

## Set up training configuration

We need to set the loss function of the model in this section, and for any probability distribution, the corresponding loss needs to be calculated by `DistributionLoss` wrapping.

```java
DefaultTrainingConfig setupTrainingConfig(
            Arguments arguments, DistributionOutput distributionOutput) {
    return new DefaultTrainingConfig(new DistributionLoss("Loss", distributionOutput))
        .addEvaluator(new RMSSE(distributionOutput)) // Use RMSSE so we can understand the performance of the model
        .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
        .optInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
}
```

## Train your model

Now you can start training.

```java
Repository repository = Repository.newInstance("test",Paths.get("YOUR PATH"));
M5Forecast trainSet = getDataset(trainingTransformation, repository, contextLength, Dataset.Usage.TRAIN);

Trainer trainer = model.newTrainer(config)
trainer.setMetrics(new Metrics());# DJL timeseries package applied on M5forecasting dataset

![img](https://cdn-images-1.medium.com/max/1200/1*vfVcF-ZaC_WbTt_LWlXH0w.png)

The timeseries package contains components and tools for building time series models and inferring on pretrained models using DJL.

## M5 Forecasting

### Introduction

M5 Forecasting competition is one of the [Makridakis competitions](https://en.wikipedia.org/wiki/Makridakis_Competitions) which aims to provide a better understanding and advancement of forecasting methodology by comparing the performance of different methods in achieving accurate predictions, estimating the levels of uncertainty, avoiding costly mistakes, and applying optimal forecasting practices.

### Goal

We need to **forecast future sales** at Walmart based on hierarchical sales in the states of California, Texas, and Wisconsin. We are tasked to estimate the accuracy of our predictions with the metric: **Weighted Root Mean Squared Scaled Error** (RMSSE), which is derived from the [Mean Absolute Scaled Error (MASE)](https://en.wikipedia.org/wiki/Mean_absolute_scaled_error) that was designed to be scale invariant and symmetric.

### Data

We will make predictions on **42,840 hierarchical time-series data**. “Hierarchical” here means that we can aggregate the data from different perspectives, including item level, department level, product category level, and state level. Also for each item, we can access information about its price, promotions, and holidays. As well as sales data from Jan 2011 all the way to June 2016.

**Notes**: Since **most of the time series data contain zero values**, we **aggregate sales data every seven days** to train and predict at a coarser granularity, considering the performance of the model. **After downloading the dataset from M5 Forecasting competition, you can use our** [**script**](https://gist.github.com/Carkham/a5162c9298bc51fec648a458a3437008) **to get coarse-grained data. This script will create “weekly_xxx.csv” files representing weekly data in the dataset directory you specify.**

## DeepAR model

DeepAR forecasting algorithm is a supervised learning algorithm for forecasting scalar (one-dimensional) time series using recurrent neural networks (RNN).

Unlike traditional time series forecasting models, DeepAR estimates the future probability distribution of time series based on the past. In retail businesses, probabilistic demand forecasting is critical to delivering the right inventory at the right time and in the right place.

Therefore, we choose the sales data set in the real scene as an example to describe how to use the timeseries package for forecasting

## Metrics

We use the following metrics to evaluate the performance of the DeepAR model in the M5 Forecasting competition.

```
> [INFO ] - metric: Coverage[0.99]:    0.92
> [INFO ] - metric: Coverage[0.67]:    0.51
> [INFO ] - metric: abs_target_sum:    1224665.00
> [INFO ] - metric: abs_target_mean:   10.04
> [INFO ] - metric: NRMSE: 0.84
> [INFO ] - metric: RMSE:  8.40
> [INFO ] - metric: RMSSE: 1.00
> [INFO ] - metric: abs_error: 14.47
> [INFO ] - metric: QuantileLoss[0.67]:    18.23
> [INFO ] - metric: QuantileLoss[0.99]:    103.07
> [INFO ] - metric: QuantileLoss[0.50]:    9.49
> [INFO ] - metric: QuantileLoss[0.95]:    66.69
> [INFO ] - metric: Coverage[0.95]:    0.87
> [INFO ] - metric: Coverage[0.50]:    0.33
> [INFO ] - metric: MSE:   70.64
```

As you can see, our pretrained model has some effect on the data prediction of item value. And some metrics can basically meet expectations. For example, **RMSSE**, which is a measure of the relative error between the predicted value and the actual value. 1.00 means that the model can reflect the changes of the time series data to a certain extent.

## Run the M5 Forecasting inference example

### Setup

To get started with DJL time series package, add the following code snippet defining the necessary dependencies to your `build.gradle` file.

```
plugins {
  id 'java'
}
repositories {                           
  mavenCentral()
}
dependencies {
  implementation "org.apache.logging.log4j:log4j-slf4j-impl:2.17.1"
  implementation platform("ai.djl:bom:0.16.0")
  implementation "ai.djl:api"
  implementation "ai.djl.timeseries"
  runtimeOnly "ai.djl.mxnet:mxnet-engine"
  runtimeOnly "ai.djl.mxnet:mxnet-model-zoo"
}
```

### Define your dataset

In order to realize the preprocessing of time series data, we define the `TimeSeriesData` as the input of the Translator, which is used to store the feature fields and perform corresponding transformations.

So for your own dataset, you need to customize the way you get the data and put it into `TimeSeriesData` as the input to the translator.

For M5 dataset we have:

```
class M5Dataset implements Iterable<NDList>, Iterator<NDList> {
      
    // coarse-grained data
    private static String fileName = "weekly_sales_train_evaluation.csv";

    private NDManager manager;
    private List<Feature> target;
    private List<CSVRecord> csvRecords;
    private long size;
    private long current;

    M5Dataset(Builder builder) {
        manager = builder.manager;
        target = builder.target;
        prepare(builder);
        size = csvRecords.size();
    }

    /** Load data into CSVRecords */
    private void prepare(Builder builder) throws IOException {
        // csvUrl represents the url path to your local data
        URL csvUrl = builder.root.resolve(fileName).toUri().toURL();
        Reader reader = new InputStreamReader(
            new BufferedInputStream(csvUrl.openStream()), StandardCharsets.UTF_8);
        CSVParser csvParser = new CSVParser(reader, builder.csvFormat);
        csvRecords = csvParser.getRecords();
    }

    @Override
    public boolean hasNext() {
        return current < size;
    }

    @Override
    public NDList next() {
        NDList data = getRowFeatures(manager, current, target);
        current++;
        return data;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Get string data of selected cell from index row in CSV file and create NDArray to save  */
    private NDList getRowFeatures(NDManager manager, long index, List<Feature> selected) {
        DynamicBuffer bb = new DynamicBuffer();
        // Feature is the djl classs used to convert strings into their corresponding data
        for (Feature feature : selected) {
            String name = feature.getName();
            String value = getCell(index, name);
            feature.getFeaturizer().featurize(bb, value);
        }
        FloatBuffer buf = bb.getBuffer();
        return new NDList(manager.create(buf, new Shape(bb.getLength())));
    }

    private String getCell(long rowIndex, String featureName) {
        CSVRecord record = csvRecords.get(Math.toIntExact(rowIndex));
        return record.get(featureName);
    }

    @Override
    public Iterator<NDList> iterator() {
        return this;
    }

    public static final class Builder {

        NDManager manager;
        List<Feature> target;
        CSVFormat csvFormat;
        Path root;

        Builder(Path root, NDManager manager) {
            this.root = root;
            this.manager = manager;
            csvFormat =
                CSVFormat.DEFAULT
                .builder()
                .setHeader()
                .setSkipHeaderRecord(true)
                .setIgnoreHeaderCase(true)
                .setTrim(true)
                .build();
            target = new ArrayList<>();
            for (int i = 1; i <= 277; i++) {
                target.add(new Feature("w_" + i, true));
            }
        }

        public M5Dataset build() {
            return new M5Dataset(this);
        }
    }
}
```

### Set your own dataset path.

```
Path m5ForecastFile = Paths.get("/YOUR PATH/m5-forecasting-accuracy");
NDManager manager = NDManager.newBaseManager();
M5Dataset dataset = M5Dataset.builder().setManager(manager).setRoot(m5ForecastFile).build();
```

### Config your translator

> *The inference workflow is combined with input preprocessing, model forward, and output post-processing. DJL encapsulates input and output processing into the translator and uses Predictor to do the model forward.*

`DeepARTranslator` provides support for data preprocessing and postprocessing for probabilistic prediction models. Referring to GluonTS, our translator can perform corresponding preprocessing on `TimeseriesData` containing data according to different parameters to obtain the input of the network model. And post-processing the output of the network to get the prediction result.

For DeepAR models, you must set the following arguments.

```
Logger logger = LoggerFactory.getLogger(TimeSeriesDemo.class);
String freq = "W";
int predictionLength = 4;
LocalDateTime startTime = LocalDateTime.parse("2011-01-29T00:00");

Map<String, Object> arguments = new ConcurrentHashMap<>();

arguments.put("prediction_length", predictionLength);
arguments.put("freq", freq); // The predicted frequency contains units and values

// Parameters from DeepAR in GluonTS
arguments.put("use_" + FieldName.FEAT_DYNAMIC_REAL.name().toLowerCase(), false); 
arguments.put("use_" + FieldName.FEAT_STATIC_CAT.name().toLowerCase(), false);
arguments.put("use_" + FieldName.FEAT_STATIC_REAL.name().toLowerCase(), false);
```

For any other GluonTS model, you can quickly develop your own translator using the classes in `transform` modules (etc. `TransformerTranslator`).

### Load your own model from the local file system

At this step, you need to construct the `Criteria` API, which is used as search criteria to look for a ZooModel. In this application, you can customize your local pretrained model path (local directory or an archive file containing .`params` and `symbol.json`.) with .`optModelPath()`. The following code snippet loads the model with the file path: `/YOUR PATH/deepar.zip` .

```
DeepARTranslator translator = DeepARTranslator.builder(arguments).build();
Criteria<TimeSeriesData, Forecast> criteria =
        Criteria.builder()
                .setTypes(TimeSeriesData.class, Forecast.class)
                .optModelPath(Paths.get("/YOUR PATH/deepar.zip"}))
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();
```

### Inference

Now, you are ready to used the model bundled with the translator created above to run inference.

Since we need to generate features based on dates and make predictions with reference to the context, for each `TimeSeriesData` you must set the values of its `**StartTime**` and `**TARGET**` fields.

```
try (ZooModel<TimeSeriesData, Forecast> model = criteria.loadModel();
             Predictor<TimeSeriesData, Forecast> predictor = model.newPredictor()) {
    data = dataset.next();
    NDArray array = data.singletonOrThrow();
    TimeSeriesData input = new TimeSeriesData(10);
    input.setStartTime(startTime); // start time of prediction
    input.setField(FieldName.TARGET, array); // target value through whole context length
    Forecast forecast = predictor.predict(input);
    saveResult(forecast); // save result and plot it with python.
    }
}
```

### Results

The `Forecast` are objects that contain all the sample paths in the form of `NDArray` with dimension `(numSamples, predictionLength)`, the start date of the forecast. You can access all these information by simply invoking the corresponding function.

You can summarize the sample paths by computing, including the mean and quantile, for each step in the prediction window.

```
logger.info("Mean of the prediction windows:\n" + forecast.mean().toDebugString());
logger.info("0.5-quantile(Median) of the prediction windows:\n" + forecast.quantile("0.5").toDebugString());
> [INFO ] - Mean of the prediction windows:
> ND: (4) cpu() float32
> [5.97, 6.1 , 5.9 , 6.11]
>
> [INFO ] - 0.5-quantile(Median) of the prediction windows:
> ND: (4) cpu() float32
> [6., 5., 5., 6.]
```

We visualize the forecast result with mean, prediction intervals, etc.

![img](https://cdn-images-1.medium.com/max/1200/1*hiLSY5NrkTZ_Sf6pexMgyA.jpeg)

### Metrics

Here we compute aggregate performance metrics in the source code

Click [here](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/DeepARTimeSeries.java) to see the **full source code**.

## Train your first time series model

In this section we will take you through the creation and training of our time series model - DeepAR

### Define your probability distribution

In timeseries package, we do not just predict the target value, but the target probability distribution.
The probability distribution provides the benefit that it can reflect the possibility of the target value in different intervals, which is of greater significance for real production scenarios. Therefore, before you train any of your timing models, you need to define a probability `DistributionOutput` for it as the predicted output.

```java
DistributionOutput distributionOutput = new NegativeBinomialOutput();
```

Here we consider that sales are more in line with the negtive binomial distribution.

### Construct your model

As with Translator you need to set some hyperparameters, including the frequency, length of the prediction even number of layers for a neural network, etc.

```java
String freq = "W";
int predictionLength = 4;
DeepARNetwork getDeepARModel(DistributionOutput distributionOutput, boolean training) {
    DeepARNetwork.Builder builder = DeepARNetwork.builder()
        .setCardinality(cardinality)
        .setFreq(freq)
        .setPredictionLength(predictionLength)
        .optDistrOutput(distributionOutput)
        .optUseFeatStaticCat(true);
    // This is because the network will output different content during training and inference
    return training ? builder.buildTrainingNetwork() : builder.buildPredictionNetwork();
```

After we set up the configuration, we can build a DeepAR Network Block and set it back to model using setBlock

```java
Model model = Model.newInstance("deepar");
DeepARNetwork trainingNetwork = getDeepARModel(distributionOutput, true);
model.setBlock(trainingNetwork);
```

### prepare dataset

For `TimeseriesDataset`, you need to specify the pre-processing transformations, which includes steps such as time-series feature generation and transformation that need to be applied to `TimeSeriesData` before feeding the input to the model.

Different models have different pre-processing requirements, so we can derive the corresponding transformations based on the previously obtained network models and obtain some necessary parameters.

```java
List<TimeSeriesTransform> trainingTransformation = trainingNetwork.createTrainingTransformation(manager);
int contextLength = trainingNetwork.getContextLength();
```

You can construct a M5Forecast builder with your own specifications.

```java
M5Forecast getDataset(
    List<TimeSeriesTransform> transformation,
    Repository repository,
    int contextLength,
    Dataset.Usage usage) {
    // In order to create a TimeSeriesDataset, you must specify the transformation of the data
    // preprocessing
    M5Forecast.Builder builder =
        M5Forecast.builder()
        .optUsage(usage)
        .setRepository(repository)
        .setTransformation(transformation)
        .setContextLength(contextLength)
        .setSampling(32, usage == Dataset.Usage.TRAIN);

    int maxWeek = usage == Dataset.Usage.TRAIN ? 273 : 277;
    for (int i = 1; i <= maxWeek; i++) {
        builder.addFeature("w_" + i, FieldName.TARGET);
    }

    // this is the static category feature
    M5Forecast m5Forecast =
        builder.addFeature("state_id", FieldName.FEAT_STATIC_CAT)
        .addFeature("store_id", FieldName.FEAT_STATIC_CAT)
        .addFeature("cat_id", FieldName.FEAT_STATIC_CAT)
        .addFeature("dept_id", FieldName.FEAT_STATIC_CAT)
        .addFeature("item_id", FieldName.FEAT_STATIC_CAT)
        .addFieldFeature(
        FieldName.START,
        // A Featurizer convert String to LocalDateTime
        new Feature("date", TimeFeaturizers.getConstantTimeFeaturizer(startTime)))
        .build();
    m5Forecast.prepare(new ProgressBar());
    return m5Forecast;
}
```

### Set up training configuration

We need to set the loss function of the model in this section, and for any probability distribution, the corresponding loss needs to be calculated by `DistributionLoss` wrapping.

```java
DefaultTrainingConfig setupTrainingConfig(
            Arguments arguments, DistributionOutput distributionOutput) {
    return new DefaultTrainingConfig(new DistributionLoss("Loss", distributionOutput))
        .addEvaluator(new RMSSE(distributionOutput)) // Use RMSSE so we can understand the performance of the model
        .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
        .optInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
}
```

### Train your model

Now you can start training.

```java
Repository repository = Repository.newInstance("test",Paths.get("YOUR PATH"));
M5Forecast trainSet = getDataset(trainingTransformation, repository, contextLength, Dataset.Usage.TRAIN);

Trainer trainer = model.newTrainer(config)
trainer.setMetrics(new Metrics());

int historyLength = trainingNetwork.getHistoryLength();
Shape[] inputShapes = new Shape[9];
// (N, num_cardinality)
inputShapes[0] = new Shape(1, 5);
// (N, num_real) if use_feat_stat_real else (N, 1)
inputShapes[1] = new Shape(1, 1);
// (N, history_length, num_time_feat + num_age_feat)
inputShapes[2] =new Shape(1, historyLength, TimeFeature.timeFeaturesFromFreqStr(freq).size() + 1);
inputShapes[3] = new Shape(1, historyLength);
inputShapes[4] = new Shape(1, historyLength);
inputShapes[5] = new Shape(1, historyLength);
inputShapes[6] = new Shape(1,predictionLength, TimeFeature.timeFeaturesFromFreqStr(freq).size() + 1);
inputShapes[7] = new Shape(1, predictionLength);
inputShapes[8] = new Shape(1, predictionLength);
trainer.initialize(inputShapes);

int epoch = 10;
EasyTrain.fit(trainer, epoch, trainSet, null);
```

```java
[INFO ] - Training on: cpu().
[INFO ] - Load MXNet Engine Version 1.9.0 in 0.086 ms.
Training:     30% |=============                           | RMSSE: 1.43, Loss: 2.42, speed: 443.11 items/sec
```

You can get the full source code [here](https://github.com/deepjavalibrary/djl/pull/2031/files#diff-77a09bb4a44d897cefd0123b3a1727eb5923e07d4efa56e85bf1ce0474f0aed3).

int historyLength = trainingNetwork.getHistoryLength();
Shape[] inputShapes = new Shape[9];
// (N, num_cardinality)
inputShapes[0] = new Shape(1, 5);
// (N, num_real) if use_feat_stat_real else (N, 1)
inputShapes[1] = new Shape(1, 1);
// (N, history_length, num_time_feat + num_age_feat)
inputShapes[2] =new Shape(1, historyLength, TimeFeature.timeFeaturesFromFreqStr(freq).size() + 1);
inputShapes[3] = new Shape(1, historyLength);
inputShapes[4] = new Shape(1, historyLength);
inputShapes[5] = new Shape(1, historyLength);
inputShapes[6] = new Shape(1,predictionLength, TimeFeature.timeFeaturesFromFreqStr(freq).size() + 1);
inputShapes[7] = new Shape(1, predictionLength);
inputShapes[8] = new Shape(1, predictionLength);
trainer.initialize(inputShapes);

int epoch = 10;
EasyTrain.fit(trainer, epoch, trainSet, null);
```

```java
[INFO ] - Training on: cpu().
[INFO ] - Load MXNet Engine Version 1.9.0 in 0.086 ms.
Training:     30% |=============                           | RMSSE: 1.43, Loss: 2.42, speed: 443.11 items/sec
```

You can get the full source code [here](https://github.com/deepjavalibrary/djl/pull/2031/files#diff-77a09bb4a44d897cefd0123b3a1727eb5923e07d4efa56e85bf1ce0474f0aed3).
