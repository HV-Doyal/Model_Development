using Microsoft.ML;  // Importing the ML.NET library for machine learning
using Microsoft.ML.Trainers;  // Importing the trainers available in ML.NET
using Microsoft.ML.Trainers.LightGbm;  // Importing the LightGBM trainer for multiclass classification
using System.Globalization;  // Importing the namespace for handling culture and date formatting

namespace Model_Development  // Declaring a namespace to organize the code
{
    public static class PredictBestCategory  // Defining a static class for the prediction logic
    {
        // File paths for the training and testing data
        static string trainFilePath = "C:\\Users\\harsh\\OneDrive\\Desktop\\Model_Development\\Dataset\\train.csv";
        static string testFilePath = "C:\\Users\\harsh\\OneDrive\\Desktop\\Model_Development\\Dataset\\test.csv";

        public static void predict()  // Defining the prediction function
        {
            var mlContext = new MLContext();  // Creating an MLContext object which is the starting point for ML.NET

            Console.WriteLine("🚀 Starting ML.NET Training Pipeline...\n");  // Output to indicate the start of the pipeline

            // STEP 1: Load and prepare training data
            Console.WriteLine("📂 Loading and preparing training data...");  // Indicating that the training data is being loaded

            var allLines = File.ReadAllLines(trainFilePath).Skip(1).ToList();  // Read all lines from the training file and skip the first line (header)

            // Process each line of data into an object format (MonthlySalesSummary)
            var allData = allLines.Select(line =>
            {
                var parts = line.Split(',');  // Splitting the line by commas to separate the fields
                var date = DateTime.Parse(parts[0]);  // Parsing the date field
                var monthYear = date.ToString("MM-yyyy");  // Extracting month and year in MM-yyyy format
                return new MonthlySalesSummary  // Creating an object for each record
                {
                    MonthYear = monthYear,
                    ProductCategory = parts[1],  // Assigning the product category
                    TotalRevenue = float.Parse(parts[4], CultureInfo.InvariantCulture)  // Parsing the total revenue
                };
            }).ToList();  // Converting the processed data into a list

            // Group and shuffle the data by product category
            var groupedByCategory = allData
                .GroupBy(x => x.ProductCategory)  // Grouping data by product category
                .SelectMany(g => g.OrderBy(_ => Guid.NewGuid())  // Shuffling the data within each category
                                  .Select((x, i) => new { Data = x, Index = i, Count = g.Count() }))  // Create a new anonymous object with shuffled data
                .ToList();  // Convert to a list

            // 20% from each category for testing
            var testRawData = groupedByCategory
                .Where(x => x.Index < x.Count * 0.2)  // Selecting 20% of data for the test set
                .Select(x => x.Data)  // Selecting the data records
                .ToList();  // Convert to a list

            // 80% for training, aggregated
            var rawData = groupedByCategory
                .Where(x => x.Index >= x.Count * 0.2)  // Selecting the remaining 80% for training data
                .Select(x => x.Data)  // Selecting the data records
                .GroupBy(x => new { x.MonthYear, x.ProductCategory })  // Grouping by MonthYear and ProductCategory
                .Select(g => new MonthlySalesSummary  // Creating a new record for each group
                {
                    MonthYear = g.Key.MonthYear,  // MonthYear is the key
                    ProductCategory = g.Key.ProductCategory,  // ProductCategory is the key
                    TotalRevenue = g.Sum(x => x.TotalRevenue)  // Summing the revenue for each group
                })
                .ToList();  // Convert to a list

            // BALANCING: Compute weights inversely proportional to category frequency
            var counts = rawData
                .GroupBy(x => x.ProductCategory)  // Grouping by product category
                .ToDictionary(g => g.Key, g => g.Count());  // Counting how many entries per category

            // Assign weights to each record based on the inverse of category frequency
            foreach (var rec in rawData)
            {
                rec.Weight = 1f / counts[rec.ProductCategory];  // Assigning a weight to each record
            }

            // Write test data to a CSV file for later evaluation
            var testCsvLines = new List<string> { "Date,Category,ID,UnitsSold,Revenue" };  // Adding the header to the test CSV file
            testCsvLines.AddRange(testRawData.Select(d =>
            {
                var date = DateTime.ParseExact(d.MonthYear, "MM-yyyy", CultureInfo.InvariantCulture)
                                   .ToString("yyyy-MM-01");  // Converting the MonthYear to a standard date format
                return $"{date},{d.ProductCategory},dummyId,0,{d.TotalRevenue.ToString(CultureInfo.InvariantCulture)}";  // Formatting each line
            }));
            File.WriteAllLines(testFilePath, testCsvLines);  // Writing the test data to a file

            var trainingData = mlContext.Data.LoadFromEnumerable(rawData);  // Loading the training data into ML.NET
            Console.WriteLine($"✅ Training data ready with {rawData.Count} entries (including weights).\n");  // Output to show that the data is ready

            // Define base data prep pipeline for feature engineering
            var dataPrepPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(MonthlySalesSummary.ProductCategory))  // Converting labels (product categories) to keys
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("MonthYearEncoded", nameof(MonthlySalesSummary.MonthYear)))  // One-hot encoding the "MonthYear" column
                .Append(mlContext.Transforms.Concatenate("Features", "MonthYearEncoded", nameof(MonthlySalesSummary.TotalRevenue)))  // Combining features into a single vector
                .Append(mlContext.Transforms.NormalizeMinMax("Features"));  // Normalizing the feature values

            // Define multiple models to train
            var modelDefs = new List<(string name, IEstimator<ITransformer> trainer)>
            {
                // SDCA: Stochastic Dual Coordinate Ascent model for multiclass classification
                ("SDCA", mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    new SdcaMaximumEntropyMulticlassTrainer.Options
                    {
                        LabelColumnName = "Label",  // Label column is "ProductCategory"
                        FeatureColumnName = "Features",  // Features column is "Features"
                        ExampleWeightColumnName = nameof(MonthlySalesSummary.Weight)  // Weight column for each example
                    })),
                // LightGBM: LightGBM model for multiclass classification
                ("LightGBM", mlContext.MulticlassClassification.Trainers.LightGbm(
                    new LightGbmMulticlassTrainer.Options
                    {
                        LabelColumnName = "Label",  // Label column is "ProductCategory"
                        FeatureColumnName = "Features",  // Features column is "Features"
                        ExampleWeightColumnName = nameof(MonthlySalesSummary.Weight)  // Weight column for each example
                    })),
                // Averaged Perceptron: Another classification model using One-Versus-All strategy
                ("AveragedPerceptronOVA", mlContext.MulticlassClassification.Trainers
                    .OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(
                        new AveragedPerceptronTrainer.Options
                        {
                            LabelColumnName = "Label",  // Label column is "ProductCategory"
                            FeatureColumnName = "Features"  // Features column is "Features"
                        })))
            };

            double bestAccuracy = 0;  // Variable to store the best model's accuracy
            ITransformer bestModel = null;  // Variable to store the best model
            string bestModelName = "";  // Variable to store the name of the best model

            // Train and evaluate each model
            foreach (var (name, trainer) in modelDefs)
            {
                Console.WriteLine($"🔧 Training model: {name}...");  // Output to indicate which model is being trained
                var fullPipeline = dataPrepPipeline
                    .Append(trainer)  // Add the trainer to the pipeline
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));  // Convert the predicted key to a label

                var trainedModel = fullPipeline.Fit(trainingData);  // Train the model on the data
                var predictions = trainedModel.Transform(trainingData);  // Get the predictions on the training data
                var metrics = mlContext.MulticlassClassification.Evaluate(predictions);  // Evaluate the model's accuracy

                Console.WriteLine($"✅ {name} Accuracy (Macro): {metrics.MacroAccuracy:F4}");  // Output the model's accuracy

                if (metrics.MacroAccuracy > bestAccuracy)  // Check if this model has the best accuracy so far
                {
                    bestAccuracy = metrics.MacroAccuracy;  // Update the best accuracy
                    bestModel = trainedModel;  // Update the best model
                    bestModelName = name;  // Update the name of the best model
                }
            }

            Console.WriteLine($"\n🏆 Best model selected: {bestModelName} with MacroAccuracy: {bestAccuracy:F4}");  // Output the best model's name and accuracy

            // Final prediction using the best model
            Console.WriteLine("🔍 Aggregating and predicting from detailed test data...");  // Output to indicate the prediction step

            var testData = File.ReadAllLines(testFilePath)  // Read test data from the test file
                .Skip(1)  // Skip the header line
                .Select(line =>
                {
                    var parts = line.Split(',');  // Split each line by commas
                    var date = DateTime.Parse(parts[0]);  // Parse the date
                    var monthYear = date.ToString("MM-yyyy");  // Extract MonthYear
                    return new MonthlySalesSummary  // Create a MonthlySalesSummary object for each test record
                    {
                        MonthYear = monthYear,
                        ProductCategory = parts[1],  // ProductCategory
                        TotalRevenue = float.Parse(parts[4], CultureInfo.InvariantCulture)  // TotalRevenue
                    };
                })
                .GroupBy(x => new { x.MonthYear, x.ProductCategory })  // Group by MonthYear and ProductCategory
                .Select(g => new MonthlySalesSummary  // Create a new record for each group
                {
                    MonthYear = g.Key.MonthYear,
                    ProductCategory = g.Key.ProductCategory,
                    TotalRevenue = g.Sum(x => x.TotalRevenue)  // Sum the revenues within the group
                })
                .ToList();  // Convert the result to a list

            var predictionEngine = mlContext.Model.CreatePredictionEngine<MonthlySalesSummary, SalesPrediction>(bestModel);  // Create a prediction engine

            // Get the first month in the test data (for prediction)
            var nextMonth = testData.Select(x => DateTime.ParseExact(x.MonthYear, "MM-yyyy", CultureInfo.InvariantCulture))
                                    .OrderBy(d => d).First();  // Find the earliest month in the data
            var nextMonthStr = nextMonth.ToString("MM-yyyy");  // Format it as MM-yyyy
            var nextMonthData = testData.Where(x => x.MonthYear == nextMonthStr).ToList();  // Filter test data for the next month

            // Variables to track the best predicted category and revenue for next month
            string bestCategoryNextMonth = "";
            float highestRevenue = float.MinValue;

            // Loop through each record in next month's data
            foreach (var record in nextMonthData)
            {
                var prediction = predictionEngine.Predict(record);  // Get the model's prediction for this record
                var predictedCategory = prediction.PredictedLabel;  // Get the predicted product category

                if (record.TotalRevenue > highestRevenue)  // Check if this record has the highest revenue
                {
                    highestRevenue = record.TotalRevenue;  // Update the highest revenue
                    bestCategoryNextMonth = predictedCategory;  // Update the best category
                }
            }

            // Output the final prediction for the best category in the next month
            Console.WriteLine($"\n🔮 Prediction for next month ({nextMonthStr}):");
            Console.WriteLine($"🏆 Best Seller: {bestCategoryNextMonth} with Revenue: {highestRevenue}");  // Display the predicted best category and revenue
        }
    }
}