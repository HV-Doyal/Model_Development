using Microsoft.ML.Data;  // Importing necessary namespaces for ML.NET.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Globalization;
using Microsoft.ML;

namespace Model_Development
{
    // Class representing the input data for the model
    public class MonthlySalesData
    {
        public string Month { get; set; }      // "05" for May (month of the sale)
        public string ProductCategory { get; set; }  // Category of the product sold (e.g., Electronics, Furniture)
        public float TotalRevenue { get; set; } // Total revenue generated for the product in that month
    }

    // Class representing the output predictions of the model
    public class MonthlySalesPrediction
    {
        [ColumnName("Score")]  // The predicted value column from the model, labeled as "Score"
        public float PredictedRevenue { get; set; } // Predicted revenue based on the features
    }

    public static class regression
    {
        // Path to the training data file
        static string trainFilePath = "C:\\Users\\harsh\\OneDrive\\Desktop\\Model_Development\\Dataset\\train.csv";
        static string modelFilePath = "trained_model.zip";  // Path where the trained model will be saved

        // Method to train the model and save it to a file
        public static void TrainAndSaveModel()
        {
            var mlContext = new MLContext();

            // Informing the user that data loading and preparation is starting
            Console.WriteLine("📊 Loading and preparing training data...");

            // STEP 1: Load and preprocess training data
            var rawData = File.ReadAllLines(trainFilePath)
                .Skip(1)  // Skipping the header row
                .Select(line =>
                {
                    var parts = line.Split(',');  // Splitting each line into components by commas
                    var date = DateTime.Parse(parts[0]);  // Parsing the first part as a DateTime (the date)
                    return new MonthlySalesData
                    {
                        Month = date.ToString("MM"), // Extracting just the month part (in "MM" format)
                        ProductCategory = parts[1],  // Extracting the product category
                        TotalRevenue = float.Parse(parts[4], CultureInfo.InvariantCulture)  // Parsing the total revenue (5th column)
                    };
                })
                .ToList();  // Converting the parsed data into a list

            // Loading the list of data into an IDataView (ML.NET's internal data structure)
            IDataView trainingData = mlContext.Data.LoadFromEnumerable(rawData);

            // Confirming that the data has been loaded
            Console.WriteLine("✅ Training data loaded.\n");

            // STEP 2: Define ML pipeline
            Console.WriteLine("🔧 Creating regression pipeline...");

            // Creating a pipeline that processes the data
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("ProductCategoryEncoded", nameof(MonthlySalesData.ProductCategory))  // One-hot encoding for the product category
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("MonthEncoded", nameof(MonthlySalesData.Month)))  // One-hot encoding for the month
                .Append(mlContext.Transforms.Concatenate("Features", "ProductCategoryEncoded", "MonthEncoded"))  // Concatenating the features into one feature column
                .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "TotalRevenue", featureColumnName: "Features"));  // Using the FastTree regression algorithm

            // STEP 3: Train the model
            Console.WriteLine("🏋️ Training model...");
            var model = pipeline.Fit(trainingData);  // Training the model on the prepared data
            Console.WriteLine("✅ Model training complete.\n");

            // STEP 4: Save the model to a file
            Console.WriteLine($"📦 Saving the model to {modelFilePath}...");
            mlContext.Model.Save(model, trainingData.Schema, modelFilePath);
            Console.WriteLine("✅ Model saved successfully.\n");
        }

        // Method to load the trained model and make predictions
        public static void predict()
        {
            var mlContext = new MLContext();

            // Load the trained model from file
            var model = mlContext.Model.Load(modelFilePath, out var modelInputSchema);
            Console.WriteLine("✅ Model loaded.\n");

            // Get unique product categories from the training data
            var rawData = File.ReadAllLines(trainFilePath)
                .Skip(1)
                .Select(line =>
                {
                    var parts = line.Split(',');
                    return new MonthlySalesData
                    {
                        Month = DateTime.Parse(parts[0]).ToString("MM"),
                        ProductCategory = parts[1]
                    };
                })
                .ToList();
            var productCategories = rawData.Select(r => r.ProductCategory).Distinct().ToList();

            // Create a prediction engine from the loaded model
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MonthlySalesData, MonthlySalesPrediction>(model);

            string monthToPredict = "05"; // Setting the month to predict as May

            Console.WriteLine($"📅 Predicting revenue for all categories in {monthToPredict}...");

            // Looping through all unique product categories to predict the revenue for each
            foreach (var category in productCategories)
            {
                var input = new MonthlySalesData  // Creating input data for prediction (using May as the month)
                {
                    ProductCategory = category,
                    Month = monthToPredict // May
                };

                // Making a prediction using the prediction engine
                var prediction = predictionEngine.Predict(input);
                Console.WriteLine($"📊 Predicted revenue for '{category}' in {monthToPredict}: ${prediction.PredictedRevenue:F2}");  // Displaying the predicted revenue
            }
        }
    }
}
