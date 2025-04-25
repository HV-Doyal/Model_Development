using System;
using Microsoft.ML.Data;

public class SalesData
{
    public DateTime Date { get; set; }
    public string ProductCategory { get; set; }
    public float QuantitySold { get; set; }
    public float PricePerUnit { get; set; }
    public float TotalPrice { get; set; }
}

public class MonthlySalesSummary
{
    // LoadColumn attributes ensure proper mapping when loading data
    [LoadColumn(0)]
    public string MonthYear { get; set; }

    [LoadColumn(1)]
    public string ProductCategory { get; set; }

    [LoadColumn(2)]
    public float TotalRevenue { get; set; }

    // Weight is used during training to balance classes
    [ColumnName("Weight")]
    public float Weight { get; set; }
}

// Optionally, define a simplified input type for prediction
public class TestInput
{
    [LoadColumn(0)]
    public string MonthYear { get; set; }

    [LoadColumn(1)]
    public float TotalRevenue { get; set; }
}

public class SalesPrediction
{
    // Maps to the label output of the model
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; }

    // Holds the scores for each category
    [ColumnName("Score")]
    public float[] Score { get; set; }
}
