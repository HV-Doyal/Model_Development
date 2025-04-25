using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using Model_Development;
using System.Globalization;

class Program
{
    static void Main(string[] args)
    {
        //PredictBestCategory.predict();
        regression.TrainAndSaveModel();
        regression.predict();
    }
}
