using DatasetGenerator;

// Dataset generation configuration
int width = 512;
int height = 512;
int mapHeight = 1000;
int datasetSize = 10000;
string outputDir = "path/to/dataset";

Console.WriteLine("=== Relief Map Dataset Generator ===");
Console.WriteLine($"Image size: {width}x{height}");
Console.WriteLine($"Dataset size: {datasetSize} samples");
Console.WriteLine();

var generator = new DatasetGenerator.DatasetGenerator(width, height, mapHeight, outputDir);

generator.GenerateDataset(
    count: datasetSize,
    minCountOfHills: 2,
    maxCountOfHills: 6,
    minElongation: 0.3f,
    maxElongation: 3.0f,
    horizontal: 5f
);

Console.WriteLine();
Console.WriteLine("Output files per sample:");
Console.WriteLine("  - {number}.jpg          : Contour line image");
Console.WriteLine("  - {number}.metadata.json: Parameters (seed, countOfHills, elongation)");
Console.WriteLine("  - {number}.data         : Raw heightmap (binary floats, row-major)");
Console.WriteLine();
Console.WriteLine($"Dataset saved to: {outputDir}");