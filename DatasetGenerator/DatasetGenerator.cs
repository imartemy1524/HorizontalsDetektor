using System.Text.Json;
using System.Threading;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace DatasetGenerator;

/// <summary>
/// Generates a dataset of relief maps with metadata for machine learning training.
/// </summary>
public class DatasetGenerator
{
    private readonly int _width;
    private readonly int _height;
    private readonly int _mapHeight;
    private readonly string _outputDirectory;

    public DatasetGenerator(int width, int height, int mapHeight, string outputDirectory)
    {
        _width = width;
        _height = height;
        _mapHeight = mapHeight;
        _outputDirectory = outputDirectory;

        if (!Directory.Exists(_outputDirectory))
        {
            Directory.CreateDirectory(_outputDirectory);
        }
    }

    /// <summary>
    /// Metadata structure for each generated sample.
    /// </summary>
    public class SampleMetadata
    {
        public int Seed { get; set; }
        public int CountOfHills { get; set; }
        public float Elongation { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
    }

    /// <summary>
    /// Generates a dataset with the specified number of samples.
    /// </summary>
    /// <param name="count">Number of samples to generate</param>
    /// <param name="minCountOfHills">Minimum count of hills (default: 2)</param>
    /// <param name="maxCountOfHills">Maximum count of hills (default: 6)</param>
    /// <param name="minElongation">Minimum elongation (default: 0.3)</param>
    /// <param name="maxElongation">Maximum elongation (default: 3.0)</param>
    /// <param name="horizontal">Contour line interval (default: 5)</param>
    public void GenerateDataset(
        int count,
        int minCountOfHills = 2,
        int maxCountOfHills = 6,
        float minElongation = 0.3f,
        float maxElongation = 3.0f,
        float horizontal = 5f)
    {
        var random = new Random();
        var jsonOptions = new JsonSerializerOptions { WriteIndented = true };

        Console.WriteLine($"Generating {count} samples to: {_outputDirectory}");
        Console.WriteLine($"Parameters: countOfHills=[{minCountOfHills}-{maxCountOfHills}], elongation=[{minElongation:F2}-{maxElongation:F2}]");

        var startTime = DateTime.Now;
        int progressStep = Math.Max(1, count / 100);
        int completed = 0;

        Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            i =>
            {
                // Generate random parameters
                int seed = random.Next();
                int countOfHills = random.Next(minCountOfHills, maxCountOfHills + 1);
                float elongation = minElongation + (float)random.NextDouble() * (maxElongation - minElongation);

                // Generate the relief
                var generator = new ReliefGenerator(_width, _height, _mapHeight, seed);
                var (heightmap, contourImage) = generator.Generate(
                    countOfHills: countOfHills,
                    horizontal: horizontal,
                    elongation: elongation,
                    drawSlopeTicks: true
                );

                // Create metadata
                var metadata = new SampleMetadata
                {
                    Seed = seed,
                    CountOfHills = countOfHills,
                    Elongation = elongation,
                    Width = _width,
                    Height = _height
                };

                // Save files
                string basePath = Path.Combine(_outputDirectory, i.ToString());

                // Save contour image as JPG
                using (var jpgImage = new Image<Rgba32>(_width, _height, new Rgba32(255, 255, 255, 255)))
                {
                    // Composite contour lines onto white background
                    jpgImage.Mutate(ctx => ctx.DrawImage(contourImage, 1f));
                    jpgImage.SaveAsJpeg($"{basePath}.jpg");
                }
                contourImage.Dispose();

                // Save metadata as JSON
                string jsonContent = JsonSerializer.Serialize(metadata, jsonOptions);
                File.WriteAllText($"{basePath}.metadata.json", jsonContent);

                // Save heightmap as raw binary floats
                SaveHeightmapBinary(heightmap, $"{basePath}.data");

                // Progress reporting with thread-safe counter
                int current = Interlocked.Increment(ref completed);
                if (current % progressStep == 0 || current == count)
                {
                    var elapsed = DateTime.Now - startTime;
                    var eta = TimeSpan.FromTicks(elapsed.Ticks * count / current) - elapsed;
                    Console.Write($"\rProgress: {current}/{count} ({current * 100 / count}%) - ETA: {eta:mm\\:ss}    ");
                }
            });

        Console.WriteLine();
        Console.WriteLine($"Dataset generation complete! Total time: {DateTime.Now - startTime:mm\\:ss}");
    }

    /// <summary>
    /// Saves a heightmap as raw binary float data.
    /// Format: row-major order, width*height floats (4 bytes each)
    /// </summary>
    private static void SaveHeightmapBinary(float[,] heightmap, string filePath)
    {
        int width = heightmap.GetLength(0);
        int height = heightmap.GetLength(1);

        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        using var writer = new BinaryWriter(stream);

        // Write data in row-major order (y then x, matching typical image conventions)
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                writer.Write(heightmap[x, y]);
            }
        }
    }

    /// <summary>
    /// Loads a heightmap from raw binary float data.
    /// </summary>
    public static float[,] LoadHeightmapBinary(string filePath, int width, int height)
    {
        var heightmap = new float[width, height];

        using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var reader = new BinaryReader(stream);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                heightmap[x, y] = reader.ReadSingle();
            }
        }

        return heightmap;
    }
}
