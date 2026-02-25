using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace DatasetGenerator;

/// <summary>
/// Generates relief heightmaps and contour line images.
/// Ported from Unity compute shader (Relief.compute) to pure C#.
/// </summary>
public class ReliefGenerator
{
    private readonly int _width;
    private readonly int _height;
    private readonly int _mapHeight;
    private float[] _result;
    private float[] _result2;
    private Vector2[] _gradients;
    private readonly Random _random;

    public ReliefGenerator(int width, int height, int mapHeight = 1000, int? seed = null)
    {
        _width = width;
        _height = height;
        _mapHeight = mapHeight;
        _result = new float[width * height];
        _result2 = new float[width * height];
        _random = seed.HasValue ? new Random(seed.Value) : new Random();

        InitializeGradients();
    }

    private void InitializeGradients()
    {
        _gradients = new Vector2[256];
        for (int i = 0; i < 256; i++)
        {
            float x = (float)(_random.NextDouble() * 2 - 1);
            float y = (float)(_random.NextDouble() * 2 - 1);
            float length = MathF.Sqrt(x * x + y * y);
            if (length > 0)
            {
                _gradients[i] = new Vector2(x / length, y / length);
            }
            else
            {
                _gradients[i] = new Vector2(1, 0);
            }
        }
    }

    #region Result Buffer Access (matches compute shader macros)

    private float GetResult(int x, int y) => _result[y * _width + x];
    private void SetResult(int x, int y, float value) => _result[y * _width + x] = value;

    private float GetResult2(int x, int y) => _result2[y * _width + x];
    private void SetResult2(int x, int y, float value) => _result2[y * _width + x] = value;

    private bool CheckValid(int x, int y) => x >= 0 && y >= 0 && x < _width && y < _height;

    private float Height(float x) => x / _mapHeight;

    private int ResultF(int x, int y, float horizontal)
    {
        return (int)(GetResult(x, y) * _mapHeight / horizontal);
    }

    #endregion

    #region Noise Generation (from Relief.compute)

    private float DropOff(float x)
    {
        float v = 1 - MathF.Abs(x);
        // nice normals version
        return 6 * MathF.Pow(v, 5) - 15 * MathF.Pow(v, 4) + 10 * MathF.Pow(v, 3);
    }

    private Vector2 Grad(int ux, int uy)
    {
        return _gradients[(ux + uy * 16) % 256];
    }

    private float NoiseSE(Vector2 g, float vx, float vy)
    {
        return (g.X * vx + g.Y * vy) * DropOff(vx) * DropOff(vy);
    }

    private float Noises(float vx, float vy, float t)
    {
        vx += t;
        vy += 0;

        int gix = (int)MathF.Floor(vx);
        int giy = (int)MathF.Floor(vy);
        int gizx = gix + 1;
        int giwy = giy + 1;

        float fracX = vx - MathF.Floor(vx);
        float fracY = vy - MathF.Floor(vy);

        return NoiseSE(Grad(gix, giy), fracX, fracY) +
               NoiseSE(Grad(gizx, giy), fracX - 1, fracY) +
               NoiseSE(Grad(gix, giwy), fracX, fracY - 1) +
               NoiseSE(Grad(gizx, giwy), fracX - 1, fracY - 1);
    }

    private float NoiseStep(float x, float y, int i, float res, float elongation, float t)
    {
        float scale = MathF.Pow(2.0f, i) / res;
        float xyX = x * elongation * scale;
        float xyY = y * elongation * scale;
        return (1.0f + Noises(xyX, xyY, t)) * MathF.Pow(2.0f, -(i + 2));
    }

    /// <summary>
    /// Generates Perlin noise-based terrain (kernel: noise from Relief.compute)
    /// </summary>
    public void GenerateNoise(int countOfHills, float elongation = 1f, float? t = null)
    {
        float tValue = t ?? (float)_random.NextDouble();
        float res = Math.Max(_width, _height);

        Parallel.For(0, _height, y =>
        {
            for (int x = 0; x < _width; x++)
            {
                float h = 0.0f;

                // Unrolled loop (matching compute shader behavior)
                h += NoiseStep(x, y, 0, res, elongation, tValue);
                if (countOfHills > 1) h += NoiseStep(x, y, 1, res, elongation, tValue);
                if (countOfHills > 2) h += NoiseStep(x, y, 2, res, elongation, tValue);
                if (countOfHills > 3) h += NoiseStep(x, y, 3, res, elongation, tValue);
                if (countOfHills > 4) h += NoiseStep(x, y, 4, res, elongation, tValue);
                if (countOfHills > 5) h += NoiseStep(x, y, 5, res, elongation, tValue);
                if (countOfHills > 6) h += NoiseStep(x, y, 6, res, elongation, tValue);
                if (countOfHills > 7) h += NoiseStep(x, y, 7, res, elongation, tValue);
                if (countOfHills > 8) h += NoiseStep(x, y, 8, res, elongation, tValue);

                float value = ((h - 0.5f) / 1.5f + 0.5f) % 1f;
                if (value < 0) value += 1f;
                SetResult(x, y, value);
            }
        });
    }

    #endregion

    #region Smoothing Filter (from Relief.compute - medium kernel)

    /// <summary>
    /// Applies smoothing filter (kernel: medium from Relief.compute)
    /// </summary>
    public void ApplySmoothing(int iterations = 4)
    {
        const float sum = 7.4f;

        for (int iter = 0; iter < iterations; iter++)
        {
            bool useResult2AsTarget = iter % 2 == 0;

            Parallel.For(0, _height, y =>
            {
                for (int x = 0; x < _width; x++)
                {
                    float ans;
                    if (x <= 1 || x >= _width - 2)
                    {
                        ans = useResult2AsTarget ? GetResult(x, y) : GetResult2(x, y);
                    }
                    else
                    {
                        float source(int sx, int sy) => useResult2AsTarget ? GetResult(sx, sy) : GetResult2(sx, sy);

                        ans = (1f * source(x - 1, y) +
                               3f * source(x, y) +
                               1f * source(x + 1, y) +
                               1.2f * source(x + 2, y) +
                               1.2f * source(x - 2, y)) / sum;
                    }

                    if (useResult2AsTarget)
                        SetResult2(x, y, ans);
                    else
                        SetResult(x, y, ans);
                }
            });
        }

        // Ensure final result is in _result
        if (iterations % 2 != 0)
        {
            Array.Copy(_result2, _result, _result.Length);
        }
    }

    #endregion

    #region Contour Line Detection (from Relief.compute - isHorizontal)

    /// <summary>
    /// Determines contour line intensity at a point (from isHorizontal in Relief.compute)
    /// </summary>
    private float IsHorizontal(int x, int y, float horizontal)
    {
        const float coef = 0.5f;
        float c = 0;
        int num = ResultF(x, y, horizontal);

        if (x > 0 && ResultF(x - 1, y, horizontal) != num) c += coef;
        if (y > 0 && ResultF(x, y - 1, horizontal) != num) c += coef;
        if (x < _width - 1 && ResultF(x + 1, y, horizontal) != num) c += coef;
        if (y < _height - 1 && ResultF(x, y + 1, horizontal) != num) c += coef;

        if (c > 0.5f) return MathF.Min(c, 1.0f);

        if (num % 5 == 0)
        {
            if (c == 0.5f) return 1f;
            if (x > 1 && ResultF(x - 2, y, horizontal) > num) c += coef;
            if (y > 1 && ResultF(x, y - 2, horizontal) > num) c += coef;
            if (x < _width - 2 && ResultF(x + 2, y, horizontal) > num) c += coef;
            if (y < _height - 2 && ResultF(x, y + 2, horizontal) > num) c += coef;
        }
        else if (num % 5 == 1)
        {
            if (c == 0.5f) return 1f;
            if (x > 1 && ResultF(x - 2, y, horizontal) < num) c += coef;
            if (y > 1 && ResultF(x, y - 2, horizontal) < num) c += coef;
            if (x < _width - 2 && ResultF(x + 2, y, horizontal) < num) c += coef;
            if (y < _height - 2 && ResultF(x, y + 2, horizontal) < num) c += coef;
        }

        return MathF.Min(c, 0.5f);
    }

    /// <summary>
    /// Calculates the gradient (slope direction) at a point. Returns normalized vector pointing downhill.
    /// </summary>
    private Vector2 GetGradient(int x, int y)
    {
        float hL = x > 0 ? GetResult(x - 1, y) : GetResult(x, y);
        float hR = x < _width - 1 ? GetResult(x + 1, y) : GetResult(x, y);
        float hD = y > 0 ? GetResult(x, y - 1) : GetResult(x, y);
        float hU = y < _height - 1 ? GetResult(x, y + 1) : GetResult(x, y);

        // Gradient points in direction of steepest descent (downhill)
        float gx = hL - hR;
        float gy = hD - hU;

        float len = MathF.Sqrt(gx * gx + gy * gy);
        if (len > 0.0001f)
        {
            return new Vector2(gx / len, gy / len);
        }
        return new Vector2(0, 0);
    }

    /// <summary>
    /// Checks if a point is on a contour line boundary (transition between height levels).
    /// </summary>
    private bool IsOnContour(int x, int y, float horizontal)
    {
        int num = ResultF(x, y, horizontal);
        if (x > 0 && ResultF(x - 1, y, horizontal) != num) return true;
        if (y > 0 && ResultF(x, y - 1, horizontal) != num) return true;
        if (x < _width - 1 && ResultF(x + 1, y, horizontal) != num) return true;
        if (y < _height - 1 && ResultF(x, y + 1, horizontal) != num) return true;
        return false;
    }

    /// <summary>
    /// Finds all local minima (depressions) in the heightmap, including edge depressions.
    /// </summary>
    private List<(int x, int y)> FindAllDepressions(int gridStep = 10, int checkRadius = 12)
    {
        var depressions = new List<(int x, int y)>();

        // Find interior depressions
        for (int gy = checkRadius; gy < _height - checkRadius; gy += gridStep)
        {
            for (int gx = checkRadius; gx < _width - checkRadius; gx += gridStep)
            {
                // Find local minimum in this cell
                int minX = gx, minY = gy;
                float minH = GetResult(gx, gy);

                for (int dy = -gridStep / 2; dy <= gridStep / 2; dy++)
                {
                    for (int dx = -gridStep / 2; dx <= gridStep / 2; dx++)
                    {
                        int px = gx + dx;
                        int py = gy + dy;
                        if (px >= 0 && px < _width && py >= 0 && py < _height)
                        {
                            float h = GetResult(px, py);
                            if (h < minH)
                            {
                                minH = h;
                                minX = px;
                                minY = py;
                            }
                        }
                    }
                }

                // Check if this is truly a local minimum (lower than all surrounding area)
                bool isMinimum = true;
                for (int dy = -checkRadius; dy <= checkRadius && isMinimum; dy += 3)
                {
                    for (int dx = -checkRadius; dx <= checkRadius && isMinimum; dx += 3)
                    {
                        if (dx == 0 && dy == 0) continue;
                        int px = minX + dx;
                        int py = minY + dy;
                        if (px >= 0 && px < _width && py >= 0 && py < _height)
                        {
                            if (GetResult(px, py) < minH)
                            {
                                isMinimum = false;
                            }
                        }
                    }
                }

                if (isMinimum)
                {
                    depressions.Add((minX, minY));
                }
            }
        }

        // Find edge depressions (terrain sloping out of map)
        int edgeCheckDepth = 15;
        int edgeStep = gridStep;

        // Top edge
        for (int x = edgeStep; x < _width - edgeStep; x += edgeStep)
        {
            float edgeH = GetResult(x, 0);
            float innerH = GetResult(x, edgeCheckDepth);
            if (edgeH < innerH - 0.01f)
            {
                depressions.Add((x, 0));
            }
        }

        // Bottom edge
        for (int x = edgeStep; x < _width - edgeStep; x += edgeStep)
        {
            float edgeH = GetResult(x, _height - 1);
            float innerH = GetResult(x, _height - 1 - edgeCheckDepth);
            if (edgeH < innerH - 0.01f)
            {
                depressions.Add((x, _height - 1));
            }
        }

        // Left edge
        for (int y = edgeStep; y < _height - edgeStep; y += edgeStep)
        {
            float edgeH = GetResult(0, y);
            float innerH = GetResult(edgeCheckDepth, y);
            if (edgeH < innerH - 0.01f)
            {
                depressions.Add((0, y));
            }
        }

        // Right edge
        for (int y = edgeStep; y < _height - edgeStep; y += edgeStep)
        {
            float edgeH = GetResult(_width - 1, y);
            float innerH = GetResult(_width - 1 - edgeCheckDepth, y);
            if (edgeH < innerH - 0.01f)
            {
                depressions.Add((_width - 1, y));
            }
        }

        return depressions;
    }

    /// <summary>
    /// Draws slope tick marks only on depression (hole) contours, pointing inward.
    /// </summary>
    private void DrawSlopeTicks(Image<Rgba32> image, float horizontal, Rgba32 color, int tickLength = 6, int tickSpacing = 25)
    {
        // Find all depressions first
        var depressions = FindAllDepressions();

        // For each depression, find nearby contour and draw ticks
        foreach (var (depX, depY) in depressions)
        {
            // Search outward from depression to find the first contour
            int searchRadius = 30;
            var ticksDrawn = new HashSet<(int, int)>();

            for (int angle = 0; angle < 360; angle += 45)
            {
                float rad = angle * MathF.PI / 180f;
                float dirX = MathF.Cos(rad);
                float dirY = MathF.Sin(rad);

                // Walk outward until we hit a contour
                for (int dist = 3; dist < searchRadius; dist++)
                {
                    int px = depX + (int)(dirX * dist);
                    int py = depY + (int)(dirY * dist);

                    if (px < 0 || px >= _width || py < 0 || py >= _height)
                        break;

                    float contourStrength = IsHorizontal(px, py, horizontal);
                    if (contourStrength >= 0.5f)
                    {
                        // Found contour - check we haven't drawn a tick nearby
                        int gridKey = (px / tickSpacing, py / tickSpacing).GetHashCode();
                        var cellKey = (px / tickSpacing, py / tickSpacing);

                        if (!ticksDrawn.Contains(cellKey))
                        {
                            ticksDrawn.Add(cellKey);

                            // Draw tick pointing toward depression (inward)
                            var gradient = GetGradient(px, py);
                            for (int i = 1; i <= tickLength; i++)
                            {
                                int tx = px + (int)(gradient.X * i);
                                int ty = py + (int)(gradient.Y * i);

                                if (tx >= 0 && tx < _width && ty >= 0 && ty < _height)
                                {
                                    image[tx, ty] = color;
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    #endregion

    #region Public API

    /// <summary>
    /// Generates relief data: heightmap and contour lines image.
    /// </summary>
    /// <param name="countOfHills">Number of noise octaves (1-9)</param>
    /// <param name="horizontal">Height interval between contour lines</param>
    /// <param name="elongation">Terrain stretch factor</param>
    /// <param name="contourColor">Color for contour lines (default: brown)</param>
    /// <param name="drawSlopeTicks">Whether to draw slope tick marks pointing downhill</param>
    /// <param name="tickSpacing">Spacing between slope tick marks in pixels</param>
    /// <returns>Tuple of (heightmap as float[,], contour lines image)</returns>
    public (float[,] Heightmap, Image<Rgba32> ContourImage) Generate(
        int countOfHills = 4,
        float horizontal = 5f,
        float elongation = 1f,
        Rgba32? contourColor = null,
        bool drawSlopeTicks = true,
        int tickSpacing = 25)
    {
        var color = contourColor ?? new Rgba32(139, 90, 43); // brown

        // Generate noise terrain
        GenerateNoise(countOfHills, elongation);

        // Apply smoothing
        ApplySmoothing(4);

        // Create heightmap array
        float[,] heightmap = new float[_width, _height];
        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                heightmap[x, y] = GetResult(x, y);
            }
        }

        // Create contour image
        var image = new Image<Rgba32>(_width, _height, new Rgba32(255, 255, 255, 0));

        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                float g = IsHorizontal(x, y, horizontal);
                if (g != 0)
                {
                    byte alpha = (byte)(g * 255);
                    image[x, y] = new Rgba32(color.R, color.G, color.B, alpha);
                }
            }
        }

        // Draw slope tick marks pointing downhill
        if (drawSlopeTicks)
        {
            DrawSlopeTicks(image, horizontal, color, tickLength: 6, tickSpacing: tickSpacing);
        }

        return (heightmap, image);
    }

    /// <summary>
    /// Gets the current heightmap as a 2D array.
    /// </summary>
    public float[,] GetHeightmap()
    {
        float[,] heightmap = new float[_width, _height];
        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                heightmap[x, y] = GetResult(x, y);
            }
        }
        return heightmap;
    }

    /// <summary>
    /// Sets the heightmap from a 2D array.
    /// </summary>
    public void SetHeightmap(float[,] heightmap)
    {
        if (heightmap.GetLength(0) != _width || heightmap.GetLength(1) != _height)
            throw new ArgumentException($"Heightmap dimensions must be {_width}x{_height}");

        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                SetResult(x, y, heightmap[x, y]);
            }
        }
    }

    /// <summary>
    /// Creates only the contour lines image from current heightmap.
    /// </summary>
    /// <param name="horizontal">Height interval between contour lines</param>
    /// <param name="contourColor">Color for contour lines (default: brown)</param>
    /// <param name="drawSlopeTicks">Whether to draw slope tick marks pointing downhill</param>
    /// <param name="tickSpacing">Spacing between slope tick marks in pixels</param>
    public Image<Rgba32> CreateContourImage(
        float horizontal = 5f,
        Rgba32? contourColor = null,
        bool drawSlopeTicks = true,
        int tickSpacing = 25)
    {
        var color = contourColor ?? new Rgba32(139, 90, 43);
        var image = new Image<Rgba32>(_width, _height, new Rgba32(255, 255, 255, 0));

        for (int y = 0; y < _height; y++)
        {
            for (int x = 0; x < _width; x++)
            {
                float g = IsHorizontal(x, y, horizontal);
                if (g != 0)
                {
                    byte alpha = (byte)(g * 255);
                    image[x, y] = new Rgba32(color.R, color.G, color.B, alpha);
                }
            }
        }

        // Draw slope tick marks pointing downhill
        if (drawSlopeTicks)
        {
            DrawSlopeTicks(image, horizontal, color, tickLength: 6, tickSpacing: tickSpacing);
        }

        return image;
    }

    #endregion
}

internal struct Vector2
{
    public float X;
    public float Y;

    public Vector2(float x, float y)
    {
        X = x;
        Y = y;
    }
}
