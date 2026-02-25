using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;

namespace DatasetGenerator;

/// <summary>
/// Interactive 3D visualization for heightmap data using OpenGL.
/// Controls: Left-drag to rotate, scroll to zoom, R to reset view.
/// </summary>
public class ReliefVisualizer : GameWindow
{
    private readonly float[,] _heightmap;
    private readonly int _mapWidth;
    private readonly int _mapHeight;

    private int _vao;
    private int _vbo;
    private int _ebo;
    private int _shaderProgram;
    private int _indexCount;

    // Camera
    private float _cameraDistance = 2.0f;
    private float _cameraYaw = 45f;
    private float _cameraPitch = 35f;
    private Vector2 _lastMousePos;
    private bool _isDragging;

    // Uniforms
    private int _modelLoc;
    private int _viewLoc;
    private int _projectionLoc;
    private int _lightDirLoc;
    private int _minHeightLoc;
    private int _maxHeightLoc;

    private float _minHeight;
    private float _maxHeight;

    private const string VertexShaderSource = @"
#version 330 core
layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in float aHeight;

out vec3 FragPos;
out vec3 Normal;
out float Height;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPosition, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    Height = aHeight;
    gl_Position = projection * view * model * vec4(aPosition, 1.0);
}
";

    private const string FragmentShaderSource = @"
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in float Height;

out vec4 FragColor;

uniform vec3 lightDir;
uniform float minHeight;
uniform float maxHeight;

vec3 getTerrainColor(float h) {
    float t = (h - minHeight) / (maxHeight - minHeight + 0.0001);

    // Color gradient: deep blue -> green -> yellow -> brown -> white
    vec3 water = vec3(0.1, 0.3, 0.6);
    vec3 grass = vec3(0.2, 0.6, 0.2);
    vec3 dirt = vec3(0.6, 0.5, 0.3);
    vec3 rock = vec3(0.5, 0.45, 0.4);
    vec3 snow = vec3(0.95, 0.95, 0.95);

    if (t < 0.2) return mix(water, grass, t / 0.2);
    if (t < 0.4) return mix(grass, dirt, (t - 0.2) / 0.2);
    if (t < 0.7) return mix(dirt, rock, (t - 0.4) / 0.3);
    return mix(rock, snow, (t - 0.7) / 0.3);
}

void main()
{
    vec3 norm = normalize(Normal);
    vec3 lightDirection = normalize(lightDir);

    float ambient = 0.3;
    float diff = max(dot(norm, lightDirection), 0.0);
    float lighting = ambient + diff * 0.7;

    vec3 terrainColor = getTerrainColor(Height);
    FragColor = vec4(terrainColor * lighting, 1.0);
}
";

    public ReliefVisualizer(float[,] heightmap, string title = "Relief Visualizer")
        : base(
            GameWindowSettings.Default,
            new NativeWindowSettings
            {
                ClientSize = new Vector2i(1024, 768),
                Title = title,
                APIVersion = new Version(3, 3)
            })
    {
        _heightmap = heightmap;
        _mapWidth = heightmap.GetLength(0);
        _mapHeight = heightmap.GetLength(1);
    }

    protected override void OnLoad()
    {
        base.OnLoad();

        GL.ClearColor(0.2f, 0.25f, 0.3f, 1.0f);
        GL.Enable(EnableCap.DepthTest);

        FindHeightRange();
        CreateMesh();
        CreateShaders();

        Console.WriteLine("Controls: Left-drag to rotate, scroll to zoom, R to reset, Esc to close");
    }

    private void FindHeightRange()
    {
        _minHeight = float.MaxValue;
        _maxHeight = float.MinValue;

        for (int y = 0; y < _mapHeight; y++)
        {
            for (int x = 0; x < _mapWidth; x++)
            {
                float h = _heightmap[x, y];
                if (h < _minHeight) _minHeight = h;
                if (h > _maxHeight) _maxHeight = h;
            }
        }
    }

    private void CreateMesh()
    {
        // Downsample for performance if needed
        int step = Math.Max(1, Math.Max(_mapWidth, _mapHeight) / 256);
        int gridW = _mapWidth / step;
        int gridH = _mapHeight / step;

        // Vertex data: position (3) + normal (3) + height (1) = 7 floats per vertex
        var vertices = new List<float>();
        var indices = new List<uint>();

        float scaleXZ = 1.0f / Math.Max(gridW, gridH);
        float scaleY = 0.5f; // Height exaggeration

        // Generate vertices
        for (int gy = 0; gy < gridH; gy++)
        {
            for (int gx = 0; gx < gridW; gx++)
            {
                int sx = gx * step;
                int sy = gy * step;
                float h = _heightmap[sx, sy];

                float x = (gx - gridW / 2.0f) * scaleXZ;
                float z = (gy - gridH / 2.0f) * scaleXZ;
                float y = (h - _minHeight) / (_maxHeight - _minHeight + 0.0001f) * scaleY;

                // Calculate normal using central differences
                float hL = gx > 0 ? _heightmap[(gx - 1) * step, sy] : h;
                float hR = gx < gridW - 1 ? _heightmap[(gx + 1) * step, sy] : h;
                float hD = gy > 0 ? _heightmap[sx, (gy - 1) * step] : h;
                float hU = gy < gridH - 1 ? _heightmap[sx, (gy + 1) * step] : h;

                var normal = Vector3.Normalize(new Vector3(hL - hR, 2.0f * scaleXZ / scaleY, hD - hU));

                // Position
                vertices.Add(x);
                vertices.Add(y);
                vertices.Add(z);
                // Normal
                vertices.Add(normal.X);
                vertices.Add(normal.Y);
                vertices.Add(normal.Z);
                // Height
                vertices.Add(h);
            }
        }

        // Generate indices for triangles
        for (int gy = 0; gy < gridH - 1; gy++)
        {
            for (int gx = 0; gx < gridW - 1; gx++)
            {
                uint topLeft = (uint)(gy * gridW + gx);
                uint topRight = topLeft + 1;
                uint bottomLeft = (uint)((gy + 1) * gridW + gx);
                uint bottomRight = bottomLeft + 1;

                // First triangle
                indices.Add(topLeft);
                indices.Add(bottomLeft);
                indices.Add(topRight);

                // Second triangle
                indices.Add(topRight);
                indices.Add(bottomLeft);
                indices.Add(bottomRight);
            }
        }

        _indexCount = indices.Count;

        // Create VAO
        _vao = GL.GenVertexArray();
        GL.BindVertexArray(_vao);

        // Create VBO
        _vbo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);
        GL.BufferData(BufferTarget.ArrayBuffer, vertices.Count * sizeof(float), vertices.ToArray(), BufferUsageHint.StaticDraw);

        // Create EBO
        _ebo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ElementArrayBuffer, _ebo);
        GL.BufferData(BufferTarget.ElementArrayBuffer, indices.Count * sizeof(uint), indices.ToArray(), BufferUsageHint.StaticDraw);

        // Position attribute
        GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 7 * sizeof(float), 0);
        GL.EnableVertexAttribArray(0);

        // Normal attribute
        GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 7 * sizeof(float), 3 * sizeof(float));
        GL.EnableVertexAttribArray(1);

        // Height attribute
        GL.VertexAttribPointer(2, 1, VertexAttribPointerType.Float, false, 7 * sizeof(float), 6 * sizeof(float));
        GL.EnableVertexAttribArray(2);

        GL.BindVertexArray(0);

        Console.WriteLine($"Mesh created: {gridW}x{gridH} grid, {_indexCount / 3} triangles");
    }

    private void CreateShaders()
    {
        int vertexShader = GL.CreateShader(ShaderType.VertexShader);
        GL.ShaderSource(vertexShader, VertexShaderSource);
        GL.CompileShader(vertexShader);
        CheckShaderCompile(vertexShader, "Vertex");

        int fragmentShader = GL.CreateShader(ShaderType.FragmentShader);
        GL.ShaderSource(fragmentShader, FragmentShaderSource);
        GL.CompileShader(fragmentShader);
        CheckShaderCompile(fragmentShader, "Fragment");

        _shaderProgram = GL.CreateProgram();
        GL.AttachShader(_shaderProgram, vertexShader);
        GL.AttachShader(_shaderProgram, fragmentShader);
        GL.LinkProgram(_shaderProgram);

        GL.GetProgram(_shaderProgram, GetProgramParameterName.LinkStatus, out int success);
        if (success == 0)
        {
            string infoLog = GL.GetProgramInfoLog(_shaderProgram);
            Console.WriteLine($"Shader link error: {infoLog}");
        }

        GL.DeleteShader(vertexShader);
        GL.DeleteShader(fragmentShader);

        _modelLoc = GL.GetUniformLocation(_shaderProgram, "model");
        _viewLoc = GL.GetUniformLocation(_shaderProgram, "view");
        _projectionLoc = GL.GetUniformLocation(_shaderProgram, "projection");
        _lightDirLoc = GL.GetUniformLocation(_shaderProgram, "lightDir");
        _minHeightLoc = GL.GetUniformLocation(_shaderProgram, "minHeight");
        _maxHeightLoc = GL.GetUniformLocation(_shaderProgram, "maxHeight");
    }

    private void CheckShaderCompile(int shader, string type)
    {
        GL.GetShader(shader, ShaderParameter.CompileStatus, out int success);
        if (success == 0)
        {
            string infoLog = GL.GetShaderInfoLog(shader);
            Console.WriteLine($"{type} shader compile error: {infoLog}");
        }
    }

    protected override void OnRenderFrame(FrameEventArgs args)
    {
        base.OnRenderFrame(args);

        GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

        GL.UseProgram(_shaderProgram);

        // Calculate camera position from spherical coordinates
        float yawRad = MathHelper.DegreesToRadians(_cameraYaw);
        float pitchRad = MathHelper.DegreesToRadians(_cameraPitch);

        var cameraPos = new Vector3(
            _cameraDistance * MathF.Cos(pitchRad) * MathF.Sin(yawRad),
            _cameraDistance * MathF.Sin(pitchRad),
            _cameraDistance * MathF.Cos(pitchRad) * MathF.Cos(yawRad)
        );

        var model = Matrix4.Identity;
        var view = Matrix4.LookAt(cameraPos, Vector3.Zero, Vector3.UnitY);
        var projection = Matrix4.CreatePerspectiveFieldOfView(
            MathHelper.DegreesToRadians(45f),
            (float)Size.X / Size.Y,
            0.01f,
            100f
        );

        GL.UniformMatrix4(_modelLoc, false, ref model);
        GL.UniformMatrix4(_viewLoc, false, ref view);
        GL.UniformMatrix4(_projectionLoc, false, ref projection);

        // Light direction (from top-left)
        var lightDir = Vector3.Normalize(new Vector3(0.5f, 1.0f, 0.3f));
        GL.Uniform3(_lightDirLoc, lightDir);
        GL.Uniform1(_minHeightLoc, _minHeight);
        GL.Uniform1(_maxHeightLoc, _maxHeight);

        GL.BindVertexArray(_vao);
        GL.DrawElements(PrimitiveType.Triangles, _indexCount, DrawElementsType.UnsignedInt, 0);

        SwapBuffers();
    }

    protected override void OnUpdateFrame(FrameEventArgs args)
    {
        base.OnUpdateFrame(args);

        if (KeyboardState.IsKeyDown(Keys.Escape))
        {
            Close();
        }

        if (KeyboardState.IsKeyPressed(Keys.R))
        {
            _cameraDistance = 2.0f;
            _cameraYaw = 45f;
            _cameraPitch = 35f;
        }

        // Mouse rotation
        var mousePos = new Vector2(MouseState.X, MouseState.Y);

        if (MouseState.IsButtonDown(MouseButton.Left))
        {
            if (_isDragging)
            {
                float deltaX = mousePos.X - _lastMousePos.X;
                float deltaY = mousePos.Y - _lastMousePos.Y;
                _cameraYaw += deltaX * 0.5f;
                _cameraPitch += deltaY * 0.3f;
                _cameraPitch = Math.Clamp(_cameraPitch, -89f, 89f);
            }
            _isDragging = true;
        }
        else
        {
            _isDragging = false;
        }

        _lastMousePos = mousePos;
    }

    protected override void OnMouseWheel(MouseWheelEventArgs e)
    {
        base.OnMouseWheel(e);
        _cameraDistance -= e.OffsetY * 0.2f;
        _cameraDistance = Math.Clamp(_cameraDistance, 0.5f, 10f);
    }

    protected override void OnResize(ResizeEventArgs e)
    {
        base.OnResize(e);
        GL.Viewport(0, 0, e.Width, e.Height);
    }

    protected override void OnUnload()
    {
        GL.DeleteVertexArray(_vao);
        GL.DeleteBuffer(_vbo);
        GL.DeleteBuffer(_ebo);
        GL.DeleteProgram(_shaderProgram);
        base.OnUnload();
    }

    /// <summary>
    /// Static helper to visualize a heightmap.
    /// </summary>
    public static void Show(float[,] heightmap, string title = "Relief Visualizer")
    {
        using var visualizer = new ReliefVisualizer(heightmap, title);
        visualizer.Run();
    }
}
