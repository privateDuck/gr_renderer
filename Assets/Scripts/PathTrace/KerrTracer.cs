using System;
using UnityEditor;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

//[CreateAssetMenu(fileName = "KerrTracer", menuName = "Path Tracing/Kerr Tracer", order = 1)]
public class KerrTracer : MonoBehaviour
{
    public ComputeShader shader;
    public Texture2D skyEquirect; // assign an equirectangular sky texture
    private Camera renderCamera;   // camera to derive position and forward from
    public int width = 512;
    public int height = 512;

    public Material material;

    [Header("Kerr Params")]
    [Range(0, 1)] public float a = 0.998f;
    public float m = 1.0f;
    public float Q = 0.5f;

    [Header("Disk Params")]
    public float T_peak = 50000;
    public float DiskMaxRadius = 15f;

    [Header("Turbulence params")]
    [Range(0, 1)] public float turbulenceStrength = 0.45f;
    public float _disk_str_azimuth = 0.5f;
    public float _disk_str_radial = 1.0f;
    public Vector3 shift = Vector3.zero;
    public float rot_speeed = 2.0f;
    public float gain = 0.5f;


    [HideInInspector] public RenderTexture target;
    int kernel;
    private readonly Vector3[] _frustumCorners = new Vector3[4];
    private Transform cameraTransform;
    private float time_ = 0.0f;

    private void Start()
    {
        cameraTransform = renderCamera.transform;
        SetupCompute();
    }

    private void LateUpdate()
    {
        Execute();
    }

    private void OnValidate()
    {
        if (Application.isPlaying)
            SetupCompute();
    }

    public void SetupCompute()
    {
        renderCamera = Camera.main;
        if (shader == null || renderCamera == null) return;
        CreateRenderTexture();
        var (mf, mr) = ComputePeakFlux();

        shader.SetTexture(kernel, ShaderIDs.Result_ID, target);
        if (skyEquirect != null)
            shader.SetTexture(kernel, ShaderIDs.SkyTex_ID, skyEquirect);

        shader.SetInt(ShaderIDs.Width_ID, width);
        shader.SetInt(ShaderIDs.Height_ID, height);

        // set kerr params
        shader.SetFloat(ShaderIDs.a_ID, a);
        shader.SetFloat(ShaderIDs.m_ID, m);
        shader.SetFloat(ShaderIDs.Q_ID, Q);
        shader.SetFloat(ShaderIDs.r_isco_ID, ComputeISCO());
        shader.SetFloat(ShaderIDs.peak_flux_ID, mf);
        shader.SetFloat(ShaderIDs.peak_flux_r_ID, mr);
        shader.SetFloat(ShaderIDs.disk_max_radius_ID, DiskMaxRadius);
        shader.SetFloat(ShaderIDs.Tpeak_ID, T_peak);

        shader.SetFloat(ShaderIDs.disk_str_azimuth_ID, _disk_str_azimuth);
        shader.SetFloat(ShaderIDs.disk_str_radial_ID, _disk_str_radial);
        shader.SetVector(ShaderIDs.shift_ID, new Vector4(shift.x, shift.y, shift.z, 0.0f));
        shader.SetFloat(ShaderIDs.rot_speed_ID, rot_speeed);
        shader.SetFloat(ShaderIDs.gain_ID, gain);
        shader.SetFloat(ShaderIDs.turbulenceStrength_ID, turbulenceStrength);
        shader.SetFloat(ShaderIDs.Time_ID, time_);

        material.SetTexture("_mainTex", target);
    }


    private float ComputeISCO()
    {
        float j = a / m;
        float Z1 = 1.0f + ((float)Math.Cbrt(1.0 - j * j) * (float)(Math.Cbrt(1.0 + j) + Math.Cbrt(1.0 - j)));
        float Z2 = Mathf.Sqrt(3.0f * j * j + Z1 * Z1);
        return m * (3.0f + Z2 - Mathf.Sqrt((3.0f - Z1) * (3.0f + Z1 + 2.0f * Z2)));
    }

    private (float, float) ComputePeakFlux()
    {
        float r_isco = ComputeISCO();
        const float dx = 0.001f;

        float max_f = 0.0f;
        float max_r = r_isco;
        for (float r = r_isco; r < DiskMaxRadius; r += dx)
        {
            float flux_r = PTFlux(r);
            if (flux_r > max_f)
            {
                max_r = r;
                max_f = flux_r;
            }
        }
        Debug.Log($"Peak flux: {max_f} at r={max_r}");
        return (max_f, max_r);
    }

    private float PTFlux(float r)
    {
        float rin = ComputeISCO();
        float x = Mathf.Sqrt(r);
        float x0 = Mathf.Sqrt(rin);
        float x1 = 2.0f * Mathf.Cos((Mathf.Acos(a) - Mathf.PI) / 3.0f);
        float x2 = 2.0f * Mathf.Cos((Mathf.Acos(a) + Mathf.PI) / 3.0f);
        float x3 = -2.0f * Mathf.Cos(Mathf.Acos(a) / 3.0f);

        float Q = 1.0f - x0 / x
                - (3.0f * a / (2.0f * x)) * Mathf.Log(x / x0)
                - 3.0f * (x1 - a) * (x1 - a) / (x * x1 * (x1 - x2) * (x1 - x3)) * Mathf.Log((x - x1) / (x0 - x1))
                - 3.0f * (x2 - a) * (x2 - a) / (x * x2 * (x2 - x1) * (x2 - x3)) * Mathf.Log((x - x2) / (x0 - x2))
                - 3.0f * (x3 - a) * (x3 - a) / (x * x3 * (x3 - x1) * (x3 - x2)) * Mathf.Log((x - x3) / (x0 - x3));

        return Q / (r * r * r);
    }

    float FluxPeak()
    {
        float rIsco = ComputeISCO();
        // evaluate the same Page-Thorne Q(x)/r^3 on the CPU,
        // scan r from rIsco*1.001 to discMax, take the max
        float peak = 0f;
        for (int i = 0; i < 4096; i++)
        {
            float r = Mathf.Exp(Mathf.Lerp(Mathf.Log(rIsco * 1.001f),
                                        Mathf.Log(DiskMaxRadius), i / 4095f));
            peak = Mathf.Max(peak, PTFlux(r) / (r * r * r));
        }
        return 1f / peak;
    }

    private void Execute()
    {
        CalculateViewDirection(renderCamera);

        shader.SetVector(ShaderIDs.BottomLeftDir_ID, _frustumCorners[0]);
        shader.SetVector(ShaderIDs.TopLeftDir_ID, _frustumCorners[1]);
        shader.SetVector(ShaderIDs.TopRightDir_ID, _frustumCorners[2]);
        shader.SetVector(ShaderIDs.BottomRightDir_ID, _frustumCorners[3]);

        shader.SetVector(ShaderIDs.CamPos_ID, renderCamera.transform.position);
        shader.SetVector(ShaderIDs.CamForward_ID, renderCamera.transform.forward);
        shader.SetInt(ShaderIDs.Frame_ID, Time.frameCount);
        shader.SetFloat(ShaderIDs.Time_ID, Time.time);

        int tx = Mathf.CeilToInt(width / 8.0f);
        int ty = Mathf.CeilToInt(height / 8.0f);
        shader.Dispatch(0, tx, ty, 1);
    }

    private void CreateRenderTexture()
    {
        if (target != null && (target.width != width || target.height != height))
        {
            target.Release();
            target = null;
        }

        if (target == null)
        {
            target = new RenderTexture(width, height, 0, GraphicsFormat.R32G32B32A32_SFloat);
            target.enableRandomWrite = true;
            target.Create();
        }
    }

    private void CalculateViewDirection(Camera cm)
    {
        // Calculate frustum corners in camera's local space for the far clip plane
        cm.CalculateFrustumCorners(new Rect(0, 0, 1, 1), cm.farClipPlane, Camera.MonoOrStereoscopicEye.Mono, _frustumCorners);

        // Transform the local-space corners to world space
        for (int i = 0; i < 4; i++)
        {
            _frustumCorners[i] = cameraTransform.TransformVector(_frustumCorners[i]);
        }
    }

    private readonly int topLeftID = Shader.PropertyToID("topLeftDir");
    private readonly int topRightID = Shader.PropertyToID("topRightDir");
    private readonly int bottomLeftID = Shader.PropertyToID("bottomLeftDir");
    private readonly int bottomRightID = Shader.PropertyToID("bottomRightDir");
    private readonly int camPosID = Shader.PropertyToID("_CamPos");

    private static class ShaderIDs
    {
        public static readonly int Result_ID = Shader.PropertyToID("_Result");
        public static readonly int SkyTex_ID = Shader.PropertyToID("_SkyTex");
        public static readonly int Width_ID = Shader.PropertyToID("_width");
        public static readonly int Height_ID = Shader.PropertyToID("_height");
        public static readonly int Frame_ID = Shader.PropertyToID("_Frame");
        public static readonly int Time_ID = Shader.PropertyToID("_Time");
        public static readonly int CamPos_ID = Shader.PropertyToID("_CamPos");
        public static readonly int CamForward_ID = Shader.PropertyToID("_CamForward");
        public static readonly int TopLeftDir_ID = Shader.PropertyToID("_topLeftDir");
        public static readonly int TopRightDir_ID = Shader.PropertyToID("_topRightDir");
        public static readonly int BottomLeftDir_ID = Shader.PropertyToID("_bottomLeftDir");
        public static readonly int BottomRightDir_ID = Shader.PropertyToID("_bottomRightDir");
        public static readonly int a_ID = Shader.PropertyToID("_a");
        public static readonly int m_ID = Shader.PropertyToID("_m");
        public static readonly int Q_ID = Shader.PropertyToID("_Q");
        public static readonly int r_isco_ID = Shader.PropertyToID("_r_isco");
        public static readonly int disk_max_radius_ID = Shader.PropertyToID("_disk_max_radius");
        public static readonly int peak_flux_ID = Shader.PropertyToID("_peak_flux");
        public static readonly int peak_flux_r_ID = Shader.PropertyToID("_peak_flux_r");
        public static readonly int Tpeak_ID = Shader.PropertyToID("_Tpeak");
        public static readonly int disk_str_azimuth_ID = Shader.PropertyToID("_disk_str_azimuth");
        public static readonly int disk_str_radial_ID = Shader.PropertyToID("_disk_str_radial");
        public static readonly int shift_ID = Shader.PropertyToID("_shift");
        public static readonly int rot_speed_ID = Shader.PropertyToID("_rot_speed");
        public static readonly int gain_ID = Shader.PropertyToID("_gain");
        public static readonly int turbulenceStrength_ID = Shader.PropertyToID("_turbulenceStrength");
        public static readonly int BottomLeft_ID = Shader.PropertyToID("_bottomLeft");
    }

    [CustomEditor(typeof(KerrTracer))]
    public class KerrTracerEditor : Editor
    {
        public enum TextureType { Final }

        public override void OnInspectorGUI()
        {
            KerrTracer settings = (KerrTracer)target;

            if (GUILayout.Button("Init"))
            {
                settings.SetupCompute();
            }

            if (GUILayout.Button("Update"))
            {
                settings.Execute();
            }

            EditorGUILayout.Space();
            if (GUILayout.Button("Show Texture Viewer"))
            {
                var window = EditorWindow.GetWindow<TextureViewer>();
                window.titleContent = new GUIContent("Texture Viewer");
                window.tracer = settings;
                window.maxSize = new Vector2(400, 400);
                window.Show();
            }

            EditorGUILayout.Space();
            DrawDefaultInspector();
        }
    }
    public class TextureViewer : EditorWindow
    {
        private RenderTexture m_temp;
        private KerrTracerEditor.TextureType textureType;
        public KerrTracer tracer;

        void OnGUI()
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Texture Type", GUILayout.Width(120));
            textureType = (KerrTracerEditor.TextureType)EditorGUILayout.EnumPopup(textureType);
            EditorGUILayout.EndHorizontal();

            m_temp = tracer.target;

            if (m_temp)
            {
                EditorGUI.DrawPreviewTexture(
                    new Rect(new Vector2(10, 50), Vector2.one * 300),
                    m_temp);
            }
            else
            {
                EditorGUILayout.HelpBox("Texture is not generated!", MessageType.Warning);
            }
        }
    }
}
