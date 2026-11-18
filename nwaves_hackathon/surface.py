"""The table top: what the work surface looks like.

The real workspace has a light wooden table top. Here it is a picture laid on the enclosure's floor, 0.2 mm above
it: one flat sheet over the whole floor, with a wood image (assets/wood_light.png). Nothing rests on the sheet and
nothing collides with it; the cube and the tray stand on the enclosure's floor right under it.

A reset gives the sheet its look for the episode (see mdp.reset_scene and layout.py): the wood as it is, the wood
with another brightness and tint, or one plain colour without grain. In an episode that shows the enclosure's bare
floor or the mat instead, the sheet is kept out of sight below the floor. The look is two numbers per colour
channel on the image: ``scale`` (multiplies the wood) and ``bias`` (a plain colour when the scale is 0).

    python -m nwaves_hackathon.surface      # writes assets/wood_light.png again (needs numpy and Pillow)

The USD file is written on first use and again whenever a number in layout.py changes.
"""
import hashlib
import os
import tempfile

from . import layout as L

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
WOOD_IMAGE = os.path.join(ASSETS, "wood_light.png")
IMAGE_METRES = 0.8            # the image covers this much table, along and across the grain; it repeats beyond


def wood_image(size: int = 1024, seed: int = 7):
    """A light wood with straight grain, as an (size, size, 3) uint8 sRGB image. The grain runs along the image's
    vertical axis, which is laid along the arm's forward direction."""
    import numpy as np
    rng = np.random.default_rng(seed)

    def smooth(n_points, length):
        """Random values at n_points, joined smoothly over ``length`` samples (repeats at the ends)."""
        knots = rng.standard_normal(n_points)
        x = np.linspace(0.0, n_points, length, endpoint=False)
        i = np.floor(x).astype(int)
        f = x - i
        f = f * f * (3.0 - 2.0 * f)
        return knots[i % n_points] * (1.0 - f) + knots[(i + 1) % n_points] * f

    across = np.arange(size)[None, :] / size                  # 0..1 across the grain
    # the grain lines are not quite straight: each row is shifted sideways a little, slowly along the grain
    wave = 0.012 * smooth(5, size)[:, None] + 0.004 * smooth(17, size)[:, None]
    pos = across + wave + 0.006 * smooth(7, size)[None, :] * smooth(9, size)[:, None]
    # year rings cut lengthwise: broad darker bands plus narrow lines
    bands = sum(a * np.sin(2.0 * np.pi * (f * pos + rng.random())) for f, a in ((7, 0.5), (13, 0.3), (29, 0.25), (61, 0.15)))
    lines = np.interp((pos % 1.0) * 400.0, np.arange(401), np.r_[rng.standard_normal(400), 0.0])
    # fibres: fine streaks, long along the grain
    fibre = rng.standard_normal((size // 16, size))
    fibre = np.repeat(fibre, 16, axis=0)[:size]
    fibre = (fibre + np.roll(fibre, 8, axis=0)) / 2.0
    blotch = smooth(3, size)[:, None] * 0.5 + smooth(4, size)[None, :] * 0.5
    shade = 1.0 - 0.055 * bands - 0.035 * lines - 0.02 * fibre + 0.03 * blotch
    base = np.array([206.0, 172.0, 122.0])                    # sRGB, the table top of the photo
    dark = np.array([0.0, 8.0, 18.0])                         # darker grain is also a little redder
    image = base[None, None, :] * shade[:, :, None] - dark[None, None, :] * np.clip(1.0 - shade, 0.0, 1.0)[:, :, None] * 4.0
    return np.clip(image, 0, 255).astype(np.uint8)


def write_wood_image(path: str = WOOD_IMAGE):
    from PIL import Image
    Image.fromarray(wood_image()).save(path, optimize=True)
    return path


def surface_usda() -> str:
    (x0, x1), (y0, y1) = L.SURFACE_SHEET
    u0, u1 = y0 / IMAGE_METRES, y1 / IMAGE_METRES             # the image's width lies across the arm (along y)
    v0, v1 = x0 / IMAGE_METRES, x1 / IMAGE_METRES             # its height, the grain, along the arm's forward x
    image = WOOD_IMAGE.replace("\\", "/")
    return f'''#usda 1.0
(
    defaultPrim = "Surface"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Surface" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    bool physics:kinematicEnabled = 1
    bool physics:rigidBodyEnabled = 1
    float physics:mass = 0.1

    def Scope "Looks"
    {{
        def Material "TableTop"
        {{
            token outputs:surface.connect = </Surface/Looks/TableTop/Shader.outputs:surface>

            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Surface/Looks/TableTop/Image.outputs:rgb>
                float inputs:metallic = 0
                float inputs:roughness = 0.55
                token outputs:surface
            }}

            def Shader "Image"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{image}@
                token inputs:sourceColorSpace = "sRGB"
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float4 inputs:scale = (1, 1, 1, 1)
                float4 inputs:bias = (0, 0, 0, 0)
                float2 inputs:st.connect = </Surface/Looks/TableTop/ST.outputs:result>
                color3f outputs:rgb
            }}

            def Shader "ST"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }}
        }}
    }}

    def Xform "geometry"
    {{
        def Mesh "Sheet" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            uniform bool doubleSided = 0
            float3[] extent = [({x0:.4f}, {y0:.4f}, 0), ({x1:.4f}, {y1:.4f}, 0)]
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            rel material:binding = </Surface/Looks/TableTop>
            normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
                interpolation = "vertex"
            )
            point3f[] points = [({x0:.4f}, {y0:.4f}, 0), ({x1:.4f}, {y0:.4f}, 0), ({x1:.4f}, {y1:.4f}, 0), ({x0:.4f}, {y1:.4f}, 0)]
            texCoord2f[] primvars:st = [({u0:.5f}, {v0:.5f}), ({u0:.5f}, {v1:.5f}), ({u1:.5f}, {v1:.5f}), ({u1:.5f}, {v0:.5f})] (
                interpolation = "vertex"
            )
            uniform token subdivisionScheme = "none"
        }}
    }}
}}
'''


def surface_usd_path() -> str:
    """Path of the table top's USD file (written if it is not there yet)."""
    text = surface_usda()
    name = f"nwaves_surface_{hashlib.sha1(text.encode()).hexdigest()[:12]}.usda"
    folder = os.path.join(os.environ.get("NWAVES_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "nwaves_hackathon"))
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        os.makedirs(folder, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    return path


if __name__ == "__main__":
    print(write_wood_image())
