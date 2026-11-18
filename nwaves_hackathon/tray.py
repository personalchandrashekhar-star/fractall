"""The tray, built out of the numbers in layout.py.

A low open box with rounded corners: 75.8 x 71.7 mm outside and 28.9 mm high. Its floor is 14.4 mm thick and its
walls rise another 14.4 mm, so the pocket where the cube goes is 72.0 x 68.1 mm and 14.4 mm deep (the size of the
real printed tray). It is one fixed (kinematic) body: the cube and the gripper collide with it, nothing moves it.
Its origin is the centre of its underside, so it stands on the work surface at that surface's height.

What you see is one mesh with the exact outline. What the cube and the gripper touch is made of simple shapes
with the same surfaces: one block for the floor (its upper face is the pocket's floor) and a row of thin blocks
for the walls (one straight piece along each side, short ones around the corners).

The USD file is written on first use and again whenever a number in layout.py changes, so editing ``TRAY_SIZE``,
``TRAY_HEIGHT``, ``TRAY_FLOOR``, ``TRAY_WALL``, ``TRAY_CORNER_RADIUS`` or ``TRAY_COLOR`` is all it takes.
"""
import hashlib
import math
import os
import tempfile

from . import layout as L

CORNER_STEPS = 8          # straight pieces per rounded corner in the picture
CORNER_BLOCKS = 4         # wall blocks per rounded corner in the collision shape
FLOOR_CORNER_STEPS = 6    # corner pieces of the floor block's collision shape (a convex shape of at most 64 points)


def outline(half, radius, steps: int):
    """Points of a rectangle with rounded corners, counter-clockwise seen from above, centred on the origin.
    ``half`` = (half length along x, half length along y), ``radius`` = (corner radius along x, along y)."""
    rx, ry = max(min(radius[0], half[0]), 0.0), max(min(radius[1], half[1]), 0.0)
    cx, cy = half[0] - rx, half[1] - ry
    points = []
    for corner, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        for k in range(steps + 1):
            ang = math.pi / 2 * (corner + k / steps)
            points.append((sx * cx + rx * math.cos(ang), sy * cy + ry * math.sin(ang)))
    return points


def _outer(steps: int):
    return outline((L.TRAY_SIZE[0] / 2, L.TRAY_SIZE[1] / 2), L.TRAY_CORNER_RADIUS, steps)


def _inner(steps: int):
    return outline((L.TRAY_POCKET[0] / 2, L.TRAY_POCKET[1] / 2),
                   (L.TRAY_CORNER_RADIUS[0] - L.TRAY_WALL[0], L.TRAY_CORNER_RADIUS[1] - L.TRAY_WALL[1]), steps)


def body_mesh():
    """The tray as one closed mesh: (points, face vertex counts, face vertex indices)."""
    outer, inner = _outer(CORNER_STEPS), _inner(CORNER_STEPS)
    n = len(outer)
    rings = [[(x, y, 0.0) for x, y in outer],                    # 0: underside
             [(x, y, L.TRAY_HEIGHT) for x, y in outer],          # 1: outer edge of the walls' top
             [(x, y, L.TRAY_HEIGHT) for x, y in inner],          # 2: inner edge of the walls' top
             [(x, y, L.TRAY_FLOOR) for x, y in inner]]           # 3: the pocket's floor
    points = [p for ring in rings for p in ring]
    counts, indices = [], []

    def face(ids):
        counts.append(len(ids))
        indices.extend(ids)

    face(list(range(n - 1, -1, -1)))                             # underside, seen from below
    for a, b in ((0, 1), (1, 2), (2, 3)):                        # outside of the walls, their top, their inside
        for i in range(n):
            j = (i + 1) % n
            face([a * n + i, a * n + j, b * n + j, b * n + i])
    face([3 * n + i for i in range(n)])                          # the pocket's floor, seen from above
    return points, counts, indices


def wall_blocks():
    """The walls as blocks: (centre x, centre y, turn about the vertical [deg], length, thickness)."""
    hx, hy = L.TRAY_SIZE[0] / 2, L.TRAY_SIZE[1] / 2
    wx, wy = L.TRAY_WALL
    rx, ry = L.TRAY_CORNER_RADIUS
    blocks = [(hx - wx / 2, 0.0, 90.0, 2 * (hy - ry), wx), (-(hx - wx / 2), 0.0, 90.0, 2 * (hy - ry), wx),
              (0.0, hy - wy / 2, 0.0, 2 * (hx - rx), wy), (0.0, -(hy - wy / 2), 0.0, 2 * (hx - rx), wy)]
    # around the corners: short blocks along the middle line between the outer and the inner outline
    outer, inner = _outer(CORNER_BLOCKS), _inner(CORNER_BLOCKS)
    per = CORNER_BLOCKS + 1
    for corner in range(4):
        for k in range(CORNER_BLOCKS):
            i = corner * per + k
            a = ((outer[i][0] + inner[i][0]) / 2, (outer[i][1] + inner[i][1]) / 2)
            b = ((outer[i + 1][0] + inner[i + 1][0]) / 2, (outer[i + 1][1] + inner[i + 1][1]) / 2)
            thick = (math.dist(outer[i], inner[i]) + math.dist(outer[i + 1], inner[i + 1])) / 2
            blocks.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])),
                           math.dist(a, b) + thick, thick))            # a little longer: the blocks overlap, no gaps
    return blocks


def _floats(values):
    return ", ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y, z in values)


def tray_usda() -> str:
    hx, hy = L.TRAY_SIZE[0] / 2, L.TRAY_SIZE[1] / 2
    color = ", ".join(f"{c:.4f}" for c in L.TRAY_COLOR)
    points, counts, indices = body_mesh()
    floor_xy = _outer(FLOOR_CORNER_STEPS)
    floor = [(x, y, 0.0) for x, y in floor_xy] + [(x, y, L.TRAY_FLOOR) for x, y in floor_xy]
    m = len(floor_xy)
    floor_counts = [m, m] + [4] * m
    floor_indices = list(range(m - 1, -1, -1)) + list(range(m, 2 * m))
    for i in range(m):
        j = (i + 1) % m
        floor_indices += [i, j, m + j, m + i]
    out = [f'''#usda 1.0
(
    defaultPrim = "Tray"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Tray" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI"]
)
{{
    bool physics:kinematicEnabled = 1
    bool physics:rigidBodyEnabled = 1

    def Scope "Looks"
    {{
        def Material "TrayMaterial"
        {{
            token outputs:surface.connect = </Tray/Looks/TrayMaterial/Shader.outputs:surface>

            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({color})
                float inputs:metallic = 0
                float inputs:roughness = 0.5
                token outputs:surface
            }}
        }}
    }}

    def Xform "geometry"
    {{
        def Mesh "Body" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            uniform bool doubleSided = 1
            float3[] extent = [({-hx:.6f}, {-hy:.6f}, 0), ({hx:.6f}, {hy:.6f}, {L.TRAY_HEIGHT:.6f})]
            int[] faceVertexCounts = {counts}
            int[] faceVertexIndices = {indices}
            rel material:binding = </Tray/Looks/TrayMaterial>
            uniform token orientation = "rightHanded"
            point3f[] points = [{_floats(points)}]
            uniform token subdivisionScheme = "none"
        }}
    }}

    def Xform "collision"
    {{
        def Mesh "Floor" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {{
            float3[] extent = [({-hx:.6f}, {-hy:.6f}, 0), ({hx:.6f}, {hy:.6f}, {L.TRAY_FLOOR:.6f})]
            int[] faceVertexCounts = {floor_counts}
            int[] faceVertexIndices = {floor_indices}
            uniform token orientation = "rightHanded"
            uniform token physics:approximation = "convexHull"
            point3f[] points = [{_floats(floor)}]
            uniform token purpose = "guide"
            uniform token subdivisionScheme = "none"
        }}
''']
    for i, (x, y, turn, length, thick) in enumerate(wall_blocks()):
        out.append(f'''
        def Cube "Wall_{i:02d}" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            double size = 1
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            uniform token purpose = "guide"
            double3 xformOp:translate = ({x:.6f}, {y:.6f}, {L.TRAY_HEIGHT / 2:.6f})
            double xformOp:rotateZ = {turn:.4f}
            double3 xformOp:scale = ({length:.6f}, {thick:.6f}, {L.TRAY_HEIGHT:.6f})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]
        }}
''')
    out.append("    }\n}\n")
    return "".join(out)


def tray_usd_path() -> str:
    """Path of the tray's USD file for the current numbers in layout.py (written if it is not there yet)."""
    text = tray_usda()
    name = f"nwaves_tray_{hashlib.sha1(text.encode()).hexdigest()[:12]}.usda"
    folder = os.path.join(os.environ.get("NWAVES_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "nwaves_hackathon"))
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        os.makedirs(folder, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    return path


def measurements() -> dict:
    """Areas and volume of the built tray, to compare with the print file (mm2, mm3)."""
    def polygon_area(ring):
        return abs(sum(ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1]
                       for i in range(len(ring)))) / 2

    outer, pocket = polygon_area(_outer(CORNER_STEPS)), polygon_area(_inner(CORNER_STEPS))
    return {"outline_mm2": outer * 1e6, "pocket_mm2": pocket * 1e6,
            "volume_mm3": (outer * L.TRAY_FLOOR + (outer - pocket) * L.TRAY_WALL_HEIGHT) * 1e9,
            "wall_blocks": len(wall_blocks())}
