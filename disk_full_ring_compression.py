import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from collections import defaultdict

# ============================================================
# 2D circular disk FEM - plane stress CST elements
#
# Model:
#   - circular disk
#   - center region fixed: r <= 0.12 R
#   - inward normal traction on the full outer circumference
#   - constant-strain triangular (CST) elements
# ============================================================

# -----------------------------
# Model parameters
# -----------------------------
R = 0.15                 # disk radius [m]
thickness = 0.005        # thickness [m]
E = 70.0e9               # Young's modulus [Pa]
nu = 0.33                # Poisson ratio [-]
pressure = 1.0e6         # inward boundary traction [Pa]

FIXED_RADIUS = 0.12 * R

# Mesh controls
N_RADIAL = 14
N_THETA = 72

# -----------------------------
# 1. Generate disk mesh
# -----------------------------
points = [[0.0, 0.0]]

for ir in range(1, N_RADIAL + 1):
    r = R * ir / N_RADIAL
    ntheta = max(12, int(np.ceil(N_THETA * r / R)))

    for it in range(ntheta):
        theta = 2.0 * np.pi * it / ntheta
        points.append([r * np.cos(theta), r * np.sin(theta)])

points = np.asarray(points, dtype=float)

tri = Delaunay(points)
elements = tri.simplices.copy()

centroids = points[elements].mean(axis=1)
inside = np.linalg.norm(centroids, axis=1) <= R * (1.0 + 1.0e-10)
elements = elements[inside]

nn = len(points)
ne = len(elements)
ndof = 2 * nn

# -----------------------------
# 2. Plane-stress material matrix
# -----------------------------
D = E / (1.0 - nu**2) * np.array([
    [1.0, nu, 0.0],
    [nu, 1.0, 0.0],
    [0.0, 0.0, (1.0 - nu) / 2.0]
])

# -----------------------------
# 3. CST element routine
# -----------------------------
def cst_B_matrix(xy):
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]

    det2A = (
        x1 * (y2 - y3)
        + x2 * (y3 - y1)
        + x3 * (y1 - y2)
    )

    area = 0.5 * abs(det2A)

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2

    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    B = (1.0 / det2A) * np.array([
        [b1, 0.0, b2, 0.0, b3, 0.0],
        [0.0, c1, 0.0, c2, 0.0, c3],
        [c1, b1, c2, b2, c3, b3]
    ])

    return B, area

# -----------------------------
# 4. Assemble global stiffness
# -----------------------------
K = np.zeros((ndof, ndof))
F = np.zeros(ndof)

for elem in elements:
    xy = points[elem]
    B, area = cst_B_matrix(xy)
    Ke = thickness * area * (B.T @ D @ B)

    dofs = np.array([
        2 * elem[0], 2 * elem[0] + 1,
        2 * elem[1], 2 * elem[1] + 1,
        2 * elem[2], 2 * elem[2] + 1
    ])

    K[np.ix_(dofs, dofs)] += Ke

# -----------------------------
# 5. Find outer boundary edges
# -----------------------------
edge_count = defaultdict(int)

for elem in elements:
    edges = [
        tuple(sorted((elem[0], elem[1]))),
        tuple(sorted((elem[1], elem[2]))),
        tuple(sorted((elem[2], elem[0])))
    ]

    for edge in edges:
        edge_count[edge] += 1

boundary_edges = [
    edge for edge, count in edge_count.items()
    if count == 1
]

outer_edges = []
for n1, n2 in boundary_edges:
    r1 = np.linalg.norm(points[n1])
    r2 = np.linalg.norm(points[n2])

    if r1 > 0.90 * R and r2 > 0.90 * R:
        outer_edges.append((n1, n2))

# -----------------------------
# 6. Apply full-ring inward traction
# -----------------------------
for n1, n2 in outer_edges:
    x1 = points[n1]
    x2 = points[n2]

    edge_length = np.linalg.norm(x2 - x1)
    midpoint = 0.5 * (x1 + x2)
    outward_normal = midpoint / np.linalg.norm(midpoint)

    traction = -pressure * outward_normal
    nodal_force = traction * thickness * edge_length / 2.0

    F[2 * n1:2 * n1 + 2] += nodal_force
    F[2 * n2:2 * n2 + 2] += nodal_force

# -----------------------------
# 7. Fix center region
# -----------------------------
r_nodes = np.linalg.norm(points, axis=1)
fixed_nodes = np.where(r_nodes <= FIXED_RADIUS)[0]

fixed_dofs = np.sort(np.ravel(
    np.column_stack((2 * fixed_nodes, 2 * fixed_nodes + 1))
))

all_dofs = np.arange(ndof)
free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

# -----------------------------
# 8. Solve K u = F
# -----------------------------
U = np.zeros(ndof)
Kff = K[np.ix_(free_dofs, free_dofs)]
Ff = F[free_dofs]
U[free_dofs] = np.linalg.solve(Kff, Ff)

ux = U[0::2]
uy = U[1::2]
umag = np.sqrt(ux**2 + uy**2)

radial_unit = np.zeros_like(points)
nonzero = r_nodes > 1.0e-14
radial_unit[nonzero] = points[nonzero] / r_nodes[nonzero, None]
ur = ux * radial_unit[:, 0] + uy * radial_unit[:, 1]

# -----------------------------
# 9. Element stress / von Mises
# -----------------------------
vm = np.zeros(ne)

for e, elem in enumerate(elements):
    xy = points[elem]
    B, area = cst_B_matrix(xy)

    dofs = np.array([
        2 * elem[0], 2 * elem[0] + 1,
        2 * elem[1], 2 * elem[1] + 1,
        2 * elem[2], 2 * elem[2] + 1
    ])

    strain = B @ U[dofs]
    stress = D @ strain
    sx, sy, txy = stress

    vm[e] = np.sqrt(
        sx**2 - sx * sy + sy**2 + 3.0 * txy**2
    )

# -----------------------------
# 10. Results
# -----------------------------
print("Number of nodes      =", nn)
print("Number of elements   =", ne)
print("Fixed center radius  =", FIXED_RADIUS, "m")
print("Max displacement     =", np.max(umag), "m")
print("Max von Mises stress =", np.max(vm), "Pa")

# -----------------------------
# 11. Mesh
# -----------------------------
plt.figure(figsize=(7, 7))
plt.triplot(points[:, 0], points[:, 1], elements, lw=0.45)
plt.scatter(points[fixed_nodes, 0], points[fixed_nodes, 1], s=12, label="Fixed center")
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Circular Disk FEM Mesh")
plt.legend()
plt.tight_layout()
plt.savefig("disk_full_ring_mesh.png", dpi=200)
plt.show()

# -----------------------------
# 12. Radial displacement
# -----------------------------
plt.figure(figsize=(7, 6))
plt.tricontourf(points[:, 0], points[:, 1], elements, ur, levels=30)
plt.colorbar(label="Radial displacement [m]")
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Radial Displacement")
plt.tight_layout()
plt.savefig("disk_full_ring_radial_disp.png", dpi=200)
plt.show()

# -----------------------------
# 13. Displacement magnitude
# -----------------------------
plt.figure(figsize=(7, 6))
plt.tricontourf(points[:, 0], points[:, 1], elements, umag, levels=30)
plt.colorbar(label="|u| [m]")
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Displacement Magnitude")
plt.tight_layout()
plt.savefig("disk_full_ring_umag.png", dpi=200)
plt.show()

# -----------------------------
# 14. von Mises stress
# -----------------------------
plt.figure(figsize=(7, 6))
plt.tripcolor(points[:, 0], points[:, 1], elements, facecolors=vm, shading="flat")
plt.colorbar(label="von Mises stress [Pa]")
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("von Mises Stress")
plt.tight_layout()
plt.savefig("disk_full_ring_vm.png", dpi=200)
plt.show()

# -----------------------------
# 15. Deformed shape
# -----------------------------
max_u = np.max(umag)
scale = 0.10 * R / max_u if max_u > 0.0 else 1.0

deformed = points + scale * np.column_stack((ux, uy))

plt.figure(figsize=(7, 7))
plt.triplot(points[:, 0], points[:, 1], elements, lw=0.35, label="Original")
plt.triplot(deformed[:, 0], deformed[:, 1], elements, lw=0.6, label="Deformed")
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"Deformed Shape (scale = {scale:.1f})")
plt.legend()
plt.tight_layout()
plt.savefig("disk_full_ring_deformed.png", dpi=200)
plt.show()
