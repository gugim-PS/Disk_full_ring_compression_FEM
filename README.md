# Circular Disk Compression by FEM

Solution with Python.

This example analyzes a two-dimensional circular disk using the Finite Element Method (FEM) with plane-stress Constant Strain Triangle (CST) elements.

Equation

$$ \mathbf{K}\mathbf{u}=\mathbf{F} $$

The CST element stiffness matrix is:

$$ \mathbf{K}_e=tA\mathbf{B}^{T}\mathbf{D}\mathbf{B} $$

The center region is fixed:

$$ r\le0.12R $$

A uniform inward normal traction is applied over the full outer circumference:

$$ \mathbf{t}=-p\mathbf{n} $$

The von Mises stress is calculated by:

$$ \sigma_{vm}=\sqrt{\sigma_x^2-\sigma_x\sigma_y+\sigma_y^2+3\tau_{xy}^2} $$

## Model

2D circular disk

Plane-stress FEM

CST triangular elements

Center fixed for r <= 0.12 R

Inward pressure applied around the full circumference

Two displacement DOFs per node: u_x, u_y

# Python

The complete code is:

disk_full_ring_compression.py

Install the required packages:

pip install numpy matplotlib scipy

Run:

python disk_full_ring_compression.py

## Output

The program calculates and plots:

FEM mesh

radial displacement

displacement magnitude

von Mises stress

deformed shape

The generated image files are:

disk_full_ring_mesh.png
disk_full_ring_radial_disp.png
disk_full_ring_umag.png
disk_full_ring_vm.png
disk_full_ring_deformed.png

## FEM Procedure

Generate the triangular disk mesh.

Construct the plane-stress material matrix.

Assemble the CST element stiffness matrices.

Apply inward traction to the outer circumference.

Fix the center region.

Solve K u = F.

Calculate displacement and von Mises stress.

Plot the FEM results.

<div align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
</div>
