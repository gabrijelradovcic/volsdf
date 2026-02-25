import trimesh
import numpy as np
m1 = trimesh.load('data/Changwei8/scan1/object_mesh.ply')
m2 = trimesh.load('exps/Changwei8_1/2026_02_20_13_12_42/plots/surface_550.ply')
print("Mesh 1 bbox min:", m1.vertices.min(axis=0))
print("Mesh 1 bbox max:", m1.vertices.max(axis=0))
print("Mesh 1 center:", m1.vertices.mean(axis=0))
print("Mesh 2 bbox min:", m2.vertices.min(axis=0))
print("Mesh 2 bbox max:", m2.vertices.max(axis=0))
print("Mesh 2 center:", m2.vertices.mean(axis=0))
