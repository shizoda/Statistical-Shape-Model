#!/usr/bin/env python
# Reconstruct shapes at certain standard deviations from the mean shape model

import vtk
import os
import tqdm
import numpy as np
import argparse

def read_mean_shape(which_shape):
    """Read the mean shape and return polydata with points and scalars."""
    reader = vtk.vtkPolyDataReader()
    mean_shape_file = '{}_ED_mean.vtk'.format(which_shape)
    reader.SetFileName(mean_shape_file)
    reader.Update()
    return reader.GetOutput()

def apply_shape_changes(which_shape, which_mode, pc, variance, n_points, out_dir, num_steps=14):
    """Apply shape changes based on the given mode and standard deviations."""
    how_much_stds = np.linspace(-2, 2, num=num_steps)  # Example: change from 0.1 to 2.0 in 20 steps
    for step, how_much_std in enumerate(tqdm.tqdm(how_much_stds, desc="Applying shape changes")):
        polydata = read_mean_shape(which_shape)  # Read mean shape for each step
        points = polydata.GetPoints()
        scalars = polydata.GetPointData().GetScalars()
        
        # Apply shape changes
        pc_i = pc[:, which_mode]
        pc_i = np.reshape(pc_i, (n_points, 3))
        std = np.sqrt(variance[which_mode])

        for j in range(n_points):
            p = np.array(points.GetPoint(j))  # Use numpy array for vectorized operation
            p += how_much_std * std * pc_i[j]
            points.SetPoint(j, *p)

        # Update the points and scalars in polydata and output as VTK files
        output_vtk_file(polydata, which_shape, which_mode, step, out_dir, num_steps)


def output_vtk_file(polydata, which_shape, which_mode, step, out_dir, total_steps):
    """Output the polydata as a VTK file and update the corresponding .pvd file."""
    mode_dir = os.path.join(out_dir, which_shape, "mode" + str(which_mode))
    os.makedirs(mode_dir, exist_ok=True)
    file_path = os.path.join(mode_dir, f"{which_shape}_mode{which_mode}_{step+1:03}.vtk")
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.Write()


def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Reconstruct shapes from PCA modes with varying standard deviations.')
    parser.add_argument('-s', '--which_shapes', type=str, nargs='+', default=["LV", "RV", "Both"], choices=["LV", "RV", "Both"], help='Specify the anatomical structure.')
    parser.add_argument('-m', '--which_modes', type=int, nargs='+', default=[0, 1, 2], help='Specify one or more modes (e.g., 0 1 2 for the 1st, 2nd, and 3rd PCs).')
    parser.add_argument('-o', '--output_dir', type=str, default="./output", help='Specify the output directory for VTK files.')
    args = parser.parse_args()


    for which_shape in args.which_shapes:
    
        polydata = read_mean_shape(which_shape)
        n_points = polydata.GetPoints().GetNumberOfPoints()

        # Read the principal component and variance
        pc = np.genfromtxt('{}_ED_pc_100_modes.csv.gz'.format(which_shape), delimiter=',')
        variance = np.genfromtxt('{}_ED_var_100_modes.csv.gz'.format(which_shape), delimiter=',')

        for which_mode in args.which_modes:
            apply_shape_changes(which_shape, which_mode, pc, variance, n_points, args.output_dir)

if __name__ == "__main__":
    main()
