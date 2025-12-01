import re
import os
import glob
from typing import List, Optional, Literal, Union

import vtk
import tqdm
import trimesh
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev

from pc.curve import Curve


class CurveSurfaceMapper:
    def __init__(self, stl_file: str):
        """
        Initialize the mapper with an STL file.

        Parameters:
        stl_file (str): Path to the STL file to load the 3D object.
        """
        self.mesh = trimesh.load(stl_file)
        self.surface_points = np.array(self.mesh.vertices)
        self.kdtree = KDTree(self.surface_points)

    def enhance_curve_resolution(self, curve: np.ndarray, num_points: int = 100) -> np.ndarray:
        """
        Increase the resolution of a curve by spline interpolation.

        Parameters:
        curve (np.ndarray): The input curve as a NumPy array of shape (n, 3).
        num_points (int): The desired number of points in the enhanced curve.

        Returns:
        np.ndarray: The enhanced curve as a NumPy array of shape (num_points, 3).
        """
        tck, _ = splprep(curve.T, s=0)  # Fit a spline
        new_points = np.linspace(0, 1, num_points)  # Uniform parameterization
        enhanced_curve = np.array(splev(new_points, tck)).T
        return enhanced_curve

    def resample_curves(self, curves: List[np.ndarray], num_points: int = 100) -> List[np.ndarray]:
        """
        Resample all curves in the list to the same resolution.

        Parameters:
        curves (List[np.ndarray]): A list of input curves, each as a NumPy array of shape (n, 3).
        num_points (int): The desired number of points in each resampled curve.

        Returns:
        List[np.ndarray]: A list of resampled curves.
        """
        return [self.enhance_curve_resolution(curve, num_points) for curve in curves]

    def smooth_curve(self, curve: np.ndarray, method: Literal["savitzky", "spline"] = "savitzky", **kwargs) -> np.ndarray:
        def savitzky(curve: np.ndarray, window_length: int, polyorder: int):
            from scipy.signal import savgol_filter
            x = curve[:, 0]
            x_smooth = savgol_filter(x, window_length=window_length, polyorder=polyorder)
            
            y = curve[:, 1]
            y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
            
            z = curve[:, 2]
            z_smooth = savgol_filter(z, window_length=window_length, polyorder=polyorder)
            return np.stack([x_smooth, y_smooth, z_smooth], axis=1)
        
        def spline(curve: np.ndarray, s: float):
            tck, u = splprep(curve.T, s=s)
            return np.array(splev(u, tck)).T
        
        if method == "savitzky":
            return savitzky(curve, kwargs.get("window_length", 11), kwargs.get("polyorder", 5))
        
        elif method == "spline":
            return spline(curve, kwargs.get("s", 1.0))
        
        else:
            raise ValueError(f"method {method} is not valid. (savitzky, spline)")
    
    def project_to_surface_nearest(self, curve: np.ndarray) -> np.ndarray:
        """
        Map a curve to the surface using the nearest-point projection.

        Parameters:
        curve (np.ndarray): The input curve as a NumPy array of shape (n, 3).

        Returns:
        np.ndarray: The mapped curve on the surface as a NumPy array of shape (n, 3).
        """
        _, indices = self.kdtree.query(curve)
        return self.surface_points[indices]

    def project_to_surface_ray_casting(self, curve: np.ndarray, ray_directions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Map a curve to the surface using ray-casting.

        Parameters:
        curve (np.ndarray): The input curve as a NumPy array of shape (n, 3).
        ray_directions (Optional[np.ndarray]): The directions of the rays as a NumPy array of shape (n, 3). 
                                                If None, defaults to uniform downward direction.

        Returns:
        np.ndarray: The intersection points on the surface as a NumPy array of shape (m, 3), where m <= n.
        """
        if ray_directions is None:
            ray_directions = -np.ones_like(curve)  # Example: uniform direction
        intersections, _, _ = self.mesh.ray.intersects_location(
            ray_origins=curve, ray_directions=ray_directions
        )
        return intersections

    def project_to_surface_sdf(self, curve: np.ndarray, max_iterations: int = 100, tolerance: float = 1e-6) -> np.ndarray:
        """
        Map a curve to the surface using signed distance field (SDF) alignment.

        Parameters:
        curve (np.ndarray): The input curve as a NumPy array of shape (n, 3).
        max_iterations (int): The maximum number of iterations for the alignment process.
        tolerance (float): The tolerance for stopping the alignment process.

        Returns:
        np.ndarray: The mapped curve on the surface as a NumPy array of shape (n, 3).
        """
        curve_mapped = curve.copy()
        for _ in range(max_iterations):
            # Unpack the three returned values
            closest_points, distances, _ = self.mesh.nearest.on_surface(curve_mapped)

            # Stop if all points are within the tolerance
            if np.max(distances) < tolerance:
                break

            # Calculate normals and move points towards the surface
            normals = (curve_mapped - closest_points) / distances[:, None]
            curve_mapped -= normals * distances[:, None]
        return curve_mapped

    def map_curves(self, curves: List[np.ndarray], method: Literal["nearest", "ray_casting", "sdf"] = "nearest", **kwargs) -> List[np.ndarray]:
        """
        Map all curves to the surface using the specified method.

        Parameters:
        curves (List[np.ndarray]): A list of input curves, each as a NumPy array of shape (n, 3).
        method (str): The method to use for mapping. Options are "nearest", "ray_casting", and "sdf".
        **kwargs: Additional arguments for the selected mapping method.

        Returns:
        List[np.ndarray]: A list of mapped curves on the surface.

        Raises:
        ValueError: If an unknown method is specified.
        """
        mapped_curves = []
        for curve in curves:
            if method == "nearest":
                mapped_curve = self.project_to_surface_nearest(curve)
            elif method == "ray_casting":
                mapped_curve = self.project_to_surface_ray_casting(curve, **kwargs)
            elif method == "sdf":
                mapped_curve = self.project_to_surface_sdf(curve, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            mapped_curves.append(mapped_curve)
        return mapped_curves

def export_multiple_curves_to_vtk(curves: Union[List[np.ndarray], np.ndarray], output_filename: str):
    if not isinstance(curves, list):
        curves = [curves]
    
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    point_offset = 0

    for curve in curves:
        num_points = curve.shape[0]

        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(num_points)

        for i, point in enumerate(curve):
            points.InsertNextPoint(point)
            polyline.GetPointIds().SetId(i, point_offset + i)

        cells.InsertNextCell(polyline)

        point_offset += num_points

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(cells)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(polydata)
    writer.Write()

def material_figure_ten_segment(results_dir: str, stl_file: str, *, method: Literal["nearest", "ray_casting", "sdf"] = "nearest"):
    mapper = CurveSurfaceMapper(stl_file)
    list_dirs = []

    beta = None
    dir_frechet_only = None
    for found_dir in glob.glob(os.path.join(results_dir, "*_frechet_*")):
        if os.path.isdir(found_dir):
            if "procrustes" not in found_dir:
                dir_frechet_only = found_dir
                beta = re.findall(r"\d+\.\d+|\d+", dir_frechet_only)
                beta = [float(num) for num in beta]
                beta = beta[0]
                list_dirs.append([dir_frechet_only, 0.0, beta])

    alpha = None
    dir_procrustes_only = None
    for found_dir in glob.glob(os.path.join(results_dir, "*_procrustes_*")):
        if os.path.isdir(found_dir):
            if "frechet" not in found_dir:
                dir_procrustes_only = found_dir
                alpha = re.findall(r"\d+\.\d+|\d+", dir_procrustes_only)
                alpha = [float(num) for num in alpha]
                alpha = alpha[0]
                list_dirs.append([dir_procrustes_only, alpha, 0.0])

    coefs = [None, None] # [alpha, beta]
    dir_both_analysis = None
    for found_dir in glob.glob(os.path.join(results_dir, "*_procrustes_*_frechet_*")):
        if os.path.isdir(found_dir):
            dir_both_analysis = found_dir
            coefs = re.findall(r"\d+\.\d+|\d+", dir_both_analysis)
            coefs = [float(num) for num in coefs]
            list_dirs.append([dir_both_analysis, coefs[0], coefs[1]])
    
    with tqdm.tqdm(
        total = len(list_dirs)*len(glob.glob(os.path.join(list_dirs[0][0], "*"))),
        desc = f"alpha: {list_dirs[0][1]}, beta: {list_dirs[0][2]}"
    ) as pbar:
        for analysis in list_dirs:
            analysis_dir = analysis[0]
            analysis_alpha = analysis[1]
            analysis_beta = analysis[2]

            pbar.desc = f"alpha: {analysis_alpha}, beta: {analysis_beta}"

            n_cluster_groups = len(glob.glob(os.path.join(analysis_dir, "*")))

            for i in range(1, 1 + n_cluster_groups):
                cluster_curve_list = [Curve(curve).curve().to_numpy() for curve in glob.glob(os.path.join(analysis[0], f"*{i}", "*.txt"))]
                resampled_curves = mapper.resample_curves(cluster_curve_list, num_points=max([len(c) for c in cluster_curve_list]))
                mapped_curves = mapper.map_curves(resampled_curves, method=method)
                average_curve = np.mean(mapped_curves, axis=0)
                average_curve = mapper.map_curves([average_curve], method=method)
                average_curve = mapper.smooth_curve(average_curve[0])
                export_multiple_curves_to_vtk(mapped_curves, os.path.join(f"{results_dir}_vtks", f"mapped_curves_p_{analysis_alpha}_f_{analysis_beta}_cluster_{i}.vtk"))
                export_multiple_curves_to_vtk(average_curve, os.path.join(f"{results_dir}_vtks", f"average_curves_p_{analysis_alpha}_f_{analysis_beta}_cluster_{i}.vtk"))
                pbar.update()