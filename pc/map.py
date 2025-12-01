from typing import List, Optional

import tqdm
import trimesh
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import interp1d

class CurveSurfaceMapper:
    def __init__(self, stl_file: Optional[str] = None):
        """
        Initialize the mapper with an STL file.

        Parameters:
        stl_file (str): Path to the STL file to load the 3D object.
        """
        if stl_file is not None:
            self.mesh = trimesh.load(stl_file)
            self.surface_points = np.array(self.mesh.vertices)
            self.kdtree = KDTree(self.surface_points)
        else:
            self.mesh = None
            self.surface_points = None
            self.kdtree = None

    def enhance_curve_resolution(self, curve: np.ndarray, num_points: int = 100) -> np.ndarray:
        """
        Increase the resolution of a curve by spline interpolation.

        Parameters:
        curve (np.ndarray): The input curve as a NumPy array of shape (n, 3).
        num_points (int): The desired number of points in the enhanced curve.

        Returns:
        np.ndarray: The enhanced curve as a NumPy array of shape (num_points, 3).
        """
        f = interp1d(np.linspace(0, 1, curve.shape[0]), curve, axis=0)
        enhanced_curve = f(np.linspace(0, 1, num_points))
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

    def project_to_surface_nearest(self, curve: np.ndarray) -> np.ndarray:
        """
        Map a curve to the surface using the nearest-point projection.

        Parameters:
        curve (np.ndarray): The input curve as a NumPy array of shape (n, 3).

        Returns:
        np.ndarray: The mapped curve on the surface as a NumPy array of shape (n, 3).

        Raises:
        ValueError: If the STL file was not provided during initialization.
        """
        if self.surface_points is not None:
            _, indices = self.kdtree.query(curve)
            return self.surface_points[indices]
        else:
            raise ValueError("To use project_to_surface_nearest method, you must provide the STL file during initialization.")

    def project_to_surface_ray_casting(self, curve: np.ndarray, ray_directions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Map a curve to the surface using ray-casting.

        Parameters:
        curve (np.ndarray): The input curve as a NumPy array of shape (n, 3).
        ray_directions (Optional[np.ndarray]): The directions of the rays as a NumPy array of shape (n, 3). 
                                                If None, defaults to uniform downward direction.

        Returns:
        np.ndarray: The intersection points on the surface as a NumPy array of shape (m, 3), where m <= n.

        Raises:
        ValueError: If the STL file was not provided during initialization.
        """
        if self.mesh is not None:
            if ray_directions is None:
                ray_directions = -np.ones_like(curve)  # Example: uniform direction
            intersections, _, _ = self.mesh.ray.intersects_location(
                ray_origins=curve, ray_directions=ray_directions
            )
            return intersections
        else:
            raise ValueError("To use project_to_surface_ray_casting method, you must provide the STL file during initialization.")

    def project_to_surface_sdf(self, curve: np.ndarray, max_iterations: int = 100, tolerance: float = 1e-6) -> np.ndarray:
        """
        Map a curve to the surface using signed distance field (SDF) alignment.

        Parameters:
        curve (np.ndarray): The input curve as a NumPy array of shape (n, 3).
        max_iterations (int): The maximum number of iterations for the alignment process.
        tolerance (float): The tolerance for stopping the alignment process.

        Returns:
        np.ndarray: The mapped curve on the surface as a NumPy array of shape (n, 3).

        Raises:
        ValueError: If the STL file was not provided during initialization.
        """
        if self.mesh is not None:
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
        else:
            raise ValueError("To use project_to_surface_sdf method, you must provide the STL file during initialization.")

    def map_curves(self, curves: List[np.ndarray], method: str = "nearest", **kwargs) -> List[np.ndarray]:
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
        for curve in tqdm.tqdm(curves, desc=f"Mapping Curves with {method} technique"):
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