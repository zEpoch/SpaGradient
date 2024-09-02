#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mesh_utils.py
@Time    :   2024/09/02 21:11:53
@Author  :   Tao Zhou
@Version :   1.0
@Contact :   zhotoa@foxmail.com
'''

from anndata import AnnData
from pyvista import PolyData
from typing import List,Optional, Union, Literal, Tuple
from pyvista import DataSet, MultiBlock, PolyData, UnstructuredGrid

import numpy as np
import pyvista as pv
from scipy.spatial.distance import cdist

def merge_models(
    models: List[PolyData or UnstructuredGrid or DataSet],
) -> PolyData or UnstructuredGrid:
    """Merge all models in the `models` list. The format of all models must be the same."""

    merged_model = models[0]
    for model in models[1:]:
        merged_model = merged_model.merge(model)

    return merged_model


def uniform_mesh(mesh: PolyData, nsub: Optional[int] = 3, nclus: int = 20000) -> PolyData:
    """
    Generate a uniformly meshed surface using voronoi clustering.

    Args:
        mesh: A mesh model.
        nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
              nface*4**nsub where nface is the current number of faces.
        nclus: Number of voronoi clustering.

    Returns:
        new_mesh: A uniform mesh model.
    """
    # Check pyacvd package
    try:
        import pyacvd
    except ImportError:
        raise ImportError("You need to install the package `pyacvd`. \nInstall pyacvd via `pip install pyacvd`")

    # if mesh is not dense enough for uniform remeshing, increase the number of triangles in a mesh.
    if not (nsub is None):
        mesh.subdivide(nsub=nsub, subfilter="butterfly", inplace=True)

    # Uniformly remeshing.
    clustered = pyacvd.Clustering(mesh)
    clustered.cluster(nclus)
    new_mesh = clustered.create_mesh().triangulate().clean()
    return new_mesh


def _scale_model_by_distance(
    model: DataSet,
    distance: Union[int, float, list, tuple] = 1,
    scale_center: Union[list, tuple] = None,
) -> DataSet:
    # Check the distance.
    distance = distance if isinstance(distance, (tuple, list)) else [distance] * 3
    if len(distance) != 3:
        raise ValueError(
            "`distance` value is wrong. \nWhen `distance` is a list or tuple, it can only contain three elements."
        )

    # Check the scaling center.
    scale_center = model.center if scale_center is None else scale_center
    if len(scale_center) != 3:
        raise ValueError("`scale_center` value is wrong." "\n`scale_center` can only contain three elements.")

    # Scale the model based on the distance.
    for i, (d, c) in enumerate(zip(distance, scale_center)):
        p2c_bool = np.asarray(model.points[:, i] - c) > 0
        model.points[:, i][p2c_bool] += d
        model.points[:, i][~p2c_bool] -= d

    return model


def _scale_model_by_scale_factor(
    model: DataSet,
    scale_factor: Union[int, float, list, tuple] = 1,
    scale_center: Union[list, tuple] = None,
) -> DataSet:
    # Check the scaling factor.
    scale_factor = scale_factor if isinstance(scale_factor, (tuple, list)) else [scale_factor] * 3
    if len(scale_factor) != 3:
        raise ValueError(
            "`scale_factor` value is wrong."
            "\nWhen `scale_factor` is a list or tuple, it can only contain three elements."
        )

    # Check the scaling center.
    scale_center = model.center if scale_center is None else scale_center
    if len(scale_center) != 3:
        raise ValueError("`scale_center` value is wrong." "\n`scale_center` can only contain three elements.")

    # Scale the model based on the scale center.
    for i, (f, c) in enumerate(zip(scale_factor, scale_center)):
        model.points[:, i] = (model.points[:, i] - c) * f + c

    return model


def scale_model(
    model: Union[PolyData, UnstructuredGrid],
    distance: Union[float, int, list, tuple] = None,
    scale_factor: Union[float, int, list, tuple] = 1,
    scale_center: Union[list, tuple] = None,
    inplace: bool = False,
) -> Union[PolyData, UnstructuredGrid, None]:
    """
    Scale the model around the center of the model.

    Args:
        model: A 3D reconstructed model.
        distance: The distance by which the model is scaled. If `distance` is float, the model is scaled same distance
                  along the xyz axis; when the `scale factor` is list, the model is scaled along the xyz axis at
                  different distance. If `distance` is None, there will be no scaling based on distance.
        scale_factor: The scale by which the model is scaled. If `scale factor` is float, the model is scaled along the
                      xyz axis at the same scale; when the `scale factor` is list, the model is scaled along the xyz
                      axis at different scales. If `scale_factor` is None, there will be no scaling based on scale factor.
        scale_center: Scaling center. If `scale factor` is None, the `scale_center` will default to the center of the model.
        inplace: Updates model in-place.

    Returns:
        model_s: The scaled model.
    """

    model_s = model.copy() if not inplace else model

    if not (distance is None):
        model_s = _scale_model_by_distance(model=model_s, distance=distance, scale_center=scale_center)

    if not (scale_factor is None):
        model_s = _scale_model_by_scale_factor(model=model_s, scale_factor=scale_factor, scale_center=scale_center)

    model_s = model_s.triangulate()

    return model_s if not inplace else None


def rigid_transform(
    coords: np.ndarray,
    coords_refA: np.ndarray,
    coords_refB: np.ndarray,
) -> np.ndarray:
    """
    Compute optimal transformation based on the two sets of points and apply the transformation to other points.

    Args:
        coords: Coordinate matrix needed to be transformed.
        coords_refA: Referential coordinate matrix before transformation.
        coords_refB: Referential coordinate matrix after transformation.

    Returns:
        The coordinate matrix after transformation
    """
    # Check the spatial coordinates

    coords, coords_refA, coords_refB = (
        coords.copy(),
        coords_refA.copy(),
        coords_refB.copy(),
    )
    assert (
        coords.shape[1] == coords_refA.shape[1] == coords_refA.shape[1]
    ), "The dimensions of the input coordinates must be uniform, 2D or 3D."
    coords_dim = coords.shape[1]
    if coords_dim == 2:
        coords = np.c_[coords, np.zeros(shape=(coords.shape[0], 1))]
        coords_refA = np.c_[coords_refA, np.zeros(shape=(coords_refA.shape[0], 1))]
        coords_refB = np.c_[coords_refB, np.zeros(shape=(coords_refB.shape[0], 1))]

    # Compute optimal transformation based on the two sets of points.
    coords_refA = coords_refA.T
    coords_refB = coords_refB.T

    centroid_A = np.mean(coords_refA, axis=1).reshape(-1, 1)
    centroid_B = np.mean(coords_refB, axis=1).reshape(-1, 1)

    Am = coords_refA - centroid_A
    Bm = coords_refB - centroid_B
    H = Am @ np.transpose(Bm)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    # Apply the transformation to other points
    new_coords = (R @ coords.T) + t
    new_coords = np.asarray(new_coords.T)
    return new_coords[:, :2] if coords_dim == 2 else new_coords


def marching_cube_mesh(
    pc: PolyData,
    levelset: Union[int, float] = 0,
    mc_scale_factor: Union[int, float] = 1.0,
    dist_sample_num: Optional[int] = None,
):
    """
    Computes a triangle mesh from a point cloud based on the marching cube algorithm.
    Algorithm Overview:
        The algorithm proceeds through the scalar field, taking eight neighbor locations at a time (thus forming an
        imaginary cube), then determining the polygon(s) needed to represent the part of the iso-surface that passes
        through this cube. The individual polygons are then fused into the desired surface.
    Args:
        pc: A point cloud model.
        levelset: The levelset of iso-surface. It is recommended to set levelset to 0 or 0.5.
        mc_scale_factor: The scale of the model. The scaled model is used to construct the mesh model.
        dist_sample_num: The down-sampling number when calculating the scaling factor using the minimum distance. Set to 100 for computation efficiency.

    Returns:
        A mesh model.
    """
    try:
        import mcubes
    except ImportError:
        raise ImportError(
            "You need to install the package `mcubes`." "\nInstall mcubes via `pip install --upgrade PyMCubes`"
        )

    pc = pc.copy()

    # Move the model so that the coordinate minimum is at (0, 0, 0).
    raw_points = np.asarray(pc.points)
    pc.points = new_points = raw_points - np.min(raw_points, axis=0)

    # Generate new models for calculatation.
    if dist_sample_num is None:
        dist = cdist(XA=new_points, XB=new_points, metric="euclidean")
        row, col = np.diag_indices_from(dist)
        dist[row, col] = None
    else:
        rand_idx = (
            np.random.choice(new_points.shape[0], dist_sample_num)
            if new_points.shape[0] >= dist_sample_num
            else np.arange(new_points.shape[0])
        )
        dist = cdist(XA=new_points[rand_idx, :], XB=new_points, metric="euclidean")
        dist[np.arange(rand_idx.shape[0]), rand_idx] = None
    max_dist = np.nanmin(dist, axis=1).max()
    mc_sf = max_dist * mc_scale_factor

    scale_pc = scale_model(model=pc, scale_factor=1 / mc_sf, scale_center=(0, 0, 0))
    scale_pc_points = scale_pc.points = np.ceil(np.asarray(scale_pc.points)).astype(np.int64)

    # Generate grid for calculatation based on new model.
    volume_array = np.zeros(
        shape=[
            scale_pc_points[:, 0].max() + 3,
            scale_pc_points[:, 1].max() + 3,
            scale_pc_points[:, 2].max() + 3,
        ]
    )
    volume_array[scale_pc_points[:, 0], scale_pc_points[:, 1], scale_pc_points[:, 2]] = 1

    # Extract the iso-surface based on marching cubes algorithm.
    # volume_array = mcubes.smooth(volume_array)
    vertices, triangles = mcubes.marching_cubes(volume_array, levelset)

    if len(vertices) == 0:
        raise ValueError(f"The point cloud cannot generate a surface mesh with `marching_cube` method.")

    v = np.asarray(vertices).astype(np.float64)
    f = np.asarray(triangles).astype(np.int64)
    f = np.c_[np.full(len(f), 3), f]

    # Generate mesh model.
    mesh = pv.PolyData(v, f.ravel()).extract_surface().triangulate()
    mesh.clean(inplace=True)
    mesh = scale_model(model=mesh, scale_factor=mc_sf, scale_center=(0, 0, 0))

    # Transform.
    scale_pc = scale_model(model=scale_pc, scale_factor=mc_sf, scale_center=(0, 0, 0))
    mesh.points = rigid_transform(
        coords=np.asarray(mesh.points), coords_refA=np.asarray(scale_pc.points), coords_refB=raw_points
    )
    return mesh


def uniform_larger_pc(
    pc: PolyData,
    alpha: Union[float, int] = 0,
    nsub: Optional[int] = 5,
    nclus: int = 20000,
) -> PolyData:
    """
    Generates a uniform point cloud with a larger number of points.
    If the number of points in the original point cloud is too small or the distribution of the original point cloud is
    not uniform, making it difficult to construct the surface, this method can be used for preprocessing.

    Args:
        pc: A point cloud model.
        alpha: Specify alpha (or distance) value to control output of this filter.
               For a non-zero alpha value, only edges or triangles contained within a sphere centered at mesh vertices
               will be output. Otherwise, only triangles will be output.
        nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
              nface*4**nsub where nface is the current number of faces.
        nclus: Number of voronoi clustering.

    Returns:
        new_pc: A uniform point cloud with a larger number of points.
    """
    coords = np.asarray(pc.points)
    coords_z = np.unique(coords[:, 2])

    slices = []
    for z in coords_z:
        slice_coords = coords[coords[:, 2] == z]
        slice_cloud = pv.PolyData(slice_coords)
        if len(slice_coords) >= 3:
            slice_plane = slice_cloud.delaunay_2d(alpha=alpha).triangulate().clean()
            uniform_plane = uniform_mesh(mesh=slice_plane, nsub=nsub, nclus=nclus)
            slices.append(uniform_plane)
        else:
            slices.append(slice_cloud)

    slices_mesh = merge_models(models=slices)
    new_pc = pv.PolyData(slices_mesh.points).clean()
    return new_pc


def marching_cube_mesh(
    pc: PolyData,
    levelset: Union[int, float] = 0,
    mc_scale_factor: Union[int, float] = 1.0,
    dist_sample_num: Optional[int] = None,
):
    """
    Computes a triangle mesh from a point cloud based on the marching cube algorithm.
    Algorithm Overview:
        The algorithm proceeds through the scalar field, taking eight neighbor locations at a time (thus forming an
        imaginary cube), then determining the polygon(s) needed to represent the part of the iso-surface that passes
        through this cube. The individual polygons are then fused into the desired surface.

    Args:
        pc: A point cloud model.
        levelset: The levelset of iso-surface. It is recommended to set levelset to 0 or 0.5.
        mc_scale_factor: The scale of the model. The scaled model is used to construct the mesh model.
        dist_sample_num: The down-sampling number when calculating the scaling factor using the minimum distance. Set to 100 for computation efficiency.

    Returns:
        A mesh model.
    """
    try:
        import mcubes
    except ImportError:
        raise ImportError(
            "You need to install the package `mcubes`." "\nInstall mcubes via `pip install --upgrade PyMCubes`"
        )

    pc = pc.copy()

    # Move the model so that the coordinate minimum is at (0, 0, 0).
    raw_points = np.asarray(pc.points)
    pc.points = new_points = raw_points - np.min(raw_points, axis=0)

    # Generate new models for calculatation.
    if dist_sample_num is None:
        dist = cdist(XA=new_points, XB=new_points, metric="euclidean")
        row, col = np.diag_indices_from(dist)
        dist[row, col] = None
    else:
        rand_idx = (
            np.random.choice(new_points.shape[0], dist_sample_num)
            if new_points.shape[0] >= dist_sample_num
            else np.arange(new_points.shape[0])
        )
        dist = cdist(XA=new_points[rand_idx, :], XB=new_points, metric="euclidean")
        dist[np.arange(rand_idx.shape[0]), rand_idx] = None
    max_dist = np.nanmin(dist, axis=1).max()
    mc_sf = max_dist * mc_scale_factor

    scale_pc = scale_model(model=pc, scale_factor=1 / mc_sf, scale_center=(0, 0, 0))
    scale_pc_points = scale_pc.points = np.ceil(np.asarray(scale_pc.points)).astype(np.int64)

    # Generate grid for calculatation based on new model.
    volume_array = np.zeros(
        shape=[
            scale_pc_points[:, 0].max() + 3,
            scale_pc_points[:, 1].max() + 3,
            scale_pc_points[:, 2].max() + 3,
        ]
    )
    volume_array[scale_pc_points[:, 0], scale_pc_points[:, 1], scale_pc_points[:, 2]] = 1

    # Extract the iso-surface based on marching cubes algorithm.
    # volume_array = mcubes.smooth(volume_array)
    vertices, triangles = mcubes.marching_cubes(volume_array, levelset)

    if len(vertices) == 0:
        raise ValueError(f"The point cloud cannot generate a surface mesh with `marching_cube` method.")

    v = np.asarray(vertices).astype(np.float64)
    f = np.asarray(triangles).astype(np.int64)
    f = np.c_[np.full(len(f), 3), f]

    # Generate mesh model.
    mesh = pv.PolyData(v, f.ravel()).extract_surface().triangulate()
    mesh.clean(inplace=True)
    mesh = scale_model(model=mesh, scale_factor=mc_sf, scale_center=(0, 0, 0))

    # Transform.
    scale_pc = scale_model(model=scale_pc, scale_factor=mc_sf, scale_center=(0, 0, 0))
    mesh.points = rigid_transform(
        coords=np.asarray(mesh.points), coords_refA=np.asarray(scale_pc.points), coords_refB=raw_points
    )
    return mesh


def clean_mesh(mesh: PolyData) -> PolyData:
    """Removes unused points and degenerate cells."""

    sub_meshes = mesh.split_bodies()
    n_mesh = len(sub_meshes)

    if n_mesh == 1:
        return mesh
    else:
        inside_number = []
        for i, main_mesh in enumerate(sub_meshes[:-1]):
            main_mesh = pv.PolyData(main_mesh.points, main_mesh.cells)
            for j, check_mesh in enumerate(sub_meshes[i + 1 :]):
                check_mesh = pv.PolyData(check_mesh.points, check_mesh.cells)
                inside = check_mesh.select_enclosed_points(main_mesh, check_surface=False).threshold(0.5)
                inside = pv.PolyData(inside.points, inside.cells)
                if check_mesh == inside:
                    inside_number.append(i + 1 + j)

        cm_number = list(set([i for i in range(n_mesh)]).difference(set(inside_number)))
        if len(cm_number) == 1:
            cmesh = sub_meshes[cm_number[0]]
        else:
            cmesh = merge_models([sub_meshes[i] for i in cm_number])

        return pv.PolyData(cmesh.points, cmesh.cells)


def fix_mesh(mesh: PolyData) -> PolyData:
    """Repair the mesh where it was extracted and subtle holes along complex parts of the mesh."""

    # Check pymeshfix package
    try:
        import pymeshfix as mf
    except ImportError:
        raise ImportError(
            "You need to install the package `pymeshfix`. \nInstall pymeshfix via `pip install pymeshfix`"
        )

    meshfix = mf.MeshFix(mesh)
    meshfix.repair(verbose=False)
    fixed_mesh = meshfix.mesh.triangulate().clean()

    if fixed_mesh.n_points == 0:
        raise ValueError(
            f"The surface cannot be Repaired. " f"\nPlease change the method or parameters of surface reconstruction."
        )

    return fixed_mesh


def smooth_mesh(mesh: PolyData, n_iter: int = 100, **kwargs) -> PolyData:
    """
    Adjust point coordinates using Laplacian smoothing.
    https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyData.smooth.html#pyvista.PolyData.smooth

    Args:
        mesh: A mesh model.
        n_iter: Number of iterations for Laplacian smoothing.
        **kwargs: The rest of the parameters in pyvista.PolyData.smooth.

    Returns:
        smoothed_mesh: A smoothed mesh model.
    """

    smoothed_mesh = mesh.smooth(n_iter=n_iter, **kwargs)

    return smoothed_mesh


def construct_surface(
    pc: PolyData,
    uniform_pc: bool = False,
    uniform_pc_alpha: Union[float, int] = 0,
    cs_args: Optional[dict] = None,
    nsub: Optional[int] = 3,
    nclus: int = 20000,
    smooth: Optional[int] = 3000,
    scale_distance: Union[float, int, list, tuple] = None,
    scale_factor: Union[float, int, list, tuple] = None,
) -> Tuple[Union[PolyData, UnstructuredGrid, None], PolyData, Optional[str]]:
    """
    Surface mesh reconstruction based on 3D point cloud model.

    Args:
        pc: A point cloud model.
        uniform_pc: Generates a uniform point cloud with a larger number of points.
        uniform_pc_alpha: Specify alpha (or distance) value to control output of this filter.
        cs_args: Parameters for various surface reconstruction methods. Available ``cs_args`` are:
                * ``'pyvista'``: {'alpha': 0}
                * ``'alpha_shape'``: {'alpha': 2.0}
                * ``'ball_pivoting'``: {'radii': [1]}
                * ``'poisson'``: {'depth': 8, 'width'=0, 'scale'=1.1, 'linear_fit': False, 'density_threshold': 0.01}
                * ``'marching_cube'``: {'levelset': 0, 'mc_scale_factor': 1, 'dist_sample_num': 100}
        nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
              nface*4**nsub where nface is the current number of faces.
        nclus: Number of voronoi clustering.
        smooth: Number of iterations for Laplacian smoothing.
        scale_distance: The distance by which the model is scaled. If ``scale_distance`` is float, the model is scaled same
                        distance along the xyz axis; when the ``scale factor`` is list, the model is scaled along the xyz
                        axis at different distance. If ``scale_distance`` is None, there will be no scaling based on distance.
        scale_factor: The scale by which the model is scaled. If ``scale factor`` is float, the model is scaled along the
                      xyz axis at the same scale; when the ``scale factor`` is list, the model is scaled along the xyz
                      axis at different scales. If ``scale_factor`` is None, there will be no scaling based on scale factor.

    Returns:
        uniform_surf: A reconstructed surface mesh, which contains the following properties:
            ``uniform_surf.cell_data[key_added]``, the ``label`` array;
            ``uniform_surf.cell_data[f'{key_added}_rgba']``, the rgba colors of the ``label`` array.
    """

    # Generates a uniform point cloud with a larger number of points or not.
    cloud = uniform_larger_pc(pc=pc, alpha=uniform_pc_alpha, nsub=3, nclus=20000) if uniform_pc else pc.copy()

    _cs_args = {"levelset": 0, "mc_scale_factor": 1, "dist_sample_num": None}
    if not (cs_args is None):
        _cs_args.update(cs_args)

    surf = marching_cube_mesh(
        pc=cloud,
        levelset=_cs_args["levelset"],
        mc_scale_factor=_cs_args["mc_scale_factor"],
        dist_sample_num=_cs_args["dist_sample_num"],
    )

    # Removes unused points and degenerate cells.
    csurf = clean_mesh(mesh=surf)

    uniform_surfs = []
    for sub_surf in csurf.split_bodies():
        # Repair the surface mesh where it was extracted and subtle holes along complex parts of the mesh
        sub_fix_surf = fix_mesh(mesh=sub_surf.extract_surface())

        # Get a uniformly meshed surface using voronoi clustering.
        sub_uniform_surf = uniform_mesh(mesh=sub_fix_surf, nsub=nsub, nclus=nclus)
        uniform_surfs.append(sub_uniform_surf)
    uniform_surf = merge_models(models=uniform_surfs)
    uniform_surf = uniform_surf.extract_surface().triangulate().clean()

    # Adjust point coordinates using Laplacian smoothing.
    if not (smooth is None):
        uniform_surf = smooth_mesh(mesh=uniform_surf, n_iter=smooth)

    # Scale the surface mesh.
    uniform_surf = scale_model(model=uniform_surf, distance=scale_distance, scale_factor=scale_factor)

    return uniform_surf


def construct_pc(adata: AnnData, 
                 spatial_key: str = "spatial"):
    adata = adata.copy()
    bucket_xyz = adata.obsm[spatial_key].astype(np.float64)
    pc = pv.PolyData(bucket_xyz)
    pc.point_data["obs_index"] = np.array(adata.obs_names.tolist())
    
    return pc
