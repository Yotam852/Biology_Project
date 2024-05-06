import logging

import meshio
import numpy as np
import pandas as pd

logger = logging.getLogger(name=__name__)


def to_mesh(sheet):
    cell_types = {
        3: "triangle",
        4: "quad"}
    max_nsides = sheet.face_df["num_sides"].max() + 1
    cell_types.update(
        {
            i: f'{"polygon"}' for i in range(5, max_nsides)
        }
    )
    sheet.reset_index(order=True)
    sheet.edge_df["f_sides"] = sheet.upcast_face("num_sides")
    cells = [
        ('line', sheet.edge_df[["srce", "trgt"]].to_numpy())
    ]
    for n, edges in sheet.edge_df.groupby("f_sides"):
        polys = np.vstack(edges.groupby("face").apply(lambda e: e["srce"].values).values)
        cells.append((cell_types[n], polys))
    mesh = meshio.Mesh(points=sheet.vert_df[sheet.coords].to_numpy(), cells=cells)
    return mesh


def save_triangular_mesh(filename, eptm):
    coords = eptm.coords
    eptm.reset_index(order=True)

    if (filename[-3:] == "ply") or (filename[-3:] == "obj"):
        points, faces = eptm.triangular_mesh(
            coords=coords,
        )
        cells = []
        for f in faces:
            cells.append(("triangle", np.array([f])))
        mesh = meshio.Mesh(points, cells)
        mesh.write(filename)
    elif filename[-3:] == "vtk":
        points, faces = eptm.vertex_mesh(coords=coords, vertex_normals=False)
        cells = []
        for f in faces:
            cells.append(("triangle", np.array([f])))
        mesh = meshio.Mesh(points, cells)
        meshio.vtk.write(filename, mesh)
    else:
        print("This format %s is not taking in charge for now", filename[-3:])

    logger.info("Saved %s as a meshio file", filename)


def import_triangular_mesh(filename):
    if (filename.endswith("ply")) or (filename.endswith("obj")):
        mesh = meshio.read(filename)
        vert_ = pd.DataFrame(mesh.points, columns=list("xyz"))
        edge_ = pd.DataFrame(columns=["srce", "trgt", "face"])
        face_ = pd.DataFrame(columns=list("xyz"))
        cpt = 0
        for c in mesh.cells[0].data:
            edge_.loc[cpt * 3] = [c[0], c[1], cpt]
            edge_.loc[cpt * 3 + 1] = [c[1], c[2], cpt]
            edge_.loc[cpt * 3 + 2] = [c[2], c[0], cpt]

            face_.loc[cpt] = [0, 0, 0]
            cpt += 1
        data = {"vert": vert_, "edge": edge_, "face": face_}

        return data
    else:
        print("This format %s is not taking in charge for now", filename[-3:])


def save_mesh(filename, eptm):
    """
    Saving mesh which is not triangular, need to call specifically the write
    function according to the format and not the generic write function
    """
    coords = eptm.coords
    eptm.reset_index(order=True)

    points, faces = eptm.vertex_mesh(coords=coords, vertex_normals=False)
    cells = []
    for f in faces:
        cells.append(("triangle", np.array([f])))
    mesh = meshio.Mesh(points, cells)

    if filename[-3:] == "ply":
        meshio.ply.write(filename, mesh)
    elif filename[-3:] == "vtk":
        meshio.vtk.write(filename, mesh)
    else:
        print("This format %s is not taking in charge for now", filename[-3:])

    logger.info("Saved %s as a meshio file", filename)


def import_mesh(filename):
    """
    Can only import ply file as polygonal sheet.
    """
    if filename[-3:] == "ply":
        mesh = meshio.read(filename)
        vert_ = pd.DataFrame(mesh.points, columns=list("xyz"))
        edge_ = pd.DataFrame(columns=["srce", "trgt", "face"])
        face_ = pd.DataFrame(columns=["x", "y", "z", "num_sides"])
        cpt_edge = 0
        cpt_face = 0

        for cells in mesh.cells:
            for c in cells.data:
                poly = len(c)
                for i in range(poly - 1):
                    edge_.loc[cpt_edge] = [c[i], c[i + 1], cpt_face]
                    cpt_edge += 1
                edge_.loc[cpt_edge] = [c[poly - 1], c[0], cpt_face]
                cpt_edge += 1
                face_.loc[cpt_face] = [0, 0, 0, poly]
                cpt_face += 1

        data = {"vert": vert_, "edge": edge_, "face": face_}

        return data
    else:
        print("This format %s is not taking in charge for now", filename[-3:])
