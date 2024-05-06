import numpy as np
import pandas as pd
from pytest import raises
from scipy.spatial import Voronoi

from tyssue import BulkGeometry, Epithelium, config, generation
from tyssue.core.sheet import Sheet, get_opposite
from tyssue.generation import (
    extrude,
    from_2d_voronoi,
    from_3d_voronoi,
    hexa_grid2d,
    hexa_grid3d,
    subdivide_faces,
)


def test_3faces():

    datasets, _ = generation.three_faces_sheet()
    assert datasets["edge"].shape[0] == 18
    assert datasets["face"].shape[0] == 3
    assert datasets["vert"].shape[0] == 13


def test_from_3d_voronoi():

    grid = hexa_grid3d(6, 4, 3)
    datasets = from_3d_voronoi(Voronoi(grid))
    assert datasets["vert"].shape[0] == 139
    assert datasets["edge"].shape[0] == 1272
    assert datasets["face"].shape[0] == 282
    assert datasets["cell"].shape[0] == 70
    bulk = Epithelium("bulk", datasets, config.geometry.bulk_spec())
    bulk.reset_index()
    bulk.reset_topo()
    BulkGeometry.update_all(bulk)
    bulk.sanitize()

    # GH 137
    assert (
        bulk.edge_df.groupby("face").apply(lambda df: df["cell"].unique().size).max()
        == 1
    )
    assert bulk.validate()


def test_from_2d_voronoi():

    grid = hexa_grid2d(6, 4, 1, 1)
    datasets = from_2d_voronoi(Voronoi(grid))
    assert datasets["vert"].shape[0] == 32
    assert datasets["edge"].shape[0] == 82
    assert datasets["face"].shape[0] == 24


def test_extrude():

    datasets, specs = generation.three_faces_sheet()
    sheet = Sheet("test", datasets, specs)
    extruded = extrude(sheet.datasets, method="translation")
    assert extruded["cell"].shape[0] == 3
    assert extruded["face"].shape[0] == 24
    assert extruded["edge"].shape[0] == 108
    assert extruded["vert"].shape[0] == 26


def test_subdivide():

    datasets, specs = generation.three_faces_sheet()
    sheet = Sheet("test", datasets, specs)
    subdivided = subdivide_faces(sheet, [0])
    assert subdivided["face"].shape[0] == 3
    assert subdivided["edge"].shape[0] == 30
    assert subdivided["vert"].shape[0] == 14

    datasets_3d = extrude(datasets, method="translation")
    sheet_3d = Epithelium("test3d", datasets_3d)
    subdivided_3d = subdivide_faces(sheet_3d, [0])
    assert subdivided_3d["face"].shape[0] == 24
    assert subdivided_3d["edge"].shape[0] == 120
    assert subdivided_3d["vert"].shape[0] == 27
    assert subdivided_3d["cell"].shape[0] == 3


def test_extrude_invalid_method():
    datasets, _ = generation.three_faces_sheet()
    with raises(ValueError):
        extrude(datasets, method="invalid_method")


def test_hexagrid3d_noise():
    np.random.seed(1)
    grid = hexa_grid3d(6, 4, 3, noise=0.1)
    datasets = from_3d_voronoi(Voronoi(grid))
    assert datasets["vert"].shape[0] == 318
    assert datasets["edge"].shape[0] == 3300
    assert datasets["face"].shape[0] == 670
    assert datasets["cell"].shape[0] == 72


def test_anchors():
    datasets, specs = generation.three_faces_sheet()
    sheet = Sheet("test_anchors", datasets, specs)

    sheet.edge_df["opposite"] = get_opposite(sheet.edge_df)

    expected_dict = {
        18: [1, 13],
        19: [2, 14],
        20: [3, 15],
        21: [4, 16],
        22: [5, 17],
        23: [6, 18],
        24: [7, 19],
        25: [8, 20],
        26: [9, 21],
        27: [10, 22],
        28: [11, 23],
        29: [12, 24],
    }

    expected_res = pd.DataFrame.from_dict(expected_dict, orient="index")
    expected_res.columns = ["srce", "trgt"]
    generation.create_anchors(sheet)

    res_srce_trgt_anchors = sheet.edge_df.loc[18:, ["srce", "trgt"]]
    assert res_srce_trgt_anchors.equals(expected_res)


def test_extract():
    datasets, specs = generation.three_faces_sheet()
    sheet = Sheet("test_sheet_extract_coordinate", datasets, specs)
    sheet.face_df.loc[0, "is_alive"] = 0
    subsheet = sheet.extract("is_alive")

    assert subsheet.face_df["is_alive"].all()
    assert subsheet.Nf == 2


def test_sheet_extract_coordinate():
    grid = hexa_grid2d(6, 4, 3, 3)
    datasets = from_2d_voronoi(Voronoi(grid))
    sheet = Sheet("test_extract_bounding_box", datasets)
    subsheet = sheet.extract_bounding_box(
        [sheet.face_df["x"].min(), sheet.face_df["x"].max() / 2],
        [sheet.face_df["y"].min(), sheet.face_df["y"].max() / 2],
    )
    assert subsheet.face_df["x"].max() <= sheet.face_df["x"].max() / 2
    assert subsheet.face_df["x"].min() >= sheet.face_df["x"].min()
    assert subsheet.face_df["y"].max() <= sheet.face_df["y"].max() / 2
    assert subsheet.face_df["y"].min() >= sheet.face_df["y"].min()
    assert subsheet.face_df["z"].max() <= sheet.face_df["z"].max()
    assert subsheet.face_df["z"].min() >= sheet.face_df["z"].min()
