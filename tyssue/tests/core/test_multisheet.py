import numpy as np
from scipy.spatial import Voronoi

import tyssue
from tyssue.core.multisheet import MultiSheet
from tyssue.generation import from_2d_voronoi, hexa_grid2d


def test_multisheet():

    base_specs = tyssue.config.geometry.flat_sheet()
    specs = base_specs.copy()
    specs["face"]["layer"] = 0
    specs["vert"]["layer"] = 0
    specs["vert"]["depth"] = 0.0
    specs["edge"]["layer"] = 0
    specs["settings"]["geometry"] = "flat"
    specs["settings"]["interpolate"] = {"function": "multiquadric", "smooth": 0}

    layer_args = [
        (24, 24, 1, 1, 0.4),
        (16, 16, 2, 2, 1),
        (24, 24, 1, 1, 0.4),
        (24, 24, 1, 1, 0.4),
    ]
    dz = 1.0

    layer_datasets = []
    for i, args in enumerate(layer_args):
        centers = hexa_grid2d(*args)
        data = from_2d_voronoi(Voronoi(centers))
        data["vert"]["z"] = i * dz
        layer_datasets.append(data)

    msheet = MultiSheet("more", layer_datasets, specs)
    bbox = [[0, 25], [0, 25]]
    for sheet in msheet:
        edge_out = sheet.cut_out(bbox, coords=["x", "y"])
        sheet.remove(edge_out)

    assert len(msheet) == 4
    datasets = msheet.concat_datasets()
    assert np.all(np.isfinite(datasets["vert"][["x", "y", "z"]]))

    msheet.update_interpolants()
    for interp in msheet.interpolants:
        assert np.isfinite(interp(10, 10))
