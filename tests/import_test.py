# TODO method to test import of all modules in FoldfFlow, and runner.


def test_import():
    from foldflow.data import (
        all_atom,
        pdb_data_loader,
        protein,
        residue_constants,
        utils,
    )
    from foldflow.models import r3_fm, se3_fm, so3_fm
    from tools.analysis import metrics, plotting, utils
