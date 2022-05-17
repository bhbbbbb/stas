import os

from stas.config import Config
from stas.stas_dataset import StasDataset # , get_labels_ratio
from stas.stas_model_utils import StasModelUtils


def _inference(
    config: Config,
    utils: StasModelUtils,
    outdir: str,
    test_dir: str,
    num_output: int,
):
    inf_set = StasDataset(config, 'inference', test_dir=test_dir)
    utils.splash(
        inf_set,
        num_of_output=num_output,
        out_dir=outdir,
    )
    return

def inference(
    config: Config, utils: StasModelUtils, test_dir: str, num_output: int = None,
    out_dir: str = None
):
    output_dirname = os.path.basename(test_dir) + '_inf'
    if out_dir is None:
        outdir = os.path.join(test_dir, '..', output_dirname)
    _inference(config, utils, test_dir=test_dir, num_output=num_output, outdir=outdir)
    return

def inference_by_valid(
    config: Config, utils: StasModelUtils, num_output: int = None
):
    outdir = os.path.join(utils.root, f'splash_{utils.start_epoch}')
    num_output = num_output or 5
    _inference(config, utils, test_dir=None, num_output=num_output, outdir=outdir)
    return
