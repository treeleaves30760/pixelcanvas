"""A reference with transparency is flattened one way everywhere: over white paper (project.flatten_alpha), by the
shell's loader and by the pipeline's."""
import numpy as np
from PIL import Image

from pixelcanvas.easel import pipeline
from pixelcanvas.easel.project import load_image_rgb


def _rgba_ref(path):
    """A red disc on a fully transparent square whose hidden RGB is black, with a soft (partly transparent) rim."""
    h = w = 96
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot(x - w / 2, y - h / 2)
    a = np.clip((30 - r) / 3.0, 0, 1)
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 0] = 200
    img[..., 1] = 40
    img[..., 2] = 50
    img[a == 0, :3] = 0                                   # hidden colour under the transparent part: black
    img[..., 3] = (a * 255).round().astype(np.uint8)
    Image.fromarray(img, "RGBA").save(path)
    return a


def test_rgba_reference_is_flattened_over_white_by_shell_and_pipeline(tmp_path):
    a = _rgba_ref(tmp_path / "ref.png")
    shell = load_image_rgb(tmp_path / "ref.png")
    pipe = pipeline.load_ref(str(tmp_path / "ref.png"))
    assert shell.dtype == pipe.dtype == np.uint8 and np.array_equal(shell, pipe)
    assert (shell[a == 0] == 255).all()                   # transparent -> white paper, not the hidden black
    rim = (a > 0.2) & (a < 0.8)
    assert (shell[rim, 1] > 40).all()                     # partly transparent: mixed with white
    out = tmp_path / "pipe"
    pipeline.run_reference_pipeline(str(tmp_path / "ref.png"), str(out), style="cel", log=lambda *x: None,
                                    work_scale="auto", fine="streaks", max_level=1, stipple=2, stipple_edges=False,
                                    deliver_filter="sharpen", contours=True, smooth_tol=0.8)
    fin = np.asarray(Image.open(out / "final.png").convert("RGB")).astype(np.float32)
    clear = a == 0
    assert fin[clear].mean() > 245 and np.percentile(fin[clear].min(-1), 5) > 230, fin[clear].mean()
