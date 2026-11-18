"""Construction sketch: axis lines and ellipses / proportion boxes of the big masses come before the outlines."""
import numpy as np

from pixelcanvas.ops import check_mark_rules
from pixelcanvas.stage_sketch import sketch_ops

from .test_hand_stages import cel_ref


def test_sketch_starts_with_construction_lines():
    ref = cel_ref()
    ops = sketch_ops(ref, None, seed=1)
    assert ops and not check_mark_rules(ops)
    con = [o for o in ops if o.opacity < 0.46 and len(o.pts) >= 2]
    assert con, "no construction lines"
    first = ops[: len(con)]
    assert all(o.opacity < 0.46 for o in first)
    # one of them is a closed ellipse around the round disc (centre near (90, 100), radius about 60)
    closed = [o for o in con if len(o.pts) > 20 and np.hypot(*(o.pts[0] - o.pts[-1])) < 12]
    assert any(abs(o.pts[:, 0].mean() - 95) < 20 and abs(o.pts[:, 1].mean() - 105) < 20 for o in closed)
