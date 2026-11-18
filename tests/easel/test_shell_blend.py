"""The blend of a soft-shaded reference in the shell: offered once, in the stage where the pipeline blends (cel: after
the shade, on flat and shade), never on the detail layer; a repeat says it changes nothing and names the next command;
a shell session with the blend there reaches the pipeline's result (real engine)."""
import numpy as np
import pytest
from PIL import Image

from pixelcanvas.easel import cli, workflow
from pixelcanvas.easel.project import Project

from .shell_helpers import run, synthetic_ref


def soft_ref(w: int, h: int) -> np.ndarray:
    """The cartoon of synthetic_ref with soft shading: a sky ramp behind it, a light-to-shadow ramp across the face
    and the ground."""
    img = synthetic_ref(w, h).astype(np.float32)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = (y < h * 0.65) & (np.abs(img - 236).sum(-1) < 3)
    ramp = np.stack([150 + 80 * y / h, 180 + 50 * y / h, 240 - 20 * y / h], -1)
    img[sky] = ramp[sky]
    cx, cy, r = w // 2, int(h * 0.38), int(w * 0.25)
    face = ((x - cx) ** 2 + (y - cy) ** 2 < (r - 3) ** 2) & (np.abs(img - (240, 200, 170)).sum(-1) < 3)
    shade = np.clip((x - cx + r) / (2.0 * r), 0, 1)[..., None]
    img[face] = (img * (1.0 - 0.35 * shade))[face]
    ground = y >= h * 0.65
    img[ground] = (img * (1.0 - 0.4 * ((y - h * 0.65) / (h * 0.35)))[..., None])[ground]
    return np.clip(img, 0, 255).astype(np.uint8)


def _new(tmp_path, monkeypatch, size, boxes):
    from pixelcanvas.easel import pipeline
    monkeypatch.chdir(tmp_path)
    ref = soft_ref(*size)
    assert pipeline.soft_ramp_share(ref) >= 0.15            # a soft-shaded reference: the pipeline blends it
    Image.fromarray(ref).save(tmp_path / "ref.png")
    assert run("new", "pic", "--ref", "ref.png")[0] == 0
    run("style", "cel")
    for name, box in boxes.items():
        run("region", "add", name, "--box", box, "--no-refine", *(["--focus", "3"] if name == "top" else []))
    assert run("stage", "done")[0] == 0


def _layer_bytes(root, name):
    f = root / "layers" / f"{name}.png"
    return f.read_bytes() if f.exists() else None


BOXES = {"top": "0,0,1000,400", "mid": "0,400,1000,700", "low": "0,700,1000,1000"}


def test_blend_is_offered_once_in_shade_and_never_touches_the_detail_layer(tmp_path, monkeypatch):
    _new(tmp_path, monkeypatch, (200, 160), BOXES)
    root = tmp_path / "pic.easel"
    st = Project(root).state
    blend = [t for t in st["todo"] if t.get("key") == "blend"]
    assert [t["stage"] for t in blend] == ["shade"]            # cel: once, after the shade (not again in finish)
    while Project(root).state["stage"] != "flat":
        run("stage", "done", "--because", "test")
    assert run("flat", "all")[0] == 0
    run("stage", "done", "--because", "test")
    assert Project(root).state["stage"] == "shade"
    code, out = run("shade", "all")
    assert code == 0, out
    assert "NEXT: easel smooth all" in out, out                  # shade all does not tick the blend away
    code, out = run("smooth", "all")
    assert code == 0 and "layer shade" in out and "layer detail" not in out, out
    assert "NEXT: easel smooth all" not in out, out              # ticked before NEXT is chosen
    step = Project(root).state["step"]
    code, out = run("smooth", "all")                             # the repeat: no step, says so, names the next
    assert code == 0 and "changes nothing" in out and "NEXT: easel smooth" not in out, out
    assert Project(root).state["step"] == step
    run("stage", "done", "--because", "test")
    assert Project(root).state["stage"] == "detail"
    code, out = run("stroke", "20,500 980,520", "--size", "3", "--color", "#203040", "--layer", "detail")
    assert code == 0, out
    run("dot", "all") if cli.dots_available() else None
    run("stage", "done", "--because", "test")
    assert Project(root).state["stage"] == "finish"
    code, out = run("next")
    assert "smooth" not in out, out                              # the closed stage's blend is not offered again
    before = _layer_bytes(root, "detail")
    assert before is not None
    code, out = run("smooth", "all")                             # a bare blend in finish: never the detail layer
    assert code == 0 and "layer detail" not in out, out
    assert _layer_bytes(root, "detail") == before
    code, out = run("smooth", "all", "--layer", "detail")        # named explicitly, it still works
    assert code == 0 and "layer detail" in out, out
    assert _layer_bytes(root, "detail") != before


def test_shell_with_the_blend_in_shade_matches_the_pipeline(tmp_path, monkeypatch):
    """A cel session on a soft-shaded reference, each stage as the todo lists it (the blend once after the shade),
    detail level 2 on all, level 1 on every region, the dots, the sharpen: the pipeline's result on the same default."""
    from pixelcanvas.easel import pipeline
    from .test_shell_parity import R10
    size = (200, 160)
    _new(tmp_path, monkeypatch, size, BOXES)
    root = tmp_path / "pic.easel"
    for stage in workflow.plan_of(Project(root).state)[1:-1]:
        proj = Project(root)
        assert proj.state["stage"] == stage
        if stage == "detail":
            assert run("detail", "all")[0] == 0
            for name in BOXES:
                assert run("detail", name, "--level", "1")[0] == 0
            if cli.dots_available():
                assert run("dot", "all")[0] == 0
            assert run("stage", "done", "--because", "parity test")[0] == 0
            continue
        for name in BOXES:
            cmd = workflow.task_command(proj.state, stage, name)
            code, out = run(*cmd.split()[1:])
            assert code == 0, (cmd, out)
        if stage == "shade":
            st = Project(root).state
            assert [t["status"] for t in st["todo"] if t.get("key") == "blend"] == ["open"]
            code, out = run("smooth", "all")
            assert code == 0 and "(flat, shade)" in out, out
        run("stage", "done", "--because", "parity test")
    if not cli._streaks_available() or not cli.dots_available():
        pytest.skip("engine without streaks or dots")
    assert run("sharpen")[0] == 0
    assert run("export", "cli.png")[0] == 0
    proj = Project(root)
    ref_d = proj.ref_deliver
    masks = [proj.mask_to_deliver(proj.region_mask(n)) for n in BOXES]
    pipeline.run_reference_pipeline(ref_d, str(tmp_path / "pipe"), style="cel", log=lambda *a: None,
                                    regions=masks, **R10)
    cli_img = np.asarray(Image.open(tmp_path / "cli.png").convert("RGB"))
    pipe_img = np.asarray(Image.open(tmp_path / "pipe" / "final.png").convert("RGB"))
    assert cli_img.shape == pipe_img.shape == ref_d.shape
    s_cli, s_pipe = pipeline.quick_scores(ref_d, cli_img), pipeline.quick_scores(ref_d, pipe_img)
    print("PARITY blend", s_cli, s_pipe)
    assert abs(s_cli["mean_dE"] - s_pipe["mean_dE"]) <= 0.25 + 0.15 * s_pipe["mean_dE"], (s_cli, s_pipe)
    assert s_cli["edge_f1"] >= s_pipe["edge_f1"] - 0.02, (s_cli, s_pipe)
