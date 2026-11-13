"""README.md / README-zh.md and gallery/: every link resolves, the gallery stays small and matches its build list."""
import importlib.util
import os
import re

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READMES = ("README.md", "README-zh.md")
GALLERY = os.path.join(ROOT, "gallery")


def _read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def _links(text):
    return [t for t in re.findall(r"\]\(([^)\s]+)\)", text) if not re.match(r"[a-z]+:", t)]


def _build_list():
    spec = importlib.util.spec_from_file_location("build_gallery", os.path.join(ROOT, "tools", "build_gallery.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_local_link_resolves():
    for name in READMES:
        for link in _links(_read(name)):
            path = link.split("#")[0]
            assert os.path.exists(os.path.join(ROOT, path)), f"{name}: {link} does not exist"


def test_readmes_show_the_same_pictures():
    imgs = [sorted(re.findall(r"\((gallery/[^)]+\.jpg)\)", _read(n))) for n in READMES]
    assert imgs[0] == imgs[1]


def test_gallery_is_small_jpeg_and_matches_the_build_list():
    files = [f for f in os.listdir(GALLERY) if os.path.isfile(os.path.join(GALLERY, f))]
    assert not [f for f in files if f.lower().endswith(".png")]
    assert sum(os.path.getsize(os.path.join(GALLERY, f)) for f in files) < 6_000_000
    mod = _build_list()
    wanted = {name + ".jpg" for name, _, _ in mod.PICTURES} | {"wall.jpg"}
    assert set(files) == wanted
    for row in mod.WALL:
        assert all(n + ".jpg" in wanted for n in row)
    readme = _read("README.md")
    for f in wanted:
        assert f"gallery/{f}" in readme, f"{f} is built but not shown in README.md"


def test_wall_size():
    path = os.path.join(GALLERY, "wall.jpg")
    assert os.path.getsize(path) < 900_000
    assert Image.open(path).width == 2400


def test_readme_code_sample_is_the_gallery_scene():
    scene = _read(os.path.join("gallery", "src", "harbour_light.py")).strip()
    for name in READMES:
        blocks = re.findall(r"```python\n(.*?)```", _read(name), re.S)
        assert scene in [b.strip() for b in blocks], f"{name}: the code sample differs from gallery/src/harbour_light.py"
