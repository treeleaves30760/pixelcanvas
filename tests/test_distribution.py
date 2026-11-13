# -*- coding: utf-8 -*-
"""Distribution: workspace mode, --deliver, the pixelcanvas command, doctor, the MCP server, plugin manifests,
install_skill targets and the skill's bootstrap."""
import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TOOLS = os.path.join(ROOT, "tools")

SCENE = '''
"""A tiny workspace scene."""
from pixelcanvas import Canvas, sdf
from scenes import zz_dist_helper as helper_pkg     # sibling module through the shared scenes package
import zz_dist_helper as helper_plain               # sibling module through sys.path

DEFAULT_SIZE = (160, 90)
KIND = "design"
COLOR = "#ff8040"


def render(width=160, height=90, seed=0, log=print):
    cv = Canvas(width, height)
    X, Y = cv.xy()
    log("background")
    cv.gradient([(0, helper_pkg.TOP), (1, helper_plain.BOTTOM)])
    log("disc")
    cv.paint(COLOR, sdf.fill(sdf.sd_circle(X, Y, width / 2, height / 2, height / 4)))
    return cv
'''
HELPER = 'TOP = "#102040"\nBOTTOM = "#406080"\n'


def _env(**extra):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for k in ("PIXELCANVAS_OUT", "PIXELCANVAS_WORK", "PIXELCANVAS_ROUNDS_DIR", "PIXELCANVAS_TRACE", "PIXELCANVAS_CMD"):
        env.pop(k, None)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _run(args, cwd, **env):
    return subprocess.run([sys.executable, *map(str, args)], cwd=str(cwd), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=_env(**env), timeout=300)


def _json_tail(stdout):
    return json.loads(stdout.strip().splitlines()[-1])


@pytest.fixture
def project(tmp_path):
    """A user project outside the repository with ./art/scene.py and a sibling helper module."""
    art = tmp_path / "art"
    art.mkdir()
    (art / "zz_dist_scene.py").write_text(SCENE, encoding="utf-8")
    (art / "zz_dist_helper.py").write_text(HELPER, encoding="utf-8")
    return tmp_path


# ------------------------------------------------------------------------------------------- workspace mode

def test_render_scene_by_path_from_any_folder(project):
    r = _run([os.path.join(TOOLS, "render.py"), "art/zz_dist_scene.py", "--check", "--json"], project)
    assert r.returncode == 0, r.stderr
    doc = _json_tail(r.stdout)
    assert doc["scene"] == "zz_dist_scene"
    png = project / "art" / "outputs" / "zz_dist_scene.png"
    assert png.exists() and (project / "art" / "outputs" / "zz_dist_scene_preview.jpg").exists()
    assert (project / "art" / "outputs" / "zz_dist_scene_check.jpg").exists()
    assert not (project / "art" / "__pycache__").exists()             # nothing but results in the user's folder
    assert "OBJECTS" in r.stdout


def test_bare_name_lookup_and_overrides(project):
    art = project / "art"
    r = _run([os.path.join(TOOLS, "render.py"), "zz_dist_scene", "--scale", "0.5"], art)          # ./NAME.py
    assert r.returncode == 0, r.stderr
    assert (art / "outputs" / "zz_dist_scene_x0.5.png").exists()
    r = _run([os.path.join(TOOLS, "render.py"), "zz_dist_scene"], project, PIXELCANVAS_WORK=art)  # $PIXELCANVAS_WORK
    assert r.returncode == 0, r.stderr
    r = _run([os.path.join(TOOLS, "render.py"), "art/zz_dist_scene.py", "--outdir", "elsewhere"], project)
    assert r.returncode == 0 and (project / "elsewhere" / "zz_dist_scene.png").exists(), r.stderr
    r = _run([os.path.join(TOOLS, "render.py"), "art/zz_dist_scene.py"], project, PIXELCANVAS_OUT=project / "env_out")
    assert r.returncode == 0 and (project / "env_out" / "zz_dist_scene.png").exists(), r.stderr
    r = _run([os.path.join(TOOLS, "render.py"), "zz_no_such_scene_anywhere"], project)
    assert r.returncode == 2 and "unknown scene" in r.stderr


def test_scenes_folder_writes_to_project_outputs(tmp_path):
    (tmp_path / "scenes").mkdir()
    (tmp_path / "scenes" / "zz_dist_scene.py").write_text(SCENE, encoding="utf-8")
    (tmp_path / "scenes" / "zz_dist_helper.py").write_text(HELPER, encoding="utf-8")
    r = _run([os.path.join(TOOLS, "render.py"), "zz_dist_scene", "--round"], tmp_path)             # ./scenes/NAME.py
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "outputs" / "zz_dist_scene.png").exists()
    assert (tmp_path / "outputs" / "rounds" / "zz_dist_scene" / "r01_preview.jpg").exists()
    r = _run([os.path.join(TOOLS, "look.py"), "rounds", "zz_dist_scene"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "outputs" / "rounds" / "zz_dist_scene" / "rounds_sheet.jpg").exists()


def test_new_scene_dir_then_render(tmp_path):
    r = _run([os.path.join(TOOLS, "new_scene.py"), "zz_dist_new", "--kind", "blank", "--dir", "art",
              "--size", "192x108"], tmp_path)
    assert r.returncode == 0, r.stderr
    path = tmp_path / "art" / "zz_dist_new.py"
    assert path.exists() and "DEFAULT_SIZE = (192, 108)" in path.read_text(encoding="utf-8")
    assert "art/zz_dist_new.py" in r.stdout and "render" in r.stdout
    assert _run([os.path.join(TOOLS, "new_scene.py"), "zz_dist_new", "--dir", "art"], tmp_path).returncode == 1
    r = _run([os.path.join(TOOLS, "render.py"), str(path)], tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "art" / "outputs" / "zz_dist_new.png").exists()


def test_styles_tool_with_scene_path(project):
    r = _run([os.path.join(TOOLS, "styles.py"), "apply", "art/zz_dist_scene.py", "pencil", "--size", "160x90"], project)
    if "invalid choice" in r.stderr:
        pytest.skip("pencil style not available")
    assert r.returncode == 0, r.stderr
    assert (project / "art" / "outputs" / "zz_dist_scene_pencil.png").exists()


# ------------------------------------------------------------------------------------------- --deliver

def test_deliver_folder_and_file(project):
    r = _run([os.path.join(TOOLS, "render.py"), "art/zz_dist_scene.py", "--deliver", "site/img/", "--json"], project)
    assert r.returncode == 0, r.stderr
    assert (project / "site" / "img" / "zz_dist_scene.png").exists()
    assert _json_tail(r.stdout)["outputs"]["delivered"]
    r = _run([os.path.join(TOOLS, "render.py"), "art/zz_dist_scene.py", "--layers", "--deliver", "dist/hero.png"],
             project)
    assert r.returncode == 0, r.stderr
    got = sorted(os.listdir(project / "dist"))
    assert "hero.png" in got and "hero.psd" in got and "hero.ora" in got, got


def test_deliver_animation(project):
    (project / "art" / "zz_dist_scene.py").write_text(SCENE.replace('COLOR = "#ff8040"', "T = 0.0\nCOLOR = \"#ff8040\""),
                                                      encoding="utf-8")
    r = _run([os.path.join(TOOLS, "render.py"), "art/zz_dist_scene.py", "--animate", "0", "1", "--frames", "2",
              "--deliver", "anim"], project)
    assert r.returncode == 0, r.stderr
    assert (project / "anim" / "zz_dist_scene.gif").exists()


def test_deliver_helper_names(tmp_path):
    from pixelcanvas import cli
    a, b = tmp_path / "x.png", tmp_path / "x_1.svg"
    a.write_bytes(b"png")
    b.write_bytes(b"svg")
    out = cli.deliver([str(a), str(b)], str(tmp_path / "out" / "final.png"), "x")
    assert sorted(os.path.basename(p) for p in out) == ["final.png", "final.svg"]


# ------------------------------------------------------------------------------------------- one command

def test_pc_dispatch(project):
    pc = os.path.join(TOOLS, "pc.py")
    r = _run([pc, "version"], project)
    assert r.returncode == 0 and re.search(r"pixelcanvas \d+\.\d+\.\d+", r.stdout), r.stderr
    r = _run([pc, "new", "zz_dist_pc", "--kind", "blank", "--dir", "."], project)
    assert r.returncode == 0, r.stderr
    assert "render zz_dist_pc.py" in r.stdout and "pc.py" in r.stdout                  # next: uses the same entry
    r = _run([pc, "render", "zz_dist_pc", "--scale", "0.1"], project)
    assert r.returncode == 0, r.stderr
    assert (project / "outputs" / "zz_dist_pc_x0.1.png").exists()
    r = _run([pc, "look", "stats", "outputs/zz_dist_pc_x0.1.png", "--json"], project)
    assert r.returncode == 0 and json.loads(r.stdout.strip().splitlines()[-1]), r.stderr
    r = _run([pc, "new", "--list"], project)
    assert r.returncode == 0 and "poster" in r.stdout
    r = _run([pc, "no-such-tool"], project)
    assert r.returncode == 2 and "unknown command" in r.stderr
    r = _run([pc], project)
    assert r.returncode == 1 and "render" in r.stdout


def test_tools_are_discovered(tmp_path, capsys):
    from pixelcanvas import cli
    (tmp_path / "tools").mkdir()
    (tmp_path / "scenes").mkdir()
    (tmp_path / "tools" / "render.py").write_text("", encoding="utf-8")
    (tmp_path / "tools" / "zz_extra.py").write_text(
        "import sys\nprint('extra', sys.argv[1:])\nsys.exit(3)\n", encoding="utf-8")
    assert {"render", "zz_extra"} <= set(cli.available_tools(str(tmp_path)))
    assert cli.run_tool("zz_extra", ["a", "b"], str(tmp_path)) == 3
    assert "extra ['a', 'b']" in capsys.readouterr().out
    assert cli.tool_path("new", ROOT).endswith("new_scene.py")


def test_pc_match(tmp_path):
    from pixelcanvas import Canvas, cli
    assert "match" in cli.available_tools(ROOT) and "match REF.png" in cli._usage()
    ref = tmp_path / "ref.png"
    Canvas(96, 54).gradient([(0, "#1d2150"), (1, "#f29a72")]).save(str(ref))
    r = _run([os.path.join(TOOLS, "pc.py"), "match", "ref.png"], tmp_path)
    assert r.returncode == 0 and r.stdout.strip(), r.stderr


def test_doctor_json(tmp_path):
    r = _run([os.path.join(TOOLS, "pc.py"), "doctor", "--json"], tmp_path)
    doc = json.loads(r.stdout.strip().splitlines()[-1])
    names = {c["name"] for c in doc["checks"]}
    assert {"python", "numpy", "Pillow", "scipy", "runtime", "fonts", "CJK fonts", "write access", "vision model",
            "self-test render"} <= names
    assert doc["ok"] is (r.returncode == 0)
    selftest = next(c for c in doc["checks"] if c["name"] == "self-test render")
    assert selftest["status"] in ("ok", "warn"), selftest


def test_cli_info_preview(tmp_path):
    from pixelcanvas import Canvas, cli
    p = tmp_path / "a.png"
    Canvas(64, 36).gradient([(0, "#000000"), (1, "#ffffff")]).save(str(p))
    assert cli.main(["info", str(p), "--json"]) == 0
    assert cli.main(["preview", str(p), str(tmp_path / "s.jpg"), "32"]) == 0
    assert (tmp_path / "s.jpg").exists()


# ------------------------------------------------------------------------------------------- MCP server

class Client:
    def __init__(self, cwd):
        env = _env(PYTHONPATH=ROOT)
        self.p = subprocess.Popen([sys.executable, "-m", "pixelcanvas.mcp_server"], cwd=str(cwd), env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.n = 0

    def send(self, obj):
        self.p.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.p.stdin.flush()

    def call(self, method, params=None):
        self.n += 1
        msg = {"jsonrpc": "2.0", "id": self.n, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        line = self.p.stdout.readline()
        assert line, self.p.stderr.read().decode("utf-8", "replace")
        reply = json.loads(line)
        assert reply["id"] == self.n
        return reply

    def tool(self, _tool, **args):
        r = self.call("tools/call", {"name": _tool, "arguments": args})
        return r["result"]

    def close(self):
        self.p.stdin.close()
        rest = self.p.stdout.read()
        self.p.wait(timeout=30)
        self.p.stderr.close()
        self.p.stdout.close()
        return rest


def test_mcp_round_trip(tmp_path):
    c = Client(tmp_path)
    try:
        r = c.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                  "clientInfo": {"name": "pytest", "version": "1"}})
        assert r["result"]["protocolVersion"] == "2025-06-18"
        assert r["result"]["serverInfo"]["name"] == "pixelcanvas" and "tools" in r["result"]["capabilities"]
        c.send({"jsonrpc": "2.0", "method": "notifications/initialized"})           # no reply expected
        assert c.call("ping")["result"] == {}
        tools = {t["name"]: t for t in c.call("tools/list")["result"]["tools"]}
        assert {"kinds", "new_scene", "write_scene", "read_scene", "render", "look", "find", "reference", "ask",
                "animate", "layers"} <= set(tools)
        for t in tools.values():
            assert t["description"] and t["inputSchema"]["type"] == "object"
            assert set(t["inputSchema"]["required"]) <= set(t["inputSchema"]["properties"])

        assert "poster" in c.tool("kinds")["content"][0]["text"]
        res = c.tool("new_scene", name="zz_mcp", kind="blank", size="320x180", dir="art")
        assert not res["isError"], res
        assert (tmp_path / "art" / "zz_mcp.py").exists() and "def render" in res["content"][0]["text"]
        src = (tmp_path / "art" / "zz_mcp.py").read_text(encoding="utf-8").replace('"#ffb347"', '"#40c0ff"')
        res = c.tool("write_scene", name="zz_mcp", code=src, dir="art")
        assert not res["isError"], res
        assert "#40c0ff" in c.tool("read_scene", name="art/zz_mcp.py")["content"][0]["text"]

        res = c.tool("render", scene="art/zz_mcp.py", scale=0.5)
        assert not res["isError"], res["content"][0]["text"]
        kinds = [b["type"] for b in res["content"]]
        assert kinds[0] == "text" and "image" in kinds
        assert "WARNINGS" in res["content"][0]["text"] and "OBJECTS" in res["content"][0]["text"]
        img = next(b for b in res["content"] if b["type"] == "image")
        assert img["mimeType"] == "image/jpeg" and len(img["data"]) > 100
        png = tmp_path / "art" / "outputs" / "zz_mcp_x0.5.png"
        assert png.exists()

        res = c.tool("look", image="art/outputs/zz_mcp_x0.5.png", view="guides")
        assert not res["isError"] and any(b["type"] == "image" for b in res["content"]), res
        res = c.tool("look", image="art/outputs/zz_mcp_x0.5.png", view="probe", args=["0.5,0.5"])
        assert not res["isError"] and "#" in res["content"][0]["text"]
        res = c.tool("look", image="art/outputs/zz_mcp_x0.5.png", view="zoom", args=["0.3", "0.3", "0.7", "0.7"])
        assert not res["isError"] and any(b["type"] == "image" for b in res["content"])

        # readable errors, never a crash
        res = c.tool("render", scene="zz_nothing_here")
        assert res["isError"] and "not found" in res["content"][0]["text"]
        (tmp_path / "zz_broken.py").write_text("def render(width=10, height=10, seed=0, log=print):\n    return 1 / 0\n",
                                               encoding="utf-8")
        res = c.tool("render", scene="zz_broken.py")
        assert res["isError"] and "ZeroDivisionError" in res["content"][0]["text"]
        res = c.tool("write_scene", name="zz_bad", code="def render(:\n")
        assert res["isError"] and "does not compile" in res["content"][0]["text"]
        assert c.tool("new_scene", name="Bad Name")["isError"]
        assert c.call("tools/call", {"name": "nope", "arguments": {}})["error"]["code"] == -32602

        assert c.tool("find", query="gradient")["content"][0]["text"]
        assert "pixelcanvas API" in c.tool("reference", page="api")["content"][0]["text"]
        assert c.tool("reference", page="zz_missing")["isError"]
        res = c.tool("layers", scene="art/zz_mcp.py", scale=0.5)
        assert not res["isError"] and "subject" in res["content"][0]["text"], res

        resources = c.call("resources/list")["result"]["resources"]
        uris = {r["uri"] for r in resources}
        assert "pixelcanvas://skill/SKILL.md" in uris and "pixelcanvas://references/api.md" in uris
        doc = c.call("resources/read", {"uri": "pixelcanvas://skill/SKILL.md"})["result"]["contents"][0]
        assert doc["mimeType"] == "text/markdown" and "name: pixelcanvas" in doc["text"]
        assert "error" in c.call("resources/read", {"uri": "pixelcanvas://references/nope.md"})
        assert c.call("no/such/method")["error"]["code"] == -32601

        # modern (per-request _meta) clients
        meta = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {}}
        d = c.call("server/discover", {"_meta": meta})["result"]
        assert "2026-07-28" in d["supportedVersions"] and d["resultType"] == "complete"
        assert d["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "pixelcanvas"
        assert c.call("tools/list", {"_meta": meta})["result"]["tools"]
        bad = c.call("tools/list", {"_meta": {"io.modelcontextprotocol/protocolVersion": "1900-01-01"}})
        assert bad["error"]["code"] == -32022 and "2026-07-28" in bad["error"]["data"]["supported"]

        c.p.stdin.write(b"this is not json\n")
        c.p.stdin.flush()
        assert json.loads(c.p.stdout.readline())["error"]["code"] == -32700
    finally:
        rest = c.close()
    assert rest == b""                                                    # nothing but protocol messages on stdout


def test_mcp_unknown_legacy_version_is_tolerated(tmp_path):
    c = Client(tmp_path)
    try:
        r = c.call("initialize", {"protocolVersion": "2099-01-01", "capabilities": {}})
        assert r["result"]["protocolVersion"] in ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
    finally:
        c.close()


# ------------------------------------------------------------------------------------------- plugin + versions

def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def test_plugin_manifests_and_versions():
    plugin = json.loads(_read(".claude-plugin", "plugin.json"))
    market = json.loads(_read(".claude-plugin", "marketplace.json"))
    assert plugin["name"] == "pixelcanvas" and re.fullmatch(r"[a-z0-9-]+", plugin["name"])
    assert market["name"] and market["owner"]["name"] and market["plugins"]
    entry = market["plugins"][0]
    assert entry["name"] == plugin["name"]                       # entry name must equal the manifest name
    assert entry["source"] == "./" and ".." not in entry["source"]
    assert os.path.isfile(os.path.join(ROOT, "skills", "pixelcanvas", "SKILL.md"))   # default skills/ folder

    import pixelcanvas
    version = re.search(r'^version = "([^"]+)"', _read("pyproject.toml"), re.M).group(1)
    boot = re.search(r'^VERSION = "([^"]+)"', _read("skills", "pixelcanvas", "scripts", "bootstrap.py"), re.M).group(1)
    assert version == pixelcanvas.__version__ == plugin["version"] == boot


def test_pyproject_ships_the_runtime():
    text = _read("pyproject.toml")
    for pkg, folder in (("tools", "tools"), ("scenes", "scenes"), ("skills", "skills")):
        assert f'"pixelcanvas._runtime.{pkg}" = "{folder}"' in text
    assert 'pixelcanvas = "pixelcanvas.cli:main"' in text and 'fast = ["scipy' in text


# ------------------------------------------------------------------------------------------- install_skill

_INSTALLER = []


def _installer():
    """tools/install_skill.py loaded once as a module: the tests call main() in-process (no interpreter start)."""
    if not _INSTALLER:
        import importlib.util
        spec = importlib.util.spec_from_file_location("install_skill_under_test", os.path.join(TOOLS, "install_skill.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _INSTALLER.append(mod)
    return _INSTALLER[0]


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """A temporary home for the installer: no test ever reads or writes the real ~/.claude, ~/.pixelcanvas ..."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("PIXELCANVAS_HOME", raising=False)
    assert os.path.expanduser("~") == str(home)
    return home


def _install(*argv, cwd=None, monkeypatch=None):
    if cwd is not None:
        monkeypatch.chdir(cwd)
    return _installer().main([str(a) for a in argv])


TARGET_DIRS = {"claude": ".claude/skills", "agents": ".agents/skills", "opencode": ".config/opencode/skills",
               "cursor": ".cursor/skills", "codex": ".codex/skills"}
PROJECT_DIRS = {"claude": ".claude/skills", "agents": ".agents/skills", "opencode": ".opencode/skills",
                "cursor": ".cursor/skills", "codex": ".codex/skills"}


def test_install_skill_targets(tmp_path, fake_home, monkeypatch, capsys):
    inst = _installer()
    assert set(inst.TARGETS) == set(TARGET_DIRS)                  # a new target needs a line in this test
    for target, rel in TARGET_DIRS.items():
        assert _install(target, cwd=tmp_path, monkeypatch=monkeypatch) == 0
        skill = fake_home / rel / "pixelcanvas" / "SKILL.md"
        assert skill.exists(), target
        assert ROOT in skill.read_text(encoding="utf-8")
        assert (fake_home / rel / "pixelcanvas" / "references" / "api.md").exists()
    capsys.readouterr()
    assert _install("--list", cwd=tmp_path, monkeypatch=monkeypatch) == 0
    assert capsys.readouterr().out.count("installed") >= 5
    proj = tmp_path / "proj"
    proj.mkdir()
    assert _install("agents", "--project", cwd=proj, monkeypatch=monkeypatch) == 0
    assert (proj / ".agents" / "skills" / "pixelcanvas" / "SKILL.md").exists()
    # a neighbouring skill survives install, reinstall and uninstall
    other = fake_home / ".claude" / "skills" / "other-skill"
    other.mkdir()
    (other / "SKILL.md").write_text("---\nname: other-skill\n---\n", encoding="utf-8")
    assert _install("claude", cwd=tmp_path, monkeypatch=monkeypatch) == 0
    assert _install("claude", "--uninstall", cwd=tmp_path, monkeypatch=monkeypatch) == 0
    assert not (fake_home / ".claude" / "skills" / "pixelcanvas").exists() and (other / "SKILL.md").exists()
    # a foreign folder that happens to be called pixelcanvas is never deleted
    foreign = tmp_path / "dest" / "pixelcanvas"
    foreign.mkdir(parents=True)
    (foreign / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        _install("--dest", tmp_path / "dest", cwd=tmp_path, monkeypatch=monkeypatch)
    assert e.value.code != 0 and (foreign / "notes.txt").exists()


def _check_lite(folder):
    """What an agent that opens ``folder/pixelcanvas-lite`` needs: the page, its bootstrap and the compose shim."""
    lite = folder / "pixelcanvas-lite"
    text = (lite / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\nname: pixelcanvas-lite\n") and "ROOT/tools/" not in text
    assert ROOT.replace("\\", "/") + "/tools/compose.py" in text
    assert (lite / "scripts" / "bootstrap.py").is_file() and (lite / "tools" / "compose.py").is_file()
    assert (lite / "scripts" / "ROOT.txt").read_text(encoding="utf-8").strip() == ROOT
    assert (lite / "tools" / "ROOT.txt").read_text(encoding="utf-8").strip() == ROOT
    assert not list(lite.rglob("__pycache__"))


@pytest.mark.parametrize("target", sorted(TARGET_DIRS))
def test_install_skill_lite_and_both_for_every_target(target, tmp_path, fake_home, monkeypatch):
    skills = fake_home / TARGET_DIRS[target]
    assert _install(target, "--skill", "pixelcanvas-lite", cwd=tmp_path, monkeypatch=monkeypatch) == 0
    _check_lite(skills)
    assert not (skills / "pixelcanvas").exists()                          # only the lite skill was written
    assert _install(target, "--skill", "both", cwd=tmp_path, monkeypatch=monkeypatch) == 0
    _check_lite(skills)
    assert (skills / "pixelcanvas" / "SKILL.md").exists() and (skills / "pixelcanvas" / "references" / "api.md").exists()
    assert (skills / "pixelcanvas" / "scripts" / "ROOT.txt").read_text(encoding="utf-8").strip() == ROOT
    assert _install(target, "--uninstall", "--skill", "pixelcanvas", cwd=tmp_path, monkeypatch=monkeypatch) == 0
    assert not (skills / "pixelcanvas").exists() and (skills / "pixelcanvas-lite" / "SKILL.md").exists()
    proj = tmp_path / "proj"                                              # project scope
    proj.mkdir()
    assert _install(target, "--project", "--skill", "pixelcanvas-lite", cwd=proj, monkeypatch=monkeypatch) == 0
    _check_lite(proj / PROJECT_DIRS[target])
    assert not (proj / PROJECT_DIRS[target] / "pixelcanvas").exists()


def test_install_skill_dest_and_unknown_skill(tmp_path, fake_home, monkeypatch):
    dest = tmp_path / "anywhere"
    assert _install("--dest", dest, "--skill", "pixelcanvas-lite", cwd=tmp_path, monkeypatch=monkeypatch) == 0
    _check_lite(dest)
    assert _install("--dest", dest, cwd=tmp_path, monkeypatch=monkeypatch) == 0
    assert (dest / "pixelcanvas" / "SKILL.md").exists() and (dest / "pixelcanvas-lite" / "SKILL.md").exists()
    with pytest.raises(SystemExit) as e:
        _install("opencode", "--skill", "nope", cwd=tmp_path, monkeypatch=monkeypatch)
    assert e.value.code != 0
    assert list(fake_home.iterdir()) == []                                # nothing landed in the fake home


def test_installed_lite_shim_and_lone_bootstrap_find_the_runtime(tmp_path, fake_home, monkeypatch):
    """Only the lite skill is installed: its compose.py shim and its bootstrap.py both resolve the runtime that
    installed it (scripts/ROOT.txt), from any folder, with nothing downloaded."""
    assert _install("opencode", "--skill", "pixelcanvas-lite", cwd=tmp_path, monkeypatch=monkeypatch) == 0
    lite = fake_home / ".config" / "opencode" / "skills" / "pixelcanvas-lite"
    outside = tmp_path / "outside"
    outside.mkdir()
    env = {"HOME": fake_home, "USERPROFILE": fake_home, "PIXELCANVAS_HOME": ""}
    r = _run([lite / "scripts" / "bootstrap.py", "--home", tmp_path / "unused"], outside, **env)
    assert r.returncode == 0, r.stderr
    assert os.path.samefile(r.stdout.strip().splitlines()[-1], ROOT) and not (tmp_path / "unused").exists()
    r = _run([lite / "tools" / "compose.py", "--kinds"], outside, **env)       # the path a small model guesses
    assert r.returncode == 0 and "poster" in r.stdout, r.stderr


def test_lite_bootstrap_inside_the_runtime_skills_folder(tmp_path, monkeypatch):
    """The runtime holds both skills: the lite bootstrap in <runtime>/skills/pixelcanvas-lite finds <runtime>."""
    mod, repo = _boot(tmp_path, monkeypatch, repo_version="0.4.0", skill_name="pixelcanvas-lite")
    nohome = str(tmp_path / "nohome")
    assert [c[0] for c in mod.candidates(nohome)] == ["repository"]
    assert os.path.samefile(mod.candidates(nohome)[0][1], repo)
    rec = _fake_runtime(tmp_path / "rec", "0.4.0")
    with open(os.path.join(repo, "skills", "pixelcanvas-lite", "scripts", "ROOT.txt"), "w", encoding="utf-8") as f:
        f.write(rec + "\n")                                               # what install_skill.py records
    assert [c[0] for c in mod.candidates(nohome)] == ["installer", "repository"]


def test_runtime_download_carries_both_skills():
    sys.path.insert(0, os.path.join(ROOT, "skills", "pixelcanvas", "scripts"))
    import bootstrap
    assert "skills/pixelcanvas" in bootstrap.SPARSE and "skills/pixelcanvas-lite" in bootstrap.SPARSE
    for p in ("skills/pixelcanvas-lite/SKILL.md", "skills/pixelcanvas-lite/scripts/bootstrap.py",
              "skills/pixelcanvas-lite/tools/compose.py", "skills/pixelcanvas/references/api.md"):
        assert bootstrap._wanted(p), p
        assert os.path.isfile(os.path.join(ROOT, *p.split("/"))), p
    assert not bootstrap._wanted("skills/pixelcanvas-lite-extra/SKILL.md") and not bootstrap._wanted("gallery/a.png")


def test_lite_skill_is_short_and_matches_bootstrap():
    text = _read("skills", "pixelcanvas-lite", "SKILL.md")
    assert text.startswith("---\nname: pixelcanvas-lite\n") and len(text.splitlines()) <= 70
    assert _read("skills", "pixelcanvas-lite", "scripts", "bootstrap.py") == \
        _read("skills", "pixelcanvas", "scripts", "bootstrap.py")


# ------------------------------------------------------------------------------------------- bootstrap

def test_bootstrap_inside_checkout(tmp_path):
    boot = os.path.join(ROOT, "skills", "pixelcanvas", "scripts", "bootstrap.py")
    r = _run([boot], os.path.join(ROOT, "tools"))                          # a sub-folder of the checkout
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert os.path.samefile(lines[-1], ROOT) and lines[-2].startswith("pixelcanvas ")
    r = _run([boot, "--home", tmp_path / "unused"], tmp_path, PIXELCANVAS_HOME="")   # the repo the skill sits in
    assert r.returncode == 0, r.stderr
    assert os.path.samefile(r.stdout.strip().splitlines()[-1], ROOT)
    assert not (tmp_path / "unused").exists()                               # nothing downloaded
    r = _run([boot, "--update"], ROOT)                                       # never touches a user's checkout
    assert r.returncode == 0 and "update skipped" in r.stderr


def test_bootstrap_archive_filter():
    sys.path.insert(0, os.path.join(ROOT, "skills", "pixelcanvas", "scripts"))
    import bootstrap
    keep = ["pixelcanvas/canvas.py", "tools/render.py", "scenes/_template_poster.py", "requirements.txt",
            "skills/pixelcanvas/references/api.md", "skills/pixelcanvas/SKILL.md"]
    drop = ["gallery/hero.png", "docs/hero.jpg", "tests/test_x.py", "skills/other/SKILL.md"]
    assert all(bootstrap._wanted(p) for p in keep) and not any(bootstrap._wanted(p) for p in drop)
    assert len(_read("skills", "pixelcanvas", "scripts", "bootstrap.py").splitlines()) <= 320


# ------------------------------------------------------------------------------------------- bootstrap versions

def _fake_runtime(path, version, scenes=True):
    path = os.fspath(path)
    os.makedirs(os.path.join(path, "pixelcanvas"), exist_ok=True)
    os.makedirs(os.path.join(path, "tools"), exist_ok=True)
    with open(os.path.join(path, "pixelcanvas", "__init__.py"), "w", encoding="utf-8") as f:
        f.write(f'__version__ = "{version}"\n')
    with open(os.path.join(path, "tools", "render.py"), "w", encoding="utf-8") as f:
        f.write("# render\n")
    if scenes:
        os.makedirs(os.path.join(path, "scenes"), exist_ok=True)
    return os.path.abspath(path)


def _boot(tmp_path, monkeypatch, skill="0.4.0", repo_version=None, name="b", skill_name="pixelcanvas"):
    """A copy of bootstrap.py in a fake skill folder (inside a complete fake repository when ``repo_version``)."""
    import importlib.util
    base = tmp_path / name
    skill_dir = base / "skills" / skill_name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f'---\nname: {skill_name}\nmetadata:\n  version: "{skill}"\n---\n',
                                        encoding="utf-8")
    dst = skill_dir / "scripts" / "bootstrap.py"
    dst.write_text(_read("skills", "pixelcanvas", "scripts", "bootstrap.py"), encoding="utf-8")
    if repo_version:
        _fake_runtime(base, repo_version)
    spec = importlib.util.spec_from_file_location(f"bootstrap_{name}", dst)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ensure_deps", lambda: ["1.0", "2.0"])
    monkeypatch.setattr(mod, "installed_runtime", lambda: None)
    monkeypatch.delenv("PIXELCANVAS_HOME", raising=False)
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    monkeypatch.chdir(outside)
    return mod, os.path.abspath(base)


def _run_main(mod, capsys, *argv):
    assert mod.main([str(a) for a in argv]) == 0
    cap = capsys.readouterr()
    return cap.out.strip().splitlines(), cap.err


def test_bootstrap_prefers_the_repository_it_sits_in_over_a_stale_home(tmp_path, monkeypatch, capsys):
    mod, repo = _boot(tmp_path, monkeypatch, repo_version="0.4.0")
    home = _fake_runtime(tmp_path / "home" / ".pixelcanvas", "0.3.0")
    monkeypatch.setattr(mod, "update", lambda root: (_ for _ in ()).throw(AssertionError("must not touch home")))
    out, err = _run_main(mod, capsys, "--home", home)
    assert os.path.samefile(out[-1], repo) and out[-2].startswith("pixelcanvas 0.4.0")
    assert "0.3.0" not in err


def test_bootstrap_candidate_order(tmp_path, monkeypatch, capsys):
    mod, repo = _boot(tmp_path, monkeypatch, repo_version="0.4.0")
    home = _fake_runtime(tmp_path / "h", "0.4.0")
    env = _fake_runtime(tmp_path / "env", "0.4.1")
    cwd = _fake_runtime(tmp_path / "work", "0.5.0")
    assert [c[0] for c in mod.candidates(home)] == ["repository", "home"]
    monkeypatch.setenv("PIXELCANVAS_HOME", env)
    assert [c[0] for c in mod.candidates(home)] == ["$PIXELCANVAS_HOME", "repository", "home"]
    os.makedirs(os.path.join(cwd, "sub"))
    monkeypatch.chdir(os.path.join(cwd, "sub"))
    assert [c[0] for c in mod.candidates(home)] == ["checkout", "$PIXELCANVAS_HOME", "repository", "home"]
    out, _ = _run_main(mod, capsys, "--home", home)
    assert os.path.samefile(out[-1], cwd)
    mod2, repo2 = _boot(tmp_path, monkeypatch, name="c")                  # a repository without scenes/ is no candidate
    _fake_runtime(repo2, "0.4.0", scenes=False)
    monkeypatch.chdir(tmp_path / "outside")
    assert [c[0] for c in mod2.candidates(home)] == ["home"]


def test_bootstrap_updates_a_stale_download_in_place(tmp_path, monkeypatch, capsys):
    mod, _ = _boot(tmp_path, monkeypatch)                           # a skill alone: no repository around it
    home = _fake_runtime(tmp_path / "home" / ".pixelcanvas", "0.3.0")
    calls = []

    def fake_update(root):
        calls.append(root)
        _fake_runtime(root, "0.4.0")
        return True
    monkeypatch.setattr(mod, "update", fake_update)
    out, err = _run_main(mod, capsys, "--home", home)
    assert calls == [home] and os.path.samefile(out[-1], home) and out[-2].startswith("pixelcanvas 0.4.0")
    assert err.count("\n") == 1 and "updated" in err and "0.3.0" in err and "0.4.0" in err


def test_bootstrap_without_network_keeps_the_newest_runtime_and_says_so(tmp_path, monkeypatch, capsys):
    mod, _ = _boot(tmp_path, monkeypatch)
    home = _fake_runtime(tmp_path / "home" / ".pixelcanvas", "0.3.0")
    monkeypatch.setenv("PIXELCANVAS_HOME", _fake_runtime(tmp_path / "env", "0.2.0"))
    (tmp_path / "env" / ".pixelcanvas-managed").write_text("x", encoding="utf-8")
    monkeypatch.setattr(mod, "update", lambda root: False)
    out, err = _run_main(mod, capsys, "--home", home)
    assert os.path.samefile(out[-1], home) and out[-2].startswith("pixelcanvas 0.3.0")
    assert "no network" in err and "0.3.0" in err and "0.4.0" in err


def test_bootstrap_keeps_a_users_own_checkout_with_a_warning(tmp_path, monkeypatch, capsys):
    mod, _ = _boot(tmp_path, monkeypatch, repo_version="0.4.0")
    mine = _fake_runtime(tmp_path / "mine", "0.3.5")
    monkeypatch.chdir(mine)
    monkeypatch.setattr(mod, "update", lambda root: (_ for _ in ()).throw(AssertionError("own checkout")))
    out, err = _run_main(mod, capsys, "--home", tmp_path / "nohome")
    assert os.path.samefile(out[-1], mine)
    assert err.count("\n") == 1 and err.startswith("warning:") and "0.3.5" in err and "0.4.0" in err
    newer = _fake_runtime(tmp_path / "newer", "0.9.0")                    # newer than the skill: silent
    monkeypatch.chdir(newer)
    out, err = _run_main(mod, capsys, "--home", tmp_path / "nohome")
    assert err == "" and os.path.samefile(out[-1], newer)


def test_bootstrap_env_home_managed_or_not(tmp_path, monkeypatch, capsys):
    mod, _ = _boot(tmp_path, monkeypatch)
    env = _fake_runtime(tmp_path / "env", "0.3.0")
    monkeypatch.setenv("PIXELCANVAS_HOME", env)
    monkeypatch.setattr(mod, "update", lambda root: (_ for _ in ()).throw(AssertionError("not managed")))
    out, err = _run_main(mod, capsys, "--home", tmp_path / "nohome")
    assert os.path.samefile(out[-1], env) and err.startswith("warning:")
    (tmp_path / "env" / ".pixelcanvas-managed").write_text("x", encoding="utf-8")   # a download made by the script
    monkeypatch.setattr(mod, "update", lambda root: (_fake_runtime(root, "0.4.0"), True)[1])
    out, err = _run_main(mod, capsys, "--home", tmp_path / "nohome")
    assert out[-2].startswith("pixelcanvas 0.4.0") and "updated" in err


def test_bootstrap_downloads_only_when_nothing_is_found(tmp_path, monkeypatch, capsys):
    mod, _ = _boot(tmp_path, monkeypatch)
    home = tmp_path / "fresh" / ".pixelcanvas"

    def fake_download(dest):
        _fake_runtime(dest, "0.4.0")
        mod.log(f"downloaded pixelcanvas (v0.4.0) to {dest}")
    monkeypatch.setattr(mod, "download", fake_download)
    out, err = _run_main(mod, capsys, "--home", home)
    assert os.path.samefile(out[-1], home) and "downloaded" in err
    monkeypatch.setattr(mod, "download", lambda dest: (_ for _ in ()).throw(mod.BootstrapError("download failed")))
    with pytest.raises(SystemExit):
        mod.main(["--home", str(tmp_path / "other")])


def test_bootstrap_update_switches_a_plain_download_and_restores_on_failure(tmp_path, monkeypatch):
    mod, _ = _boot(tmp_path, monkeypatch)
    root = _fake_runtime(tmp_path / "dl", "0.3.0")
    monkeypatch.setattr(mod, "download", lambda dest: _fake_runtime(dest, "0.4.0"))
    assert mod.update(root) is True and mod.version_of(root) == "0.4.0" and not os.path.exists(root + ".old")
    root = _fake_runtime(tmp_path / "dl2", "0.3.0")
    monkeypatch.setattr(mod, "download", lambda dest: (_ for _ in ()).throw(mod.BootstrapError("offline")))
    assert mod.update(root) is False and mod.version_of(root) == "0.3.0"


def test_bootstrap_update_of_a_git_download_fetches_the_skills_tag(tmp_path, monkeypatch):
    mod, _ = _boot(tmp_path, monkeypatch, skill="0.4.0")
    root = _fake_runtime(tmp_path / "git", "0.3.0")
    os.makedirs(os.path.join(root, ".git"))
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        ok = not ("fetch" in cmd and "refs/tags/v0.4.0" in cmd)            # the tag is missing: default branch
        return type("R", (), {"returncode": 0 if ok else 1, "stderr": "", "stdout": ""})()
    monkeypatch.setattr(mod, "run", fake_run)
    assert mod.update(root) is True
    fetches = [c for c in seen if "fetch" in c]
    assert fetches[0][-1] == "refs/tags/v0.4.0" and fetches[1][-1] == "HEAD"
    assert any("checkout" in c and "FETCH_HEAD" in c for c in seen)
    assert any("sparse-checkout" in c and "skills/pixelcanvas-lite" in c for c in seen)   # both skills stay in
    assert mod.REFS[0] == "v0.4.0" and mod.skill_version() == "0.4.0"


def test_bootstrap_reads_versions_without_importing(tmp_path, monkeypatch):
    mod, _ = _boot(tmp_path, monkeypatch, skill="1.2.3")
    assert mod.skill_version() == "1.2.3"
    assert mod.vkey("0.10.0") > mod.vkey("0.9.9") and mod.vkey("?") < mod.vkey("0.0.1")
    rt = _fake_runtime(tmp_path / "r", "0.7.1")
    assert mod.version_of(rt) == "0.7.1" and mod.version_of(tmp_path / "none") == "?"
