SCENE ?= mountains_sunset

.PHONY: draft check stages render report test list new install-skill release-check

draft:            ## draft (half resolution)
	python tools/render.py $(SCENE) --scale 0.5

check:            ## draft + diagnostic images (guides | values | edges | thumbnail) + automatic warnings
	python tools/render.py $(SCENE) --scale 0.5 --check

stages:           ## draft + per-step snapshots
	python tools/render.py $(SCENE) --scale 0.5 --stages

render:           ## full size
	python tools/render.py $(SCENE)

report:           ## text report of the draft (for models that cannot see images)
	python tools/look.py report outputs/$(SCENE)_x0.5.png

test:
	python -m pytest -q

list:
	python tools/render.py --list

new:              ## make new SCENE=night_lake
	python tools/new_scene.py $(SCENE)

install-skill:    ## install the skill to ~/.claude/skills (make install-skill TOOL=codex)
	python tools/install_skill.py --tool $(or $(TOOL),claude)

release-check:    ## build sdist + wheel, twine check, install in a clean venv and smoke test the command
	python tools/release_check.py
