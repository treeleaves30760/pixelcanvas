SCENE ?= mountains_sunset

.PHONY: draft check stages render report test list new install-skill

draft:            ## 草稿（半解析度）
	python tools/render.py $(SCENE) --scale 0.5

check:            ## 草稿 + 診斷圖（guides | values | edges | thumbnail）+ 自動警告
	python tools/render.py $(SCENE) --scale 0.5 --check

stages:           ## 草稿 + 逐步驟快照
	python tools/render.py $(SCENE) --scale 0.5 --stages

render:           ## 全尺寸
	python tools/render.py $(SCENE)

report:           ## 草稿的文字報告（給看不到圖的模型）
	python tools/look.py report outputs/$(SCENE)_x0.5.png

test:
	python -m pytest -q

list:
	python tools/render.py --list

new:              ## make new SCENE=night_lake
	python tools/new_scene.py $(SCENE)

install-skill:    ## 安裝 skill 到 ~/.claude/skills（make install-skill TOOL=codex）
	python tools/install_skill.py --tool $(or $(TOOL),claude)
