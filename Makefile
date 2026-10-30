SCENE ?= mountains_sunset

.PHONY: draft render test list new

draft:            ## 草稿（半解析度）
	python tools/render.py $(SCENE) --scale 0.5

render:           ## 全尺寸
	python tools/render.py $(SCENE)

test:
	python -m pytest -q

list:
	python tools/render.py --list

new:              ## make new SCENE=night_lake
	python tools/new_scene.py $(SCENE)
