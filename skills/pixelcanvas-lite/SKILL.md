---
name: pixelcanvas-lite
description: >-
  Draw finished pictures from a short JSON file (no Python, no image model): posters, flyers, social cards,
  wallpapers, icons, illustrated scenes with animals, people, trees and skies, still lifes, rooms in 3D or isometric
  view, diagrams, flowcharts, charts, dashboards and UI mock-ups, with exact text. Use whenever the user asks to
  draw, paint, illustrate or render a picture; make a poster, flyer, card, banner, thumbnail or wallpaper; draw a
  diagram or chart; or mock up an app screen. Made for small local models: one tool draws, its last line says
  what to do next. 觸發詞：畫一張、海報、插畫、圖表、流程圖。
license: MIT
compatibility: Python 3.9+ with numpy and Pillow (installed by scripts/bootstrap.py). Works without vision.
---

# pixelcanvas-lite: a picture from one JSON file

You write one small JSON file. `compose.py` draws it. Its **last line always starts with `NEXT:`** and names the
one thing to do next. Do every step yourself, in this session, with no subagent or task.

## Setup

The commands below are ready to run. If a command still contains the word ROOT, write the folder of this
SKILL.md in its place: its `tools/compose.py` runs the pixelcanvas runtime (the first run sets it up).

## The loop

1. Pick KIND from the table below and a short NAME such as `jazz_poster`.
2. `python ROOT/tools/compose.py --example KIND > ./art/NAME.json`
3. Read `./art/NAME.json`. Write the whole file again with your content:
   - `"prompt"`: the request, copied word for word.
   - Replace every example text and thing with what the request asks for. Keep the keys and the structure.
   - Every text in quotes in the request goes into a `"text"` layer, spelled exactly the same.
4. `python ROOT/tools/compose.py ./art/NAME.json --check`
5. Do what the `NEXT:` line says. It is one of three things:
   - fix the lines under PROBLEMS or EDITS: write the whole JSON file again, then run step 4 again;
   - `deliver: ...`: run that command;
   - `tell the user the picture is at ...`: tell the user that path. You are done.

## Rules

- This skill uses exactly one tool: `compose.py`. You never write Python, never run `render.py`, `new_scene.py`
  or `look.py`, never open `scenes/` or the pixelcanvas source, and never search the disk for other skills.
  If you catch yourself doing any of that, stop and run the command in the last `NEXT:` line.
- Write the whole JSON file with the write tool every time. JSON only: double quotes, no comments.
- For `"draw"`, use names from the example or look one up: `python ROOT/tools/compose.py --names apple`
  (one plain word). When a name is unknown, use the closest name the error suggests.
- Places in words: `"at": "left"`, `"top right"`, `"under oak"`, `"on table"`, `"in the sky"`.
- Sizes in words: `"size": "small"` ... `"huge"`; one word per value: `"pose": "sit"`, `"sky": "night"`.
- Repeat a thing with `"count": 3`. Do not write three layers. Poster text needs no `"box"`.
- The report is text: trust it. You do not need to see the picture.

## Kinds

| The request is | KIND |
|---|---|
| poster, flyer, event, film or book cover | `poster` |
| social card, banner, thumbnail, quote | `card` |
| wallpaper, background; a set of icons or symbols | `wallpaper`; `icons` |
| an outdoor scene; animals or people in a place | `scene`; `animals`; `people` |
| trees, flowers, a garden | `nature` |
| mountains, sea, a view | `landscape` |
| fruit, cups, vases on a table | `still_life` |
| a room, bedroom, isometric room | `room` |
| 3D objects, a product shot, toys | `scene3d` |
| boxes and arrows, flowchart, process, architecture | `diagram` |
| one chart (bar, line, pie); KPIs and several charts | `chart`; `dashboard` |
| an app or website screen | `ui` |

## Local install location

(Filled in by `tools/install_skill.py`; when empty, ROOT is the folder of this SKILL.md, as in Setup.)
