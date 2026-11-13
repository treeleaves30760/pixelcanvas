# Creatures — animals in one call

`pixelcanvas.creature` draws 54 animals — mammals, birds, sea life and small creatures — as shaded 3D figures
built from smooth blended volumes on a real skeleton. You pick the species, the pose and the view; the module
does the anatomy (elbows bend back, hocks bend back, stifles forward), the gait, the fur, the markings and the face.

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas.creature import Creature, creature

cv = Canvas(960, 540, "#e9e4d8")
Creature("fox", size=0.35, view="three-quarter").pose("trot", phase=0.2).draw(cv, at=(0.3, 0.85))
creature(cv, "cat", at=(0.72, 0.85), size=0.4, pose="sit", view="three-quarter")
cv.save("outputs/creature_quick.png")
```

## Units

| argument | meaning |
|---|---|
| `at=(u, v)` | canvas fractions: the ground point under the animal; the body centre for fish, flying birds, insects in flight. Divide pixels by the canvas size; a pixel value such as `at=(640, 360)` raises a ValueError that says so |
| `size` | fraction of the canvas height: standing height (walkers, birds, frogs, octopus, jellyfish); body length (fish, whales, snakes, lizards, insects); wingspan (butterfly). It stays the *standing* height in every pose, so a lying dog keeps the scale of a standing one |
| `fit=(u0, v0, u1, v1)` | instead of `at` / `size`: fill this box (standing on its bottom edge, or centred for swimmers and fliers) |
| `at=ground.place(x, depth)` | a `form` Spot: `size` is then in world units and shrinks with distance (in a `Room`, 2.8 units high: a dog ≈ 1.1, a cat ≈ 0.8, a horse ≈ 2.5) |
| `.mask`, `.bbox`, `.anchor(name)` | full-frame mask, pixel box and pixel landmarks after `draw` |

## Calls

| call | what it does |
|---|---|
| `Creature(species, coat=None, size=0.3, view="side", facing="right", style="soft", seed=0, **anatomy)` | one animal; `species()` lists the names |
| `.pose(name, phase=0.0, **options)` | choose the pose (returns self) |
| `.draw(cv, at, log=None, shadow=True, cast=False, opacity=1.0, fit=None)` | paint it as one layer step; returns self |
| `.anchor("mouth")` | landmark in pixels: `mouth nose eye ear head back withers chest tail paw_front_l paw_front_r paw_hind_l paw_hind_r center top ground`; birds add `beak wing foot` |
| `.fit(cv, box)` | set `size` for a box and return the `at` point (used by `fit=`) |
| `creature(cv, species, at, size, pose=None, phase, view, facing, style, coat, fit, cast, **options)` | the same in one call |
| `flock(cv, "crow", n=7, region=(u0, v0, u1, v1), size=0.04, pose="fly")` | a group of birds with their own wing beats, depth sorted |
| `school(cv, "fish", n=9, region=..., size=0.06)` | a school of swimmers heading the same way |
| `species(kind=None)` | names; `kind` = `"quad"`, `"bird"`, `"sea"`, `"small"` |
| `COATS` | named coats: `tabby grey tabby black white siamese calico tuxedo ginger dalmatian chocolate black lab grey palomino black horse white horse fawn polar panda leopard cheetah` |

**Species.** Mammals: `cat dog labrador golden shepherd corgi poodle husky fox wolf horse zebra cow sheep pig goat deer
bear lion tiger elephant giraffe rabbit mouse squirrel`. Birds: `sparrow robin crow owl eagle duck swan penguin
parrot flamingo chicken`. Sea: `fish goldfish clownfish shark dolphin whale jellyfish octopus turtle`. Small:
`butterfly bee ladybird dragonfly snail frog snake lizard spider`. Everyday names work too: `"golden retriever"`, `"puppy"`, `"kitten"`,
`"pony"`, `"stag"`, `"bunny"`, `"hen"`, `"macaw"`, `"ladybug"`, and a coat in front of the name: `"siamese cat"`,
`"tabby cat"`, `"black lab"`, `"dalmatian"`, `"palomino horse"`, `"polar bear"`.

**Poses.** Mammals: `stand walk trot gallop sit lie sleep crouch jump graze sniff rear` — `phase` (0–1) runs
through the stride of `walk`, `trot` and `gallop` (the gallop has a real gathered suspension phase). Birds: `stand
perch fly glide swim sit peck` (`phase` = wing beat; `perch` stands on a twig). Fish and sea life: `swim` (`phase` =
tail beat). Small creatures: `stand`, `fly` / `open` / `rest` for butterflies, `fly` for bees, `jump` for frogs.

**Pose options.** `head_turn=20` (degrees towards the viewer), `head_tilt`, `tail="up" | "down" | "out" | "curl" |
"wag"` or an angle, `ears="back" | "down" | "alert"`, `mouth="open" | "pant"`, `eyes="closed"`, `wings="open" |
"up"`, `pitch` (camera elevation in degrees).

**Views.** `"side"`, `"three-quarter"`, `"front"`, `"back"`, `"top"` or a yaw angle in degrees; `facing="left"`
mirrors.

**Styles.** `"soft"` (shaded illustration with fur at the silhouette), `"flat"` (clean vector tones with an
outline), `"cute"` (big head, big eyes, short legs — sticker and picture-book look).

**Coats.** `coat` is a name, a colour or a dict. A name from `COATS` (`coat="calico"`). A colour (`coat="#d8883a"` or `(0.85, 0.5, 0.2)`) recolours the animal's main colour and the colours close to it, and keeps the markings (stripes, socks, spots, eyes, noses): `creature(cv, "cat", coat="#d8883a")` is an orange cat, `creature(cv, "sparrow", coat="#3a7bd5")` a blue bird. A dict sets part colours: `coat=dict(base=, belly=, pattern=, colours=[...])` with `pattern` = `solid tabby stripes tiger zebra
spots rosettes giraffe patches points saddle`, plus `socks=(height, colour)`, `tail_tip`, `mane`, `muzzle`,
`ear_back`, `inner_ear`, `eye`, `nose`, `points`. Birds: `back belly head cap face throat breast wing wing2 wing_bar
covert tail beak legs`. A wrong coat name lists the valid names with the closest one first; a wrong key in a dict does the same for the part names.

**Anatomy.** Any rig parameter can be overridden: `neck_len=0.5`, `muzzle=0.25`, `skull=(0.14, 0.12, 0.12)`,
`ear=dict(kind="floppy", len=0.2)`, `tail=dict(len=0.8)`, `f=(0.3, 0.35, 0.12)` (front leg segment lengths),
`horns=dict(kind="goat", len=0.3, r=0.03, color="#8a7a66")`, `antlers=dict(len=0.4, tines=3)`.

## Recipes

**A field-guide sheet** — let `fit=` size every animal for its cell:

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas.creature import Creature

cv = Canvas(900, 300, "#f2ebdc")
for i, (name, pose) in enumerate([("horse", "gallop"), ("owl", "perch"), ("clownfish", "swim"), ("frog", "stand")]):
    cell = (0.02 + i * 0.245, 0.08, 0.24 + i * 0.245, 0.82)
    Creature(name, view="three-quarter").pose(pose, phase=0.2).draw(cv, fit=cell)
    cv.text(name.title(), (cv.w * (cell[0] + cell[2]) / 2, cv.h * 0.92), size=18, color="#3a2e24", anchor="mm")
cv.save("outputs/creature_sheet.png")
```

**Animals in a landscape** — draw back to front, give the main animal a cast shadow, put grass in front of
the ones that hide in it, and use one `log` step per animal so each is its own layer:

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas.creature import Creature, flock
from pixelcanvas.form import set_light

cv = Canvas(960, 540, "#9cc0e0")
cv.paint("#7f9a42", (cv.uv()[1] > 0.55).astype("float32"))
set_light(direction=(-0.7, -0.6))
flock(cv, "crow", n=5, region=(0.55, 0.08, 0.92, 0.3), size=0.035, facing="left")
h = Creature("horse", size=0.3, facing="left", coat="palomino").pose("gallop", phase=0.2)
h.draw(cv, at=(0.38, 0.75), cast=dict(opacity=0.35, length=0.6))
Creature("fox", size=0.2, facing="left", view="three-quarter").pose("sit").draw(cv, at=(0.8, 0.93))
Creature("butterfly", size=0.07).pose("fly", phase=0.75).draw(cv, at=(0.15, 0.62))
cv.save("outputs/creature_scene.png")
```

**On a perspective ground** — a `form` Spot scales the animal with depth:

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas.creature import Creature
from pixelcanvas.form import Ground

cv = Canvas(960, 540, "#a9c46a")
g = Ground(cv, horizon=0.38)
for x, z in ((-2.0, 9.0), (1.5, 6.0), (-0.5, 3.5)):
    Creature("sheep", size=0.9, view="three-quarter").pose("graze").draw(cv, at=g.place(x, z), cast=True)
cv.save("outputs/creature_flock.png")
```

**Pets at home** — a `Room` floor gives every animal its perspective size and a cast shadow from the window:

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas.creature import Creature
from pixelcanvas.form import Room, set_light

cv = Canvas(960, 540)
set_light(direction=(-0.8, -0.4), elevation=22, color="#ffc890")     # low evening sun from the left window
room = Room(cv, horizon=0.42)
room.draw(floor="planks", window="left")
dog = Creature("golden retriever", size=1.1, view="three-quarter").pose("lie", head_turn=15)
dog.draw(cv, at=room.place(-0.45, 2.75), cast=True)
cat = Creature("tabby cat", size=0.8, view="three-quarter", facing="left").pose("sit")
cat.draw(cv, at=room.place(0.8, 2.9), cast=True)
```

**Use the anchors** — attach things to the animal after drawing it:

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas.creature import creature

cv = Canvas(600, 400, "#efe7da")
dog = creature(cv, "corgi", at=(0.45, 0.9), size=0.6, pose="sit", view="three-quarter", mouth="pant")
x, y = dog.anchor("mouth")
cv.text("Woof!", (x + 40, y - 60), size=36, color="#3a2e24", anchor="lm")
```

**Under water** — fish centre on `at`; `school` spreads a group:

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas.creature import Creature, school

cv = Canvas(960, 540, "#1d5f86")
school(cv, "fish", n=9, region=(0.1, 0.2, 0.6, 0.6), size=0.07, seed=4)
Creature("turtle", size=0.3, view="three-quarter").pose("swim", phase=0.3).draw(cv, at=(0.72, 0.45))
Creature("jellyfish", size=0.18).pose("swim").draw(cv, at=(0.25, 0.75))
```

Options are checked: a misspelt anatomy or pose option (`neck_lenn=`, `head_turnn=`) raises an error that names the
closest valid one, and both spellings are accepted (`pos` / `at`, `facing=-1` for `"left"`). Unknown species, poses,
views and styles list the nearest names. Every drawn animal is one entry in the scene manifest, named by the word you asked for (`"panda"` stays panda, with `bear` and `animal` as aliases), so `--check` counts it once; creatures register themselves, so no extra `manifest.note` is needed.
