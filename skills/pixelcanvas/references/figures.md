# People — `pixelcanvas.figure`

Posable, dressed people in one call: a real skeleton with joint angles, tapered limbs, hands with fingers, faces
with expressions, 15 hair styles, clothing over the body, accessories, four rendering styles.

## Quick start

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas.figure import person, Figure

cv = Canvas(1600, 900, (0.95, 0.92, 0.87))
person("chef").draw(cv, at=(0.25, 0.92))                                   # a ready-made character
person("a tall woman in a red coat waving", height=0.7).draw(cv, at=(0.5, 0.92))
Figure(height=0.7, build="child", view="three_quarter", style="anime",
       hair=dict(style="curly", color="ginger")).pose("cheer").draw(cv, at=(0.78, 0.92))
cv.save("people.png")
```

`at` is the ground point under the feet in canvas fractions (`(0.25, 0.92)` is a quarter across, 92% down); divide
pixels by the canvas size (`(x / cv.w, y / cv.h)`). A value that is clearly pixels raises a ValueError that says so.
`height` is the standing height as a fraction of the canvas height. Every draw adds a cast shadow and contact shadows
(`shadow=False` turns them off).

## Calls

| call | what it does |
|---|---|
| `Figure(height, build, body, view, facing, skin, hair, outfit, face, style)` | a person; `build`: average slim broad child elder chibi heroic; `body`: neutral feminine masculine; `view`: front three_quarter side back three_quarter_back (or degrees); `facing`: 1 right / -1 left |
| `skin=` | `SKIN_TONES`: porcelain fair light medium tan olive brown deep ebony, or any colour |
| `hair=dict(style, color)` | style: short crop buzz side_part spiky bob pixie long ponytail bun curly afro braids bald receding; color: `HAIR_COLORS` (black espresso dark_brown brown auburn copper ginger blonde platinum gray white pink blue lilac teal) or hex |
| `outfit=dict(top, bottom, shoes, colors=dict(...), extras=[...])` | top: tshirt shirt blouse hoodie sweater jacket coat suit dress tank spacesuit; bottom: trousers jeans shorts skirt leggings (None with a dress); shoes: sneakers boots shoes flats (None = barefoot); colors keys: top bottom shoes inner accent hat scarf bag apron, values hex or `PALETTE` names (red blue navy teal mustard coral white ...); `long_sleeves=False`, `sleeves="short"` |
| extras | glasses sunglasses hat cap beanie beret chef_hat helmet headphones scarf bag backpack tie apron stethoscope beard mustache |
| `face=dict(expression, eyes, gaze, look, blush)` | expression: neutral smile laugh surprised sad angry wink sleepy; `eyes` iris colour; `gaze=(x, y)` -1..1; `look=(u, v)` canvas point |
| `style=` | `flat` editorial vector, one shadow tone · `soft` lit volumes · `anime` soft-cel: base plus a hard shadow band on skin, hair and cloth, coloured outlines, big eyes · `line` ink outlines with spot colour |
| `fig.pose(name, phase=0, mix=None, amount=0.5, item=None, face=None, **joints)` | name: stand walk run sit wave point reach hold jump dance kneel lean arms_crossed hands_on_hips thinking cheer talk phone guitar; one-arm poses (wave, point, reach, hold, talk, phone) use the right arm; `walk` / `run` `phase` 0~1: 0 = legs passing, 0.25 = full stride (a run is airborne there); returns the figure |
| joints | `lean bend twist hip_tilt pelvis_turn roll lift nod turn tilt` (degrees), `shoulder_l=(flex, abduct, roll)`, `elbow_r=deg`, `hip_l=(flex, abduct, roll)`, `knee_l`, `ankle_l`, `wrist_r`, `hand_r=` relaxed open fist point grip wave peace thumbs_up, `palm_r=` in out front back up down, `reach_r=(u, v)` canvas target or `("chest", (dx, dy, dz))`. `_r` is the figure's own right: in a front view `reach_r` is the hand on the viewer's left and `reach_l` the one on the viewer's right, so aim each hand at its own side |
| `hold_both=(u, v)` | both hands grip one object centred at that canvas point (a pot, a tray, a cake): `fig.pose("hold", hold_both=(0.4, 0.6), hold_width=0.29)`; `hold_width` is the distance between the hands (canvas fraction). The hands pick their sides from the view, so the arms never cross, and the elbows keep a natural bend |
| `seat=` (sit) | `SEATS`: `auto` (default: a wooden stool under the sitter at a plain `(u, v)` point; at a scene `Spot` the room supplies the chair) `stool` `bench` `chair` `none` (seated on your own sofa, wall or step); `seat_color="#6b3f2a"` |
| `item=` | cup phone book bag briefcase guitar ball flower (held in the right hand) |
| `build="chibi"` | a big head with large eyes: the face stays readable at thumbnail and sticker sizes, where an `average` face is a few pixels wide |
| `name=`, `alias=` | the words the scene report knows the figure by: `person("barista", name="friend", alias=["classmate"])`; "person" is always an alias, so three friends count as three people |
| `fig.draw(cv, at, log=None, shadow=True)` | draws; afterwards `fig.anchor(name)`, `fig.bbox`, `fig.mask`, `fig.head_size` |
| `fig.layout(cv, at)` | geometry and anchors without drawing (plan a scene first) |
| `fig.anchor(name)` → (x, y) px | head_top head eye_l eye_r nose mouth chin ear_l neck chest waist pelvis back shoulder_l elbow_l wrist_l hand_l fingertip_l hip_l knee_l ankle_l foot_l toe_l heel_l (and `_r`), feet seat lap speech |
| `fig.look_at((u, v))` | turn the eyes to a point |
| `person(name_or_description, **overrides)` | `PEOPLE`: office_worker businessman runner chef doctor student kid grandma grandpa musician astronaut barista artist hiker teen hero dancer tourist; or a sentence ("an old man with glasses sitting") |
| `random_person(seed, **overrides)` | varied skin, hair, build and outfit from tuned palettes |
| `crowd(cv, n, region, seed, ground=None, height=0.34, style="flat")` | n people placed in `region=(u0, v0, u1, v1)` (feet positions), drawn back to front, smaller with distance |
| `pose_lerp(a, b, t)`, `walk_pose(phase)`, `run_pose(phase)` | pose dictionaries for in-betweens and cycles |

## Recipes

A person with a balloon and a speech bubble, using anchors:

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas import sdf
from pixelcanvas.figure import Figure

cv = Canvas(1600, 900, (0.93, 0.95, 0.98))
kid = Figure(height=0.55, build="child", view="three_quarter", hair=dict(style="bun", color="black"),
             outfit=dict(top="dress", colors=dict(top="#ec6b56")), face=dict(expression="laugh"))
kid.pose("wave", hand_r="grip", palm_r="in").draw(cv, at=(0.4, 0.9))
hx, hy = kid.anchor("hand_r")
X, Y = cv.xy()
cv.paint("#555555", sdf.fill(sdf.sd_segment(X, Y, hx, hy, hx + 40, hy - 170, 1.2)))   # string
cv.paint("#e63946", sdf.fill(sdf.sd_ellipse(X, Y, hx + 40, hy - 225, 55, 65)))         # balloon
bx, by = kid.anchor("speech")
cv.paint("#ffffff", sdf.fill(sdf.sd_box(X, Y, bx + 120, by - 40, 120, 50, 30)))
cv.text("Look!", (bx + 120, by - 40), 44, "#222222", anchor="mm", bold=True)
```

A girl running on the beach with a kite: `reach_r` aims the arm at a canvas point in any view, and the string
starts at `anchor("hand_r")`:

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas import sdf
from pixelcanvas.figure import Figure

cv = Canvas(1080, 1350, (0.98, 0.9, 0.72))
kite = (0.8, 0.15)
girl = Figure(height=0.5, build="slim", body="feminine", view="side", style="anime", skin="fair",
              hair=dict(style="long", color="dark_brown"),
              outfit=dict(top="tshirt", bottom="shorts", shoes=None, colors=dict(top="white", bottom="blue")),
              face=dict(expression="laugh", look=kite))
girl.pose("run", phase=0.25, reach_r=kite, hand_r="grip").draw(cv, at=(0.35, 0.9))
X, Y = cv.xy()
hx, hy = girl.anchor("hand_r")
kx, ky = kite[0] * cv.w, kite[1] * cv.h
cv.paint("#555555", sdf.fill(sdf.sd_segment(X, Y, hx, hy, kx, ky, 1.2)))
cv.paint("#ff4d6d", sdf.fill(sdf.sd_polygon(X, Y, [(kx, ky - 70), (kx + 50, ky), (kx, ky + 90), (kx - 50, ky)])))
```

Friends of different heights waving: vary `height`, `build`, `skin`; `facing` turns people towards each other. To
wave with the left arm instead, set the left joints (`shoulder_l=(16, 112, 60), elbow_l=70, hand_l="wave"`):

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas.figure import Figure

cv = Canvas(1920, 1080, (0.8, 0.9, 0.97))
specs = [(0.62, "slim", "deep", "afro", "#f4a261", 1), (0.7, "broad", "light", "side_part", "#2a9d8f", 1),
         (0.55, "average", "tan", "ponytail", "#e76f51", -1)]
for i, (h, build, skin, hair, top, facing) in enumerate(specs):
    Figure(height=h, build=build, skin=skin, view="three_quarter", facing=facing, hair=dict(style=hair),
           outfit=dict(colors=dict(top=top))).pose("wave", face="smile").draw(cv, at=(0.3 + 0.2 * i, 0.92))
```

A street with a crowd in perspective, plus a hero character in front:

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas.form import Ground
from pixelcanvas.figure import crowd, person

cv = Canvas(1920, 1080, (0.85, 0.88, 0.9))
g = Ground(cv, horizon=0.42)
cv.paint(g.tiles(size=0.8), g.mask())
crowd(cv, 14, region=(0.03, 0.5, 0.97, 0.85), seed=5, ground=g)
person("runner", view="side", facing=-1, height=1.7).draw(cv, at=g.spot(0.7, 0.95))   # metres at a Spot
```

Two people talking at a table (side views facing each other), each on a chair; draw people first, then the
table in front:

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas.figure import person

cv = Canvas(1600, 900, (0.94, 0.9, 0.84))
a = person("a woman with a bob in a yellow sweater", view="side", facing=1, height=0.62)
a.pose("sit", seat="chair", reach_r=("chest", (0.0, -0.5, 0.75)), hand_r="grip", face="laugh")
a.draw(cv, at=(0.38, 0.9))
b = person("man", view="side", facing=-1, height=0.66)
b.pose("sit", seat="chair", reach_l=("chest", (0.1, -0.7, 0.8)), hand_l="open", palm_l="up").draw(cv, at=(0.62, 0.9))
seat = a.anchor("seat")          # seat="none" + your own sofa or wall under (seat[0], seat[1])
```

A chef carrying a pot with both hands (`hold_both` sets both grips at once):

```python
from pixelcanvas.canvas import Canvas
from pixelcanvas.figure import person

cv = Canvas(900, 1100, "#efc99a")
chef = person("chef", name="chef", style="soft", height=0.74, face=dict(expression="smile", blush=True))
chef.pose("hold", hold_both=(0.5, 0.6), hold_width=0.3).draw(cv, at=(0.5, 0.95))
hx, hy = chef.anchor("hand_r")          # the grip on the viewer's left: hang a pot handle here
```

Tips: one `set_light(...)` from `pixelcanvas.form` lights every figure and object the same way. Use `style="line"`
for diagrams and comics, `anime` for character art, `soft` next to `form` solids. At a perspective `Spot`
(`ground.place`, `room.place`) `height` is in metres (1.6~1.85 adults, 1.1~1.3 children). Pass `log=log` in a scene
so each person becomes its own layer.

Options are checked: a misspelt option (`hieght=`), a misspelt key inside `hair` / `outfit` / `face`, or an unknown
joint in `pose(...)` raises an error that names the closest valid one, and both spellings are accepted
(`colour` / `color`, `pos` / `at`). Unknown poses, builds, hair styles, tops and extras list the nearest names. Every
drawn person is one entry in the scene manifest (named by `name=`, else `person`), so `--check` can count them.
