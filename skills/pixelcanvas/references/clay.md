# clay — 3D characters, animals and props with a clay / designer-toy look

`pixelcanvas.clay` places finished, posable 3D figures in a `pixelcanvas.solid` Scene with one call each: people
with a real skeleton, faces that read at a glance, sculpted hair and clothes; 60+ animals with their own rigs,
poses and markings; and props to hold or sit near. Soft rounded volumes, eyes set low and wide, big clear
silhouettes: stop-motion studio, vinyl toy, modern 3D illustration.

Reach for it whenever a picture needs people or animals — a cast line-up, a story scene, a mascot, a field guide,
a product shot with a figure, an illustration with characters.

## Quick start

```python
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera
from pixelcanvas.clay import cast, animal, prop

sc = Scene(lighting="studio", ground="#e9e2d6")
sc.add(cast("chef", pose="wave").place(x=-0.5), "chef")
sc.add(animal("corgi", pose="sit").place(x=0.4, z=0.2), "corgi")
sc.add(prop("balloon", color="#3a8ae8").place(x=1.0, z=-0.2), "balloon")
sc.camera = Camera.frame(sc, azimuth=12, elevation=8)
cv = Canvas(960, 540)
res = sc.render(cv)
cv.save("clay.png")
```

Units are metres, y is up, figures stand on y = 0 and face +z (towards the default camera). `.face(a)` turns a
figure to face `(sin a, 0, cos a)`: 90 looks to +x (right in the default view), -90 to -x, 180 away; the solid
`rotate_y(a)` uses the same sign, so a figure turned like the chair it sits by matches it (`.face(-35)` next to
`armchair().rotate_y(-35)`). `_l` / `_r` names are the figure's own sides: `hand_r` is at -x when it faces +z.
Turn first, then `.on(chair, "armrest_r")` rests it there, touching the surface. Animals default to real sizes
(a cat 0.38 m to the ears, a labrador 0.6 m), which read right next to `solid` furniture. Every figure is a
solid `Model` with named parts (`skin`, `hair`, `top`, `bottom`, `shoes`, `eyes`, `mouth`, `fur`, ...), so
`res.mask("chef.top")`, recolouring, materials, layers and the manifest all work unchanged.

## Calls

| Call | What it does |
|---|---|
| `cast(name, pose=, phase=, build=, **changes)` | ready-made character: kid student office_worker chef doctor nurse scientist farmer firefighter police_officer astronaut artist musician athlete runner hiker grandma grandpa baby wizard knight pirate princess superhero robot_friend santa teacher gardener builder dancer skater explorer |
| `random_person(seed, build=, pose=)` | varied, harmonious person from good skin / hair / outfit palettes (crowds) |
| `Character(name, build, height, skin, hair, face, outfit, material, detail)` | a custom character (see below) |
| `ch.pose(name, phase=0, mirror=False, **joints)` | named pose; `phase` animates walk / run / wave / clap / dance; overrides like `elbow_r=60`, `head=dict(turn=20)` |
| `ch.blend(name, t)` | blend the current pose toward another |
| `ch.look_at(point)` `ch.face(angle_or_point)` | aim head and eyes; turn the whole body (degrees: 0 = +z, 90 = +x) |
| `fig.on(model, part, at=(u, v))` | rest a figure or animal on a placed solid model or one of its parts (`seat`, `armrest_r`, `top`, `shelf`, `deck`) |
| `ch.sit_on(seat_or_None, height=)` | sit on a placed sofa / chair / bench / stool: on its cushions, back to the backrest, knees at the front edge; `None` = on the ground |
| `ch.hold(prop_or_name, hand="r")` `ch.drop()` | put a prop in a hand (`"l"`, `"r"` or `"both"`); balloons, flags and umbrellas stay upright |
| `ch.express(name)` `ch.recolor(top="#e33", hair=...)` | change the face or colours |
| `ch.anchor(name)` | 3D point: hand_l/r, head, head_top, mouth, eye_l/r, nose, chest, belly, back, lap, seat, feet, foot_l/r, shoulder_l/r, elbow_l/r, hip_l/r, knee_l/r, ear_l/r |
| `animal(species, pose=, phase=, size=, coat=, color=, cute=)` | clay animal; `size` = standing height in metres; `cute=1.6` = big head, big eyes, blush and a smile |
| `prop(name, **options)` | balloon book cup phone umbrella guitar ball flower bouquet ice_cream sign flag lantern fishing_rod skateboard bicycle picnic_blanket basket campfire bench stool spoon flask paintbrush staff sword wand briefcase apple sandwich cake juice kite |
| `print_on(cv, res, who, text, where="chest")` | print a word or number on clothes after the render; it stays on the garment |
| `decal_frame(res, who, where)` | where a surface frame lands on the canvas (for your own 2D detail) |
| `names(kind)` `find(text)` `describe(name)` | registry: kinds character animal prop pose animal_pose expression hair outfit build |

**Character options.** `build`: toy (3 heads) chibi (2.5) baby kid (4) teen adult (5.5) elder hero (6.5).
`skin`: an index 0–9 into `SKIN_TONES` or a colour. `hair=dict(style=, color=)` with styles short crop buzz
side_part quiff spiky bob long ponytail pigtails bun top_knot curly afro braids mohawk bald balding hijab pixie wavy
baby_tuft and colours from `HAIR_COLORS` (black brown auburn ginger blonde grey white ...) or any colour.
`face=dict(expression=, eyes=, brows=, cheeks=, glasses=, beard=, freckles=, nose=, ears=, head=)`; expressions neutral smile
happy laugh surprised sad angry wink sleepy sleep determined shy cheeky worried calm. `outfit` = a preset name
(`OUTFITS`: casual school office suit sporty winter summer dress pyjamas hiking cosy hoodie overalls) or
`dict(top=, bottom=, shoes=, hat=, extras=[...], colours=dict(top=, bottom=, shoes=, hat=, accent=))`.
Tops (`TOPS`): tshirt tank longsleeve shirt polo sweater hoodie jacket coat raincoat lab_coat suit dress gown robe
jersey spacesuit armor onesie; bottoms (`BOTTOMS`): trousers jeans shorts skirt long_skirt leggings overalls puffy;
shoes (`SHOES`): sneakers boots shoes sandals barefoot rain_boots slippers heels; hats: `HATS`; extras (`EXTRAS`):
backpack bag scarf cape glasses sunglasses headphones tie bowtie necklace medal stethoscope belt apron watch wings.
Everyday words work too (jumper, pants, wellies, trainers, dungarees ...), and a misspelt piece raises an error
that names the closest one. `material`: clay plastic plush
wood vinyl. `detail`: low (small in frame) normal high (separate fingers, more hair tufts).

**Poses** (`POSES`): stand contrapposto walk run jump sit sit_floor sit_legs kneel lie wave point cheer clap
think arms_crossed hands_on_hips hold carry hug dance kick throw reach_up look_up sleep phone typing reading shrug
bow. The feet are planted on y = 0 after posing; action poses shift the weight and counter-rotate hips and
shoulders. `phone` comes with a phone in the right hand and `reading` (a seated pose, like `sit` and `typing`) with
an open book in both hands; `cast()` moves a signature prop to the left hand when the pose needs the right one
(wave, point, phone ...).

**Animals** (`SPECIES`): cat (coats tabby ginger black white grey tuxedo calico siamese), dog labrador corgi
poodle husky dachshund shiba, fox wolf rabbit mouse squirrel pig sheep goat cow horse pony zebra deer giraffe
elephant rhino hippo lion tiger bear polar_bear panda koala monkey kangaroo; birds chick chicken duck penguin owl
parrot flamingo eagle sparrow; sea life fish shark whale dolphin turtle octopus crab seal; small things frog snail
ladybird bee butterfly caterpillar; t_rex triceratops brontosaurus stegosaurus dragon unicorn. Poses
(`ANIMAL_POSES`): stand walk trot gallop sit lie sleep jump beg play_bow graze rear, plus fly (birds, `phase` =
wing beat), swim, peck. `sleep` curls any four-legged animal up on its side, legs tucked, tail wrapped round, eyes
shut. Markings (stripes, spots, patches, socks, masks, saddles) are 3D colour functions that
follow the body; `coat=` picks a named coat or takes `dict(fur=, marks=[...])`.

**Cute.** `cute=` scales the head and eyes (1.15 is the house look; 1.3–1.8 is the vinyl-toy, sticker and mascot
range) and, above 1.2, adds pink blush under the eyes and a small smile under the nose. It works on every family:
birds get the big head, big eyes and blush (the beak stays the mouth), fish and whales the eyes and blush.
`animal("whale", shape="round")` is the round cartoon whale (big round body, flukes up, smile, blowhole); the
default whale is the long sperm-whale shape.

**Campfire.** `prop("campfire")` has crossed logs with glowing ember ends, a stone ring and layered flames (a pale
core inside orange tongues with red tips). Add `sc.light_from(fire)` and the flames light the stones, the ground and
anyone sitting near it.

## Recipes

A line-up with name labels and a printed shirt:

```python
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera
from pixelcanvas.clay import cast, print_on

sc = Scene(lighting="studio", ground=None)
sc.backdrop("#f0dcc8")
who = [("kid", "wave"), ("doctor", "arms_crossed"), ("astronaut", "cheer")]
figs = []
for i, (name, pose) in enumerate(who):
    ch = cast(name, pose=pose).face(10 - 10 * i).place(x=(i - 1) * 0.8)
    sc.add(ch, name)
    figs.append(ch)
sc.camera = Camera.frame(sc, azimuth=0, elevation=6)
cv = Canvas(960, 540)
res = sc.render(cv)
print_on(cv, res, figs[0], "7", where="chest", color="#ffffff")
for (name, _), ch in zip(who, figs):
    x, y = sc.project(ch.anchor("feet"))
    cv.text(name.title(), (x, y + 12), 22, color="#5a4436", anchor="mt", bold=True)
```

A custom character holding a prop, a dog looking up at her:

```python
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera
from pixelcanvas.clay import Character, animal, prop

mia = Character("Mia", build="kid", skin=3, hair=dict(style="pigtails", color="brown"),
                face=dict(expression="laugh", freckles=True),
                outfit=dict(top="hoodie", bottom="jeans", shoes="sneakers", extras=["backpack"],
                            colours=dict(top="#f2a03a", bottom="#3f6fb0")))
mia.pose("walk", phase=0.25).hold(prop("balloon", color="#e8413a"), hand="l")
dog = animal("shiba", pose="sit").face(-60).place(x=0.8, z=0.5)
dog.look_at(mia.anchor("head"))
mia.look_at((1.0, 1.0, 4.0))            # towards the viewer
sc = Scene(lighting="sunny", ground="#cfe0a8")
sc.add(mia, "Mia")
sc.add(dog, "dog")
sc.camera = Camera.frame(sc, azimuth=10, elevation=8)
res = sc.render(Canvas(960, 540))
```

Sitting on a bench, and a crowd:

```python
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera
from pixelcanvas.clay import cast, random_person, prop

sc = Scene(lighting="soft_box", ground="#e4ddd2")
bench = prop("bench").place(x=2.4, z=-0.6)
sc.add(bench, "bench")
sc.add(cast("grandpa").sit_on(bench).hold("book"), "grandpa")
for i in range(5):
    sc.add(random_person(seed=i, pose="walk", phase=i / 5).face(90).place(x=-2.6 + i * 1.1, z=0.6), f"person {i}")
sc.camera = Camera.frame(sc, azimuth=15, elevation=10)
res = sc.render(Canvas(960, 540))
```

A rainy evening: a reader on the sofa, and friends at a stop with an umbrella, a phone and a wave:

```python
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, sofa
from pixelcanvas.clay import Character, cast, animal
from pixelcanvas.elements import rain

sc = Scene(lighting="interior", ground="#d8b48c")
couch = sofa("#6f8c7a").place(z=-0.8)
sc.add(couch, "sofa")
sc.add(Character("Noa", build="kid", outfit="cosy").pose("reading").sit_on(couch), "Noa")   # book included
sc.add(animal("labrador", pose="lie").face(-30).place(x=0.5, z=0.5), "dog")
sc.add(animal("cat", coat="ginger", pose="sit").face(20).on(couch, "armrest_l"), "cat")   # curled on the arm
sc.camera = Camera.frame(sc, azimuth=10, elevation=12)
res = sc.render(Canvas(960, 540))

out = Scene(lighting="overcast", ground="#4a4e58")
out.add(Character("Ari", build="adult", outfit="winter").pose("hold").hold("umbrella").place(x=-0.8), "Ari")
out.add(cast("student", pose="phone").place(x=0.0), "Bea")
out.add(Character("Cy", build="kid", outfit=dict(top="raincoat", shoes="wellies")).pose("wave", phase=0.75)
        .face(40).place(x=0.8), "Cy")
out.camera = Camera.frame(out, azimuth=-8, elevation=6)
cv = Canvas(960, 540)
out.render(cv)
rain(cv, density=1.0)                                       # 2D weather over the render
```

Night by the campfire: a cat asleep, a cute chick and a round whale toy:

```python
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, Lighting
from pixelcanvas.clay import animal, prop

sc = Scene(lighting=Lighting("night", ambient=0.35), ground="#2f3d2c")
fire = sc.add(prop("campfire").scale(1.4), "campfire")
sc.light_from(fire, power=2.2)                                   # firelight on the stones, ground and animals
sc.add(animal("cat", coat="grey", pose="sleep").face(-30).place(x=0.75, z=0.3), "cat")
sc.add(animal("chick", size=0.22, cute=1.6).face(-20).place(x=-0.55, z=0.5), "chick")
sc.add(animal("whale", shape="round", size=0.3, cute=1.4).face(40).place(x=-0.9, z=-0.2), "whale toy")
sc.camera = Camera.orbit(target=(0.0, 0.3, 0.0), distance=3.0, azimuth=10, elevation=16, fov=36)
res = sc.render(Canvas(960, 540))
```

Tips: `detail="low"` for figures that are small in frame (crowds); give each figure its own `sc.add` name so it
gets its own layer and mask; `res.depth_of_field(cv, focus="Mia")` and `res.fog(...)` finish a story scene; a
960x540 draft with five characters and a few props renders in about 8 s, 1920x1080 in about 30 s.

**Starting points.** `python tools/new_scene.py NAME --kind characters` (two or three posed characters, one held
prop, studio ground and backdrop) and `--kind pets` (two animals on a rug with a ball) give a framed, lit scene to
edit. In a Scene each character is recorded as a `person` (aliases chef, person, people, character) and each animal
as a `creature` (aliases the species, the family word such as dog, and animal), so `--check` counts "three people" or
"two dogs" against the prompt. Misspelt options or names raise an error naming the closest valid ones
(`clay.cast("chf")` -> did you mean 'chef'); `clay.dog` answers with `clay.animal("dog")`.
