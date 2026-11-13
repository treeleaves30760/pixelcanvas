# Changelog

## 0.4.0 (2026-11-13)

pixelcanvas grows from a painting toolkit into a set of kits that draw a finished picture in a few calls. Ask your
agent for a poster, a diagram, a dashboard, a floor plan, a 3D product shot or a fox in a snowy forest, and it finds
the right kit, starts from a matching template and checks the result before handing it over.

First release on PyPI: `pip install pixelcanvas`, or `uvx pixelcanvas doctor` to try it without installing.

### Design: posters, diagrams, charts and screens

- **vector**: shapes, paths, groups and 123 icons on a `Drawing`, rendered as PNG and saved as SVG.
- **diagram**: flowcharts, architecture diagrams, trees, sequences, timelines, tables and callouts with automatic
  layout (13 diagram kinds, 12 shapes).
- **chart**: 15 chart kinds, KPI tiles and whole dashboards, styled for publication.
- **ui**: declarative app screens with auto layout in 7 device frames, from 14 templates and 62 components.
- **layout** and **text**: 12 poster / card templates, 9 themes, 18 text effects, auto-fitting type in any
  language.
- **backdrop**: 57 one-call backgrounds and wallpapers that keep a calm area for the title, with 60 palettes.

### Technical drawings

- **drafting**: engineering sheets with projected views, sections, dimensions and title blocks, and architectural
  floor plans with walls, doors and 16 plan symbols. Every sheet saves as PNG, SVG and DXF.

### 3D

- **solid**: numpy ray-marching with 40 prefabs (furniture, tableware, products ...), 18 materials, studio lighting,
  perspective and isometric cameras.
- **clay**: toy-like 3D characters with faces, 22 hair styles, 13 outfits, 15 expressions and 31 poses; 64 animal
  species in 15 poses; 33 props.

### Illustration

- **figure**: posed, dressed people on a skeleton, with hands, faces, hair and clothes.
- **creature**: 54 animal species on a skeleton, with poses, gaits and coats.
- **flora**: 97 trees, shrubs, flowers and potted plants, grass and rocks.
- **elements**: fire, smoke, water, weather, 12 skies and light effects.
- **texture**: 80 lit materials: wood, stone, fabric, metal, fur and more.
- **form**: 2.5D objects with volume, rooms in perspective, one light and soft shadows.
- **scenery**: landscape recipes from sky to god rays.
- **pictograms**: 633 pictograms in six styles (silhouette, duotone, line, shaded, sticker, cutout), also usable as
  masks.
- **stylize**: 43 art styles, from oil and watercolour to ukiyo-e and risograph; `styles pick "REQUEST"` chooses
  one from the request.

### From request to picture

- `kit find "REQUEST"` searches every kit at once and answers with the text to draw, one call per subject in order,
  the style and the scene kind.
- `new NAME --prompt "REQUEST"` starts from the matching one of 29 starter templates, already filled in for the
  request.
- `render SCENE --check` writes a check sheet (guides, values, edges, thumbnail) and a report: warnings, the
  objects drawn and where, and whether the picture covers the prompt (quoted text, subjects, style).
- Spec mode: `compose --example poster` writes a JSON spec and `compose SPEC.json --check` draws it, in 16 kinds
  from posters to dashboards and 3D scenes. The new **pixelcanvas-lite** skill is built on it, so small local models
  draw finished pictures too.

### Install anywhere

- The `pixelcanvas` command reaches every tool: `pixelcanvas kit`, `new`, `render`, `compose`, `look`, `styles`,
  `layers`, `fonts`, `install-skill`, `doctor`.
- `pixelcanvas install-skill` installs both skills for Claude Code, Codex, Cursor, opencode or any Agent Skills
  tool, for one user or one project.
- `pixelcanvas mcp` runs an MCP server with 11 tools (new scene, render, look, find, reference pages ...) for any MCP
  client.
- Claude Code plugin marketplace manifest, and a skill bootstrap that finds or fetches the runtime by itself.

## 0.3.0 (2026-11-11)

Editable layers (PSD / ORA / PNG), media stylization, inspection views for vision models, typography and SDF
shapes; the skill installs with `npx skills add`.
