# FLOCK 2.0 replay viewer

A top-down, zoomable, pannable map of a recorded meadow run, for watching what the sheep do. Vanilla
JS and canvas, no build step, no external dependencies.

## Run it

From the repository root (the page fetches the replays from `docs/flock-2/meadow2-results/replays/`,
so the server must serve the whole repository):

    python -m http.server 8765

then open <http://localhost:8765/web/replay/>. The `.claude/launch.json` entry `replay` starts the same
server. Opening `index.html` from the file system does not work: browsers refuse `fetch` on `file://`.
A different replay directory can be given as `?dir=../../runs/replays/` (relative to this page).

## Record a run

    python scripts/m2_record.py --policy scripted --willows 3 --days 3 --label "Scripted, plain world" --description "..."
    python scripts/m2_record.py --policy hybrid --agent ~/flock2_data/m2_hybrid/s2f_s0 --inner 8 --fog 0.75 --days 5 --label "..." --description "..."

The world options are `scripts/m2_eval.py`'s (`--willows`, `--inner`, `--edge`, `--fog`, `--thirst`,
`--sheep`, `--trample`, `--zero_column`), the meadows are the evaluation's (seeds 50,000 + i; `--meadows`
runs several, `--pick` writes one), so a recording is one of the evaluation's meadows seen from above.
`--grass_every` sets the grass snapshot interval (default 100 steps). Each run writes
`<name>.flock.gz` and refreshes `index.json`, which the menu reads. `runs/m2_record.sh` reproduces the
seven labelled recordings (a-g) in the replay directory.

## What you see

- **Ground**: grass density from bare (tan) to lush (green), from the latest snapshot; water with the
  sim's curved shoreline; the mud ring (stage 2a's `regrow_map`) as a brown tint where regrowth is
  slowed, opaque where it is zero. The fence is the meadow's edge; the dashed line is the half-tile
  band no sheep may enter.
- **Objects**: trees (dark green), willows (pale green with drooping lines; they stand on the pond
  banks), the hut (brown square), salt (white diamond), flowers (pink), mushrooms (red cap).
- **Rings** at 8, 12 and 16 tiles from each pond's edge (approximated as circles round the pond's
  centre of mass): the distance bins the memory rule M1 and the R1 reading use. 8 tiles is also the
  eyes' near range.
- **Sheep**: a triangle pointing along the heading, filled by the chosen colouring (action by default;
  sheep id, hydration, energy, fat or plain). The arc round a sheep is its hydration (blue above 0.5,
  orange above 0.25, red below). A centre dot marks a resting sheep; a blue dot ahead of the mouth a
  successful sip, a green one a successful bite. Dead sheep are grey with a red cross where they fell.
- **Trails** of the last N steps (slider); the **field of view** wedge is the 300-degree cone to 8 tiles.
- **Night and fog** tint the map; the scrub bar shades nights (dark bands), fog days (grey top band),
  and ticks deaths (red, full height), drinking bouts (blue, bottom) and each sheep's first approach to
  within 8 tiles of water (green).
- **Clock**: day number (`t / DAY`, where t counts from dawn of day 0 and a run starts at a random time
  of day), a clock with dawn at 06:00, the step in the recording, light level, night / fog, sheep alive.
- **Side panel**: one row per sheep with its action, hydration, energy, fat, distance to the nearest
  water tile centre and state; click a row or a sheep to select it (highlight, its FOV, its trail
  emphasised); "follow" keeps it centred. Hover the map for tooltips on sheep, objects and tiles.

Keys: space play / pause, left / right step (shift: 100), + / - zoom, 0 reset view, f follow, e / E next
/ previous event, [ ] slower / faster, home / end, 1-9 select sheep 1-9 (click the others), esc deselect.
Mouse: wheel zooms at the cursor, drag pans, click the scrub bar to jump.

The theme follows `prefers-color-scheme`; the theme button pins light or dark.

## File format (`.flock.gz`)

A gzip stream (decoded in the browser with `DecompressionStream`). Inside:

    bytes 0-3    "FLK2"
    bytes 4-7    header length H, little-endian uint32
    bytes 8-8+H  JSON header, UTF-8
    padding      zeros to the next multiple of 8
    data         binary blobs, each 8-byte aligned

The header's `arrays` table gives each blob's `dtype`, `shape`, `offset` (from the start of the data
section) and `nbytes`; `scales` says how to decode quantised arrays (value / scale). Per step: `t`
(int32), `light` (uint8 / 255), `step_flags` (bit 0 night, bit 1 fog). Per step and sheep: `pos`
(int16, hundredths of a tile, row then column), `heading` (uint16 turns / 65535; the sim's direction
(row, col) = (-cos, sin)), `action` (uint8, names in `meta.action_names`), `flags` (bit 0 alive, bit 1
resting, bit 2 drank, bit 3 grazed), `levels` (uint8 / 200, `meta.drive_names` order), `fat` (uint8 /
255), `wdist` (uint16 / 100, tiles to the nearest water tile centre). Static: `water` (uint8 48 x 48),
`regrow_map` (uint8 / 255, 96 x 96 half-tile cells). `grass`: (snapshots, 96, 96) uint8 / 85, taken at
the steps in `grass_steps`. The header also carries `meta` (label, description, policy, brain, world
options, seeds, git commit, date, DAY, N_TILES), `world` (object kinds with a name table, positions,
presence, fog days, pond centres, mud settings) and `events` (deaths with cause and levels, drinking
bouts, first approaches to water). `scripts/m2_record.py`'s `read_replay` decodes a file in Python.
