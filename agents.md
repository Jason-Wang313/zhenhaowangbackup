No interference with any other folders at all.

Before every work session, explicitly tell the user: "I am working in zhenhaowang, which is the correct workspace."

DELETION AND DESTRUCTIVE-CHANGE CONFIRMATION RULE

For every future request that could delete, overwrite, move, replace, reset, or otherwise make files, code, generated outputs, directories, repository history, or other data difficult to recover:

- Never act immediately, even if the request appears clear.
- Ask the user to identify the exact targets and what must be preserved.
- Go back and forth with the user for at least two clarification rounds.
- In the second clarification round, restate the exact deletion/change scope and the protected items, and ask for confirmation.
- Act only after the user confirms that restatement.
- Use the narrowest possible target and verify it immediately before acting.
- Report exactly what changed afterward.

Never infer permission to delete related, neighboring, generated, backup, or environment files from a request to delete one item.

You are the execution agent for the image-restoration synthetic-data quality gate.

Read and obey the existing AGENTS.md first.

WORKSPACE BOUNDARY
Only work inside /inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang unless the user explicitly asks otherwise. Do not inspect, modify, or run commands in other users' folders.


Do not inspect, modify, delete, or execute commands in any other user's directory. Do not reuse previous project paths, datasets, checkpoints, environments, manifests, or results unless directly verified inside this workspace.

Assume this is a fresh start. Treat every dataset, checkpoint, download, environment, manifest, and result as UNKNOWN until checked directly.

OBJECTIVE

Complete a verified small synthetic image-restoration quality gate.

Do not start VLM training, VLM evaluation, or a 100k dataset run.

Target:

- 100 clean source images.
- AgenticIR noise: 100 outputs.
- AgenticIR blur: 100 outputs.
- AgenticIR haze: 100 outputs.
- AgenticIR rain: 100 outputs.
- GenDeg low-light L1-L5: 500 outputs.
- Total degraded outputs: 900.

Do not fake unavailable experiments. Do not silently substitute unrelated implementations or unmatched source/depth pairs.

DATASET SCOPE

The full AgenticIR/MiO100 dataset is not automatically required.

Minimum requirements:

- Noise, blur, and rain: 100 clean images.
- Haze: the same 100 clean images plus 100 matching depth maps.
- GenDeg low-light: verified gendeg.ckpt.

Prefer a verified 100-image clean/depth subset if available. Download the complete AgenticIR dataset only if a smaller verified subset cannot be obtained or the user explicitly requests the full dataset.

GenIRData, LSDIR, and FoundIR are not required for this 900-output pilot. Do not download them unless the user explicitly expands the scope and the storage plan is updated.

Never claim haze is complete without matching depth maps. Never reuse an example depth map for unrelated images.

EXECUTION LOOP

Do not wait passively for downloads.

At the beginning and after every meaningful milestone:

1. Check all download processes, logs, file sizes, and disk space.
2. Classify each transfer as ACTIVE, COMPLETE, FAILED, or UNKNOWN.
3. Start every experiment whose required inputs are available and verified.
4. Leave healthy downloads running while independent experiments execute.
5. Recheck all downloads after each independent experiment milestone.
6. When a blocked asset finishes, verify it immediately.
7. Start its dependent experiment without waiting for another user prompt.
8. Continue until all unblocked and newly unblocked work is finished.

Record each loop in a durable journal with:

- timestamp;
- process ID;
- source URL or file ID;
- destination path;
- current file size;
- download status;
- log tail and exit status;
- experiment started/completed;
- next action;
- remaining blockers.

FIRST ACTIONS

1. Read AGENTS.md completely.
2. Print the current working directory.
3. Verify it is exactly:
   /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
4. Inspect the workspace tree, Git status, scripts, data, checkpoints, logs, and reports.
5. Check available disk space.
6. Create or update a project journal.
7. Create a status ledger using:
   VERIFIED, ACTIVE, COMPLETE, FAILED, UNKNOWN, MISSING.
8. Do not overwrite existing files until their purpose and ownership are understood.

DOWNLOAD RULES

Use the network-enabled environment for network downloads. If a GPU environment is offline, do not download there.

For every download, record:

- source URL or file ID;
- destination path;
- process ID and launch time;
- current size and timestamp;
- expected size if available;
- log path;
- final file type;
- SHA-256.

Rules:

- A growing .part file is active progress; do not restart it.
- Do not launch duplicate downloads.
- Preserve failed logs and partial files.
- Resume partial transfers instead of restarting from zero.
- Do not delete partial files unless corruption is proven.
- Do not treat HTML, login pages, error pages, or partial archives as valid files.
- Verify final size, file type, archive integrity, and SHA-256 before use.
- Never download from an offline GPU environment.

AGENTICIR

Locate the official AgenticIR implementation and record its repository path and Git revision.

Record:

- Python version;
- PyTorch version;
- torchvision version;
- OpenCV version;
- BasicSR version;
- CUDA/runtime information if used.

Before batch synthesis:

1. Run the official one-image smoke test for noise, blur, haze, and rain.
2. Use the official example image only with its matching example depth map.
3. Save smoke outputs separately.
4. Verify output readability, non-zero size, dimensions, deterministic behavior, and hashes.
5. Fix dependency problems in an isolated environment or runtime shim.
6. Do not silently change official degradation semantics.

For the final AgenticIR batch:

- select exactly 100 deterministic clean/depth pairs;
- create the source-selection manifest before synthesis;
- generate exactly one isolated degradation per source for noise, blur, haze, and rain;
- preserve seed, parameters, operation order, paths, code revision, and hashes.

Each manifest row must contain at least:

{
  "source_id": "stable-id",
  "source": "/absolute/path/inside/workspace",
  "depth": "/absolute/path/inside/workspace-or-null",
  "type": "noise|blur|haze|rain",
  "seed": 0,
  "parameters": {},
  "operation_order": ["type"],
  "output": "/absolute/path/inside/workspace",
  "source_sha256": "...",
  "output_sha256": "...",
  "code_revision": "..."
}

Accept the AgenticIR batch only when there are exactly 400 readable outputs and 400 valid manifest rows.

GENDEG

Do not use GenDeg until gendeg.ckpt is verified.

Required sequence:

1. Locate the completed generator checkpoint only inside the approved workspace.
2. Confirm it is not a .part file.
3. Confirm it is a checkpoint, not HTML or an error page.
4. Confirm its size is stable.
5. Compute and record SHA-256.
6. Copy it to the expected checkpoints/gendeg.ckpt path.
7. Keep any optional structure-correction checkpoint separately named.
8. Read the official environment configuration.
9. Use an isolated environment.
10. Run a one-image low-light smoke test.
11. Verify that the smoke output is readable and valid.
12. Generate exactly five levels, L1-L5, for the same 100 source IDs.
13. Record checkpoint hash, source ID, level, parameters, output path, and output checksum.

Accept the GenDeg batch only when there are exactly 500 readable outputs and 500 valid manifest rows.

VISUAL QUALITY CONTROL

Visual inspection is mandatory. Do not accept results based only on scripts, counts, or hashes.

Create aligned contact sheets showing:

- clean;
- noise;
- blur;
- haze;
- rain;
- low-light L1;
- low-light L2;
- low-light L3;
- low-light L4;
- low-light L5.

Inspect every output at contact-sheet scale for:

- blank or corrupted files;
- clipping;
- unnatural color shifts;
- excessive or invisible degradation;
- implausible rain streaks;
- unnatural haze;
- blur destroying all structure;
- low-light levels that are not visibly ordered;
- wrong source/depth pairing;
- scene-content changes;
- inconsistent dimensions or borders.

Inspect every suspicious thumbnail at full resolution. Record rejected files and reasons in a QA report.

Every output must appear in a contact sheet. It is not necessary to open all 900 files individually at full resolution, but every suspicious case must receive full-resolution inspection.

Do not scale to 100k or start VLM training until the user has inspected the contact sheets and approved the quality.

RECOMMENDED PROJECT LAYOUT

Use a clear layout inside the approved workspace:

data/clean/
data/agenticir/noise/
data/agenticir/blur/
data/agenticir/haze/
data/agenticir/rain/
data/gendeg/low_light_L1/
data/gendeg/low_light_L2/
data/gendeg/low_light_L3/
data/gendeg/low_light_L4/
data/gendeg/low_light_L5/
manifests/
reports/contact_sheets/
reports/qa/
logs/

Do not hardcode paths outside the approved workspace.

COMPLETION CRITERIA

Do not declare completion until all of these are VERIFIED:

- exactly 100 clean source images selected;
- exactly 400 AgenticIR outputs generated;
- exactly 500 GenDeg outputs generated;
- valid source/depth matching for haze;
- all files readable and non-empty;
- all source/output/checkpoint hashes recorded;
- all manifests contain the expected number of valid rows;
- contact sheets cover every output;
- suspicious outputs were inspected at full resolution;
- QA report records failures and acceptance decisions;
- download logs and environment/version records are preserved;
- no required download remains ACTIVE, FAILED, or UNKNOWN;
- no required .part file is treated as final;
- user inspection and approval are recorded;
- no VLM training or evaluation was launched;
- GenIRData, LSDIR, and FoundIR were not downloaded.

Begin now.

Read AGENTS.md, verify the workspace path, perform the initial data/download/process/disk audit, and then execute every experiment that is not blocked.

Continue automatically through newly unblocked work. Recheck downloads after every milestone. Do not stop merely because one download is slow.

GENDEG VISUAL-CALIBRATION RECORD (CURRENT APPROVED RULE)

The current visually approved five-level smoke rule is a two-stage process:

1. Generate the official GenDeg sample using the verified local checkpoint and
   the parameters below.
2. Apply the final review mapping below. The raw GenDeg images remain preserved
   separately for audit and are never overwritten by the review remap.

Generator/runtime parameters:

- checkpoint: `checkpoints/gendeg.ckpt`
- GenDeg implementation revision: `24656bfd4458e87f0efefae4692e7ad48dd4ca44`
- prompt: `a photograph in low-light conditions`
- output resolution: 512 x 512
- diffusion steps: 100
- classifier-free guidance scale: 7.5
- structure/conditioning scale: 1.5
- smoke seed base: 2026080400
- full-batch seed for source index `i`: `2026080400 + i * 10`
- official conditioning levels:
  - L1: `(mu=0.10, sigma=0.05)`
  - L2: `(mu=0.22, sigma=0.10)`
  - L3: `(mu=0.34, sigma=0.16)`
  - L4: `(mu=0.48, sigma=0.23)`
  - L5: `(mu=0.62, sigma=0.30)`

Initial exposure calibration targets used after official sampling:

- original calibrated L1: 75% of clean luminance
- original calibrated L2: 60% of clean luminance
- original calibrated L3: 45% of clean luminance
- original calibrated L4: 30% of clean luminance
- original calibrated L5: 18% of clean luminance

Current final review mapping:

- new L1: clean source copied byte-for-byte
- new L2: original calibrated L3
- new L3: original calibrated L4
- new L4: original calibrated L5
- new L5: original calibrated L5 darkened again

New L5 calibration rule:

- initial target is 10% of clean luminance;
- target is additionally limited to 55% of the old L5 and 55% of new L4;
- after rendering, if measured new-L5 luminance is not below 65% of new-L4
  luminance, halve the target and repeat, up to six tightening passes;
- this is a visual-quality floor/ordering rule, not a claim that GenDeg's
  official `(mu, sigma)` values directly equal brightness percentages.

Measured approved smoke reference (`data/clean/001.png`):

- clean/new L1 luminance: 92.95
- new L2 luminance: 38.90
- new L3 luminance: 26.70
- new L4 luminance: 16.01
- new L5 luminance: 8.41
- new L5 is approximately 9% of clean luminance and 52.5% of new L4

The remap implementation is `smoke/apply_gendeg_level_remap.py`. Any new
non-tiger validation sample must use the same generator, calibration, mapping,
and visual inspection criteria unless the user explicitly approves a change.
