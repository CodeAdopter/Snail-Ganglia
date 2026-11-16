<div align="center">
  <img src="assets/logo.svg" alt="Ganglia logo" width="128">
  <h1>Ganglia</h1>
  <p>Predicts the chess move a person plays, and how long they think</p>
  <p>
    <a href="https://github.com/CodeAdopter/Snail-Ganglia/issues"><img src="https://img.shields.io/github/issues/CodeAdopter/Snail-Ganglia?label=issues&color=1c6f52" alt="Open issues"></a>
    <a href="https://github.com/CodeAdopter/Snail-Ganglia/commits/main"><img src="https://img.shields.io/github/last-commit/CodeAdopter/Snail-Ganglia?label=last%20commit&color=1c6f52" alt="Last commit"></a>
    <a href="LICENSE"><img src="https://img.shields.io/github/license/CodeAdopter/Snail-Ganglia?label=license&color=1c6f52" alt="License"></a>
    <a href="https://huggingface.co/buckets/AngeloWolff/Ganglia"><img src="https://img.shields.io/badge/nets-Hugging%20Face-1c6f52" alt="The nets are on Hugging Face"></a>
  </p>
  <p>
    <a href="https://ganglia-chess.vercel.app">Website</a> ·
    <a href="#nets">Nets</a> ·
    <a href="#build">Build</a> ·
    <a href="#data">Data</a> ·
    <a href="#train-a-move-net">Train</a> ·
    <a href="#score-a-net">Score</a> ·
    <a href="https://github.com/CodeAdopter/Snail-Chess">Snail Chess, the engine</a>
  </p>
  <sub>Windows and Linux · NVIDIA GPU, RTX 30 series or newer · <a href="LICENSE">MIT</a> · written in <a href="https://github.com/CodeAdopter/Snail">Snail</a></sub>
  <br>
  <sub>
    Snail projects:
    <a href="https://github.com/CodeAdopter/Snail">Snail</a> ·
    <a href="https://github.com/CodeAdopter/Snail-Chess">Snail Chess</a> ·
    <a href="https://github.com/CodeAdopter/Snail-Chess-Gui">Snail Chess GUI</a> ·
    <b>Ganglia</b> ·
    <a href="https://github.com/CodeAdopter/Snail-Db">Snail Db</a>
  </sub>
</div>

---

A transformer that predicts which move a human plays in a chess position and how long they
think about it, trained on Lichess games. It powers the Human Mode of Snail Chess 2.0.

Ganglia is a trainer written in [Snail](https://github.com/CodeAdopter/Snail). Its GPU kernels are
written in Snail too and compiled with the program, so it needs neither PyTorch nor the CUDA
toolkit, only an NVIDIA driver. It builds and runs on Windows and Linux.

## Nets

| Net | Size | Parameters | Bullet top-1 | Blitz top-1 | Rapid top-1 |
|---|---|---|---|---|---|
| v1 | 8 layers x 256 | 8,716,839 | 47.01% | 55.98% | 55.61% |
| v2 | 12 layers x 320, smolgen | 22,784,479 | 52.36% | 56.85% | 56.40% |
| v3 small | 9 layers x 192, smolgen | 8,732,829 | 54.66% | 56.26% | 55.78% |

Top-1 against the moves played in Lichess games of August 25-31, 2026 (2.69 million positions),
with the think time hidden. v3 small was trained with Ganglia. v1 and v2 were trained in the
PyTorch project that Ganglia was ported from, and never saw a bullet game. There is also a small
think-time net (349,056 parameters) that predicts the think time alone, for a program that
chooses its moves another way.

The nets are not in this repository. They are in the Hugging Face bucket
[AngeloWolff/Ganglia](https://huggingface.co/buckets/AngeloWolff/Ganglia), which holds the whole
project. To fetch them into `weights/`:

```
hf sync hf://buckets/AngeloWolff/Ganglia/weights weights
```

## Build

You need:

- `snailc`, the Snail compiler, from [github.com/CodeAdopter/Snail](https://github.com/CodeAdopter/Snail),
  with the `base` and `cuda` libraries that come with it. Use a recent one: the kernels need
  `warp_load_matrix` and `shared_copy_async`.
- clang 19 or later, which Snail needs too.
- An NVIDIA GPU of the RTX 30 series or newer, with its driver.

```
snailc build -c release
```

writes `build/ganglia.exe` (`build/ganglia` on Linux). `ganglia` alone lists every command and
option.

## Data

Training reads positions made from Lichess games in two steps.

1. **Game files.** A month of rated games from
   [database.lichess.org](https://database.lichess.org/#standard_games), converted to compact game
   files by the `extract` tool. `extract` is not in this repository yet.
2. **Positions.** `tools/positions` turns game files into the 512-byte positions the net reads
   (`src/kernels/records.snail` has the layout). It uses the board and move generator of the
   [Snail Chess engine](https://github.com/CodeAdopter/Snail-Chess): set the two paths in
   `tools/positions/snail.toml` to the engine's `src` and to Snail's `lib/chess`, then run
   `snailc build -c release` in that folder.

Games before the cutoff date are for training, and games from it on are the test set. Cut the
test file once:

```
positions <games dir> val --out test.bin
```

The cutoff is 2026-08-25 unless `--cutoff YYYY-MM-DD` says otherwise; give `ganglia train` the
same option. `positions` reads all the game files into memory, about 30 GB for a month.

## Train a move net

```
ganglia train --run runs/my-net --val test.bin --games <games dir> --positions-exe tools/positions/build/positions.exe
```

trains a net of v1's shape on 100 million positions. `ganglia train` starts `positions.exe` itself
and reads its stream. The run's folder gets `log.txt`, `last.gnet`, `best.gnet` (the net at its
best blitz top-1) and `evals.tsv` (each evaluation per time class and 100-point rating bin).

| Option | Default | |
|---|---|---|
| `--width`, `--layers`, `--heads` | 256, 8, 8 | The net's shape |
| `--smolgen 1` | off | Attention biases generated from the whole position |
| `--bullet 1` | off | Trains on bullet games too, with rating and clock tables of their own |
| `--positions N` | 100,000,000 | The length of the run |
| `--batch`, `--lr`, `--wd` | 1024, 0.001, 0.01 | Batch size, peak learning rate, weight decay |
| `--seed N` | 1 | The starting weights and the order of the positions |
| `--resume <last.gnet>` | | Continues a run that stopped, with its optimizer and schedule |
| `--init <net.gnet>` | | Starts a new schedule from a trained net's weights |
| `--records <file>` | | Trains on a file of positions read in order (`positions <games dir> train --limit N > file`), so that runs being compared see the same positions |

For a net of your own, keep the heads 32 wide (`--width` = 32 x `--heads`) and the width at most
512. Those shapes train in bf16 on the tensor cores; any other falls back to a slower f32 pass.
v3 small, for example, was trained with

```
ganglia train --run runs/v3-small --val test.bin --games <games dir> --positions-exe <positions.exe> --width 192 --layers 9 --heads 6 --smolgen 1 --bullet 1 --positions 400000000
```

which took five and a half hours on an RTX 5080.

| Shape | Positions a second, RTX 5080 | GPU memory at batch 1024 |
|---|---|---|
| v1, 8 x 256 | 19,590 | 6.5 GB |
| v3 small, 9 x 192, smolgen | 20,800 | |
| v2, 12 x 320, smolgen | 8,320 | 12 GB |

A run ends with exit code 3 when no step finishes for `--stall-seconds` (300); continue it with
`--resume`.

## Score a net

```
ganglia eval <net.gnet> test.bin
```

prints top-1, top-3 and the loss on the move played for each time class, and their means over
100-point rating bins. `--stride 10` scores every 10th position, as the evaluations during
training do.

## The small think-time net

```
ganglia timedata --teacher <net.gnet> --out train.tset --games <games dir> --positions-exe <positions.exe>
ganglia timedata --teacher <net.gnet> --out test.tset --test test.bin
ganglia timetrain --run runs/time --train train.tset --val test.tset --distill 1 --distill-end 0
ganglia timeeval --net runs/time/best.tnet --val test.tset
```

`timedata` writes time sets: positions with a move net's think-time prediction beside each.
`timetrain` trains the small net on them and writes `last.tnet`, `best.tnet` and `report.json`.
With `--distill 1 --distill-end 0` the target starts as the move net's prediction and ends as the
real think time. `timeeval` scores a net against real think times, in bits per move. The net in
`weights/tnets` was trained on 296 million positions at `--lr 0.003`, which takes under four
minutes.

## Quantized nets

```
ganglia quantize <net.gnet|net.tnet> <out> --bits 8
```

stores a net in fewer bits: 16 for half floats, 8 down to 2 for integer weights. Every command
reads the result as the net it was made from. At 8 bits a net is about a quarter of its size at
no cost that can be measured; `weights/quantized/README.md` has the measurements.

## Checks

Run these after changing a kernel:

```
ganglia check weights/gnets/v1-8x256.gnet weights/gnets/checks/v1-forward.ref
ganglia gradcheck weights/gnets/v1-8x256.gnet weights/gnets/checks/v1-grad.ref
ganglia halfcheck weights/gnets/v1-8x256.gnet test.bin
```

`check` and `gradcheck` compare the f32 forward pass and every gradient with float64 values from
the PyTorch nets (the `.ref` files, for v1 and v2). `halfcheck` compares the bf16 pass with the
f32 one, `gemmcheck` and `mmcheck` the tensor cores' products with f32, and `timecheck` the
think-time net on the GPU with the CPU.

## Layout

| Path | Contents |
|---|---|
| `src/main.snail` | The commands and their options |
| `src/model.snail`, `src/checkpoint.snail` | The net's tensors and the `.gnet` format |
| `src/forward.snail`, `src/backward.snail`, `src/half.snail` | The f32 passes, and the bf16 pass that `eval` and `train` run |
| `src/train.snail`, `src/evaluation.snail` | `train` and `eval` |
| `src/timenet.snail`, `src/timeset.snail`, `src/timetrain.snail`, `src/timeeval.snail` | The small think-time net |
| `src/quantize.snail`, `src/quantized.snail` | Quantized nets |
| `src/check.snail`, `src/bench.snail` | The checks and `profile` |
| `src/kernels/` | The GPU kernels |
| `tools/positions/` | Training and test positions from game files |
| `weights/` | The nets, on Hugging Face only |

## License

Copyright 2026 Angelo Wolff. Ganglia, its code and its nets, is released under the MIT License;
see [LICENSE](LICENSE). v1 and v2 were first released under CC0 with the PyTorch project.
