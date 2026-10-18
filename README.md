# Missionaries and Cannibals

A Python search-algorithm comparison assignment implementing breadth-first search (BFS), depth-first search (DFS), A*, and strict hill climbing for the river-crossing problem.

Each file is a standalone program with its own state model, successor generation, search implementation, path checks, and benchmark/reporting functions. The original filenames and source contents are preserved.

## Run

Use Python with the standard library; no third-party packages are required. The search checks described below were run with Python 3.14 on Windows.

From the repository directory, run the desired algorithm:

```powershell
python TanYiHeng.py
python CheahYongShun.py
python LamYiHen.py
python LimJunYi.py
```

Each command runs its built-in ten-case benchmark and prints results to the console. Timing calibration repeats searches, so a full program run takes longer than a single search call. The scripts do not require the local dataset PDF or a spreadsheet to run.

## Model and test cases

A state records missionaries on the left bank, cannibals on the left bank, and boat side (`L` or `R`). A crossing costs one move. Safety rules reject groups where cannibals outnumber missionaries while missionaries are present, and moves must respect availability and boat capacity.

The embedded cases use a total population of **5 missionaries and 5 cannibals**, a **boat capacity of 3**, and a goal of `(0, 0, R)`. This is not the usual 3-and-3, capacity-2 configuration.

| Case | Initial state (missionaries left, cannibals left, boat side) |
| --- | --- |
| 1 | (5, 4, R) |
| 2 | (5, 5, L) |
| 3 | (4, 4, L) |
| 4 | (5, 3, R) |
| 5 | (5, 2, R) |
| 6 | (4, 4, R) |
| 7 | (3, 3, R) |
| 8 | (5, 4, L) |
| 9 | (5, 3, L) |
| 10 | (5, 1, L) |

These are the configurations embedded in the source. Their correspondence to the separately supplied PDF has not been checked during import.

## Coding contributions

| Contributor | File | Algorithm |
| --- | --- | --- |
| [Tan Yi Heng](https://github.com/tan0480) | `TanYiHeng.py` | BFS using a FIFO queue, path reconstruction, and result reporting |
| [Cheah Yong Shun](https://github.com/ys012007) | `CheahYongShun.py` | DFS using a LIFO frontier, path reconstruction, and result reporting |
| [Lam Yi Hen](https://github.com/lamyihen77-beep) | `LamYiHen.py` | A* using a priority queue, path cost plus heuristic, and result reporting |
| [Lim Jun Yi](https://github.com/ljunyi0212) | `LimJunYi.py` | Strict-improvement hill climbing, neighbour evaluation, stopping conditions, and result reporting |

The contribution mapping was confirmed by the team representative. Tan Yi Heng performs the repository import and push, while each algorithm commit credits its contributor as author. Import timestamps are not the original development dates.

## Verification and limitations

Before import, each search function was run once for each of the ten embedded cases. Returned solutions were checked with the existing path verifier; transitions in returned partial paths were also checked against the module's successor generator.

| Algorithm | Cases solved | Observed result |
| --- | --- | --- |
| BFS | 10/10 | Crossings: 12, 11, 11, 10, 10, 10, 10, 9, 9, 9 |
| DFS | 10/10 | Crossings: 14, 13, 13, 14, 12, 14, 12, 13, 11, 11 |
| A* | 10/10 | Crossing counts matched BFS for all ten cases |
| Hill climbing | 0/10 | Stopped at a local optimum with no strictly improving neighbour |

Hill climbing has no hidden restart, backtracking, or exhaustive-search fallback. Failure on these cases is reported honestly rather than replaced with another algorithm. DFS and hill climbing do not guarantee an optimal solution.

The checks above reuse the implementation's move generator and are not an independent proof of all puzzle rules or correctness on arbitrary inputs. Full timing/memory benchmarks and their console formatting were not rerun during import; performance depends on the machine and Python version.

Only the four Python files, README, and `.gitignore` are tracked. Dataset PDFs, caches, and generated output are excluded.
