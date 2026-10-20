# Time-series branches graph

| Branch | Commit | Nodes | Edges |
| --- | --- | ---: | ---: |
| feat/time-series-arrays | f03e34fca | 21,529 | 41,655 |
| feat/time-series-container | 590242ad7 | 21,530 | 41,657 |
| feat/time-series-controller | 4cbe78ae5 | 21,542 | 41,670 |

Merged: 64,189 nodes and 124,982 edges.

Branch nodes retain their worktree name in the repo attribute. Common imported-module nodes are shared. These are alternative branch snapshots; shared imports do not establish that their experimental APIs call one another.
Direct edges between branch-tagged nodes: 0. Missing endpoints: 0. Self-loops: 321.

Code extraction token cost: 0 input / 0 output. Documentation semantics were not refreshed.

Query from the main checkout:

```sh
graphify query "TimeSeries temporal arrays controller" --graph graphify-out/time-series-branches/graph.json --budget 2500
```

Open graph.html for the aggregated community view. sources.json records source branches and commits.
