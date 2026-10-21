# B4 comparator-accounting correction

The original B3 always-miss comparator inherited the common exact-hit shortcut.
Semantic numerators and denominators were still correct because exact controls
were excluded. The off comparator must nevertheless miss exact controls too.
The corrected baseline does so, and semantic exclusion now uses the query exact
flag rather than its decision cause. Exact query and acceptance counts are
explicit. Original B3 aggregate files are retained under *-b3-original.json.
No model scores, data, labels, veto rules, thresholds or semantic decisions were
changed. This is an accounting bug fix, not tuning. The regression test verifies
0 versus 1 exact accept and identical semantic denominators.
