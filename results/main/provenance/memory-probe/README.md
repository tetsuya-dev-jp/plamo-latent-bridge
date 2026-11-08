# Preliminary memory probe

This resource check used the text-conditioned soft-prefix path with 32 training
and eight validation questions, data seed 20261109, batch 16 and one epoch.
It completed two optimizer steps with 9.149 GiB peak allocated GPU memory.
No test answers were evaluated, and its losses are not efficacy results.

The probe ran before the full experiment's source freeze. The exact uncommitted
source snapshot was not recorded; the saved configuration, upstream model
revisions, runtime environment, history and measurement are retained here.
The full comparative experiment has a separate source revision in
`../execution.json`.
