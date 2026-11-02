"""ariadne.pgtests: PostgreSQL SERIALIZABLE (SSI) probes for similarity reads.

These modules examine how PostgreSQL's serializable isolation (SSI) behaves on
index access methods, in particular pgvector's HNSW, when a transaction runs the
common "check-then-insert" pattern (search for a near duplicate, insert if none
is found). They reproduce four findings:

  locks    : a serializable similarity read takes a relation level SIREAD lock on
             the whole index (pgvector AMs declare ampredlocks=false), so any
             concurrent insertion conflicts.
  hole     : index_insert() runs CheckForSerializableConflictIn BEFORE aminsert,
             and an HNSW insertion searches the graph before linking the new
             element, leaving a window in which a concurrent serializable reader
             misses the conflict and both transactions commit a duplicate.
  hammer   : the same window reached probabilistically, without a debugger.
  generic  : the same window on other AMs with ampredlocks=false (SP-GiST, BRIN,
             hash, gist, gin, bloom), with B-tree as the negative control.
  gist     : built-in GiST (ampredlocks=true) misses a conflict when a search is
             pruned at an internal page and a later insertion enlarges a box.
  keyed    : the same check-then-insert pattern on a scalar key with a B-tree, as
             a control (low abort rate, no relation level lock).
  patched  : build/compare a patched pgvector that repeats the SSI check.

Every module is runnable as  python -m ariadne.pgtests.<name>  from the repo root
and shares options and helpers from ariadne.pgtests.common.

Findings were observed on PostgreSQL 16.15 with pgvector 0.8.7 only.
"""
