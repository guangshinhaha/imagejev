# Archived: bench v0 results (superseded)

These are the first baseline runs, on the original, smaller benchmark data (bench v0). They are kept
only for provenance. **Do not compare them with current numbers.** The data they were built from had
a bug: Rico screenshots were keyed by a non-unique `request_id`, so most Rico screens were paired
with another screen's image (fixed in PR #80), and the bench v0 splits predate the `val-tasks`
tier. The current results are in `../`.
