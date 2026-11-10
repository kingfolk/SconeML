# M3 complete shader expectations

These two MSL files are hand-authored expectations for the documented uint ABI,
SSA order, guarded flattened index, row/column division/remainder and singleton
zero index. They are not captured or refreshed from compiler output.

`generate_m3.py` embeds them into self-contained source fixtures as complete
`text-equal` assertions. Numeric expectations still come from an independent
Python nested-loop oracle. When changing export intentionally, review the whole
shader contract and edit the golden explicitly, then regenerate source fixtures.
The read-only fixture check never writes a golden or runs the compiler.
