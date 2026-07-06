# Language Tracking Benchmark Data

This directory stores lightweight benchmark manifests and annotation files for the
optional `locate_tracking` language-guided semantic tracking subsystem.

The checked-in `smoke` fixture is synthetic and tiny. It exists only to validate the
benchmark/evaluation/reporting plumbing without GPU, internet, large videos, or model
weights.

Real benchmark manifests should reference local source videos, raw MOT artifacts, and
manual language-query annotations. Do not use benchmark ground truth during runtime
prediction.
