# Data

Raw Cranfield files are intentionally **not** stored in this repository.

The benchmark code loads the public `cranfield` collection through
[`ir_datasets`](https://ir-datasets.com/cranfield.html), which handles local caching.
The original collection is also distributed by the University of Glasgow Information
Retrieval Group.

This keeps the repository lightweight, reproducible, and independent of any
course-specific copy of the dataset.
