# data format

`train.py` wants two JSON files, one per model's specialty, each a flat
list of `{"prompt": ..., "answer": ...}` objects. `prompt` gets wrapped
in the model's own chat template, `answer` is what the loss is computed
on.

```json
[
  {"prompt": "...", "answer": "..."},
  {"prompt": "...", "answer": "..."}
]
```

`example_a.json` and `example_b.json` here are just a handful of
entries to show the shape -- swap in your own data for the actual
specialties you're merging. A couple hundred examples per side is
plenty; this is training two scalars per attention head, not the model
itself.
