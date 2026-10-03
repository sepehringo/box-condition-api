# Included model

`best.pt` is the project's trained YOLOv8 box-condition detector, with classes
`damaged` and `intact`. The API loads it directly and warms each CPU worker;
neither training nor a remote weight download is needed for inference.

- File size: 22,514,905 bytes
- SHA-256: `81647c8839ddcf437fb1e1edea7ea9b914dedb6b332c16fe123b28e014fd82c8`
- Runtime: pinned PyTorch 2.1.0 and Ultralytics 8.0.226, Python 3.11

This is a demonstration model. Outputs are detections rather than a guaranteed
condition assessment. See the main README for measured API performance and
the sample-image attribution in `samples/README.md`.
