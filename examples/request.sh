#!/bin/sh
set -eu
: "${BOX_API_KEY:?Set the privately shared demo key}"
: "${BOX_API_URL:?Set the HTTPS demo URL, or your local URL}"
curl --fail-with-body --max-time 240 "$BOX_API_URL/v1/predict" \
  -H "X-API-Key: $BOX_API_KEY" \
  -F 'files=@samples/box-1.jpg' \
  -F 'files=@samples/box-2.jpg' \
  -F 'confidence=0.5'
