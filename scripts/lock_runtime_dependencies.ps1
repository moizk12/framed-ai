$ErrorActionPreference = "Stop"
uv pip compile `
    --python-version 3.11 `
    --python-platform x86_64-unknown-linux-gnu `
    --generate-hashes `
    --output-file requirements.lock `
    requirements.runtime.in
