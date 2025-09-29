\# Architecture



```mermaid

flowchart LR

req\[HTTP request] --> routes\[Flask routes]

routes --> analyzers\[Analysis Pipeline]

analyzers -->|YOLO/CLIP/DeepFace/NIMA| json\[Unified JSON Schema]

json --> echo\[ECHO Identity]

json --> poet\[Critique/Remix Engine]

echo --> report\[Identity/Assignments]

poet --> report

report --> sign\[ProofLens signing]

