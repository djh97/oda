# Static Analysis

The final compiler-pinned contract was analyzed with Slither 0.11.6 and solc
0.8.26 using optimization and the IR pipeline. Slither executed 102 detectors
and returned zero findings. The machine-readable result is
`slither_v30.json`; the exact environment is frozen in
`../../requirements-slither.txt`.

The retained run analyzes contract SHA-256
`178c9323754634a8c5c18c4c482cbbcddd9e0278dfa0e47c8c468b47f7af510a`.

Static analysis is not a proof of correctness. It supplements the Foundry
state-transition and failure-path tests and does not establish clinical safety
or production deployment readiness.
