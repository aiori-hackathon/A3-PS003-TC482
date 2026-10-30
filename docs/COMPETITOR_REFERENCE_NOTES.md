# External project inspection notes

Three user-provided competitor ZIPs were inspected to identify useful research-evaluation patterns. This project does not wholesale-copy their implementations.

- The PyTorch/FedAvg project suggested useful patterns for client-local training, weighted aggregation, experiment history, and IID/non-IID comparisons.
- The regression/federated prediction project suggested per-client and communication-cost reporting.
- The dashboard-oriented project suggested interactive display and repeated-seed evaluation.

The resulting additions in this prototype are per-round/client logs, convergence summaries, IID partition option, a multi-seed runner, and a results-backed dashboard. The primary datasets remain the user's supplied METR-LA and PEMS-BAY files. Verify each algorithm and all results locally before using them in a submission.
