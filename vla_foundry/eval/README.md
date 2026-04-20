# Eval

The release supports evaluation with [`lbm_eval`](https://github.com/ToyotaResearchInstitute/lbm_eval). Policies run as a gRPC server on the host; the `lbm_eval` Docker container runs the simulator and connects to the policy over gRPC.

See `vla_foundry/inference/scripts/README.md` (or the [deployment guide](../../docs/guides/deployment.md)) for instructions on launching the policy server that `lbm_eval` connects to.

The end-to-end flow (policy server + Docker sim + results aggregation) is also walked through in [`tutorials/sim_evaluation_tutorial.ipynb`](../../tutorials/sim_evaluation_tutorial.ipynb).
