#!/usr/bin/env bash
# Launches a scripted "wave" policy that streams sinusoidal joint poses over
# gRPC. Useful for verifying the gRPC stack end-to-end without a trained model.

uv run --group inference python packages/grpc-workspace/src/grpc_workspace/wave_around_policy_server.py
