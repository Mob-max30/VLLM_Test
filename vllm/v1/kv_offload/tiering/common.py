# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants for tiered KV offloading."""

from vllm.v1.kv_offload.base import OffloadingCounterMetadata, OffloadingMetricMetadata


class TieringOffloadingMetrics:
    READ_BYTES = "vllm:kv_offload_tiering_read_bytes"
    READ_TIME = "vllm:kv_offload_tiering_read_time"
    WRITE_BYTES = "vllm:kv_offload_tiering_write_bytes"
    WRITE_TIME = "vllm:kv_offload_tiering_write_time"
    PROMOTION_JOB_FAILURES = "vllm:kv_offload_tiering_promotion_job_failures"
    CASCADE_JOB_FAILURES = "vllm:kv_offload_tiering_cascade_job_failures"
    BLOCK_QUERIES = "vllm:kv_offload_tiering_block_queries"
    BLOCK_HITS = "vllm:kv_offload_tiering_block_hits"


TIERING_METRIC_LABELNAMES = ("tier",)


def get_tiering_metric_definitions() -> dict[str, OffloadingMetricMetadata]:
    return {
        TieringOffloadingMetrics.READ_BYTES: OffloadingCounterMetadata(
            documentation=(
                "Total bytes read from secondary tiers into the primary tier."
            ),
            labelnames=TIERING_METRIC_LABELNAMES,
        ),
        TieringOffloadingMetrics.READ_TIME: OffloadingCounterMetadata(
            documentation=(
                "Total time spent reading from secondary tiers into the primary "
                "tier, in seconds."
            ),
            labelnames=TIERING_METRIC_LABELNAMES,
        ),
        TieringOffloadingMetrics.WRITE_BYTES: OffloadingCounterMetadata(
            documentation=(
                "Total bytes written from the primary tier to secondary tiers."
            ),
            labelnames=TIERING_METRIC_LABELNAMES,
        ),
        TieringOffloadingMetrics.WRITE_TIME: OffloadingCounterMetadata(
            documentation=(
                "Total time spent writing from the primary tier to secondary "
                "tiers, in seconds."
            ),
            labelnames=TIERING_METRIC_LABELNAMES,
        ),
        TieringOffloadingMetrics.PROMOTION_JOB_FAILURES: OffloadingCounterMetadata(
            documentation="Number of failed secondary-tier promotion jobs.",
            labelnames=TIERING_METRIC_LABELNAMES,
        ),
        TieringOffloadingMetrics.CASCADE_JOB_FAILURES: OffloadingCounterMetadata(
            documentation="Number of failed secondary-tier cascade jobs.",
            labelnames=TIERING_METRIC_LABELNAMES,
        ),
        TieringOffloadingMetrics.BLOCK_QUERIES: OffloadingCounterMetadata(
            documentation="Number of block lookup queries sent to secondary tiers.",
            labelnames=TIERING_METRIC_LABELNAMES,
        ),
        TieringOffloadingMetrics.BLOCK_HITS: OffloadingCounterMetadata(
            documentation="Number of block lookup hits in secondary tiers.",
            labelnames=TIERING_METRIC_LABELNAMES,
        ),
    }
