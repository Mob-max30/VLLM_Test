# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECCPUScheduler — CPU offload scheduler delegate.

Owns the mmap region and the offload bookkeeping dicts, and handles the
producer (GPU->CPU offload) and consumer (CPU->GPU reload) scheduler-side
logic directly for the ECCPUConnector.
"""

import threading
from math import ceil
from typing import TYPE_CHECKING, Any

from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    ECRegionContext,
    setup_ec_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    AllocationError,
    ECSharedRegion,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECCPUScheduler:
    """Scheduler delegate for the ECCPUConnector."""

    def __init__(self, vllm_config: "VllmConfig") -> None:
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        self._is_producer: bool = ec_config.is_ec_producer
        self._is_consumer: bool = ec_config.is_ec_consumer

        self._memory_context: ECRegionContext = setup_ec_region(vllm_config)

        # Offload cache: an ec_both instance reuses its own offloaded
        # encodings by serving reloads from _local_encodings / _blocks.
        # _shared_lock guards every access to these two dicts (and the region
        # pin/free calls made against their blocks), so a future NIXL subclass
        # can touch them from a background transfer thread.
        self._local_encodings: dict[str, None] = {}
        self._blocks: dict[str, list[int]] = {}
        self._shared_lock = threading.Lock()

        # Per-step working sets, built and drained on the scheduler thread
        # within a single build_connector_meta; not shared, so not locked.
        # mm_hash -> size_bytes for pending GPU->CPU saves not yet allocated.
        self._encodings_pending_offload: dict[str, int] = {}
        # Locally cached mm_hashes pinned for CPU->GPU re-copy this step.
        self._pending_reload: set[str] = set()

        self._ec_config = ec_config
        self._nixl_enabled: bool = bool(getattr(ec_config, "ec_enable_nixl", False))
        # NIXL fields default to None/empty so the gate-off path is untouched.
        self._data: Any = None
        self._compat_hash: str | None = None
        self._first_in_batch: bool = True
        self._transport: Any = None
        self._producer_session: Any = None
        self._sessions: dict = {}
        self._in_flight: set[str] = set()
        self._tombstones: set[str] = set()
        self._step_completed: set[str] = set()
        self._peer_host: str | None = None
        self._peer_port: int | None = None
        if self._nixl_enabled:
            self._setup_nixl(vllm_config)

    def _setup_nixl(self, vllm_config: "VllmConfig") -> None:
        # Lazy imports keep nixl/zmq off the gate-off path.
        from vllm import envs
        from vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq import (
            ZmqClientTransport,
            ZmqServerTransport,
        )
        from vllm.distributed.ec_transfer.ec_connector.cpu.data.nixl import (
            NixlDataTransport,
        )
        from vllm.distributed.ec_transfer.ec_connector.cpu.protocol import (
            compute_ec_compatibility_hash,
        )
        from vllm.distributed.ec_transfer.ec_connector.cpu.session import (
            ProducerSession,
        )
        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
        from vllm.version import __version__ as VLLM_VERSION

        if NixlWrapper is None or nixl_agent_config is None:
            raise RuntimeError(
                "ec_enable_nixl=True requires NIXL; install the `nixl` package "
                "or set ec_enable_nixl=False."
            )
        engine_id = self._ec_config.engine_id
        assert engine_id is not None
        self._data = NixlDataTransport(
            agent_name=engine_id,
            base_ptr=self._memory_context.region.base_ptr,
            num_blocks=self._memory_context.num_blocks,
            block_size_bytes=self._memory_context.block_size_bytes,
            total_size_bytes=self._memory_context.region.total_size_bytes,
        )
        self._compat_hash = compute_ec_compatibility_hash(
            vllm_version=VLLM_VERSION,
            model=str(vllm_config.model_config.model),
            dtype=str(self._memory_context.dtype),
            block_size_bytes=self._memory_context.block_size_bytes,
        )
        if self._is_producer:
            self._peer_host = envs.VLLM_EC_SIDE_CHANNEL_HOST
            self._peer_port = envs.VLLM_EC_SIDE_CHANNEL_PORT
            self._producer_session = ProducerSession(
                transport=ZmqServerTransport(
                    host=self._peer_host, port=self._peer_port
                ),
                data=self._data,
                region=self._memory_context.region,
                local_encodings=self._local_encodings,
                blocks=self._blocks,
                lock=self._shared_lock,
                compat_hash=self._compat_hash,
            )
            self._producer_session.start()
        if self._is_consumer:
            self._transport = ZmqClientTransport()

    def has_cache_item(self, identifier: str) -> bool:
        if not self._is_consumer:
            return False
        return identifier in self._local_encodings

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        if not self._nixl_enabled:
            return True  # CPU offload never blocks.
        first = self._first_in_batch
        self._first_in_batch = False
        if not self._is_consumer:
            return True
        if first:
            self._poll_step()
        return self._nixl_consumer_admit(request, num_computed_tokens)

    def _nixl_consumer_admit(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        params: dict[str, dict[str, Any]] = (
            getattr(request, "ec_transfer_params", None) or {}
        )
        if not params:
            return True
        pending = False
        for feature in request.mm_features:
            pos = feature.mm_position
            if pos.offset + pos.length <= num_computed_tokens:
                continue
            mm_hash = feature.identifier
            with self._shared_lock:
                is_local = mm_hash in self._local_encodings
                if is_local:
                    if mm_hash not in self._pending_reload:
                        self._memory_context.region.pin(self._blocks[mm_hash])
                    self._pending_reload.add(mm_hash)
            if is_local:
                continue
            if mm_hash in self._in_flight:
                pending = True
                continue
            if mm_hash in self._step_completed:
                pending = True
                continue
            if mm_hash in self._tombstones:
                self._tombstones.discard(mm_hash)
                continue
            info = params.get(mm_hash)
            if info is None:
                continue
            expected = (
                pos.length
                * self._memory_context.hidden_dim
                * self._memory_context.element_size
            )
            if int(info.get("size_bytes", -1)) != expected:
                logger.warning("EC: size mismatch mm_hash=%s; local encode", mm_hash)
                continue
            try:
                self._start_xfer(mm_hash, info, expected)
            except Exception:
                logger.exception("EC: start xfer failed mm_hash=%s", mm_hash)
                continue
            self._in_flight.add(mm_hash)
            pending = True
        return not pending

    def _poll_step(self) -> None:
        import time

        now = time.monotonic()
        all_messages = self._transport.poll()
        for addr, session in list(self._sessions.items()):
            session.poll(all_messages.get(addr, []), now)
        for addr in self._transport.poll_dead():
            self._on_peer_down(addr)
        for session in self._sessions.values():
            self._process_session_results(session)

    def _process_session_results(self, session) -> None:
        r = session.take_results()
        for mm_hash in r.completed:
            self._in_flight.discard(mm_hash)
            self._step_completed.add(mm_hash)
        for mm_hash in r.tombstoned:
            self._in_flight.discard(mm_hash)
            blocks = self._blocks.pop(mm_hash, None)
            if blocks:
                self._memory_context.region.free(blocks)
            self._tombstones.add(mm_hash)
        for mm_hash in r.quarantined:
            self._in_flight.discard(mm_hash)
            self._tombstones.add(mm_hash)
        for mm_hash in r.cancelled:
            self._in_flight.discard(mm_hash)
            blocks = self._blocks.pop(mm_hash, None)
            if blocks:
                self._memory_context.region.free(blocks)
        for mm_hash, block_indices in r.settled:
            self._memory_context.region.free(block_indices)

    def _start_xfer(
        self, mm_hash: str, info: "dict[str, Any]", size_bytes: int
    ) -> None:
        import time
        from math import ceil

        from vllm.distributed.ec_transfer.ec_connector.cpu.session import (
            ConsumerSession,
        )

        n_blocks = max(1, ceil(size_bytes / self._memory_context.block_size_bytes))
        indices = self._fifo_alloc(n_blocks)
        self._blocks[mm_hash] = indices
        addr = (info["peer_host"], int(info["peer_port"]))
        if addr not in self._sessions:
            zmq_conn = self._transport.connect(addr)
            assert self._compat_hash is not None
            self._sessions[addr] = ConsumerSession(
                addr=addr,
                zmq_conn=zmq_conn,
                transport=self._transport,
                data=self._data,
                compat_hash=self._compat_hash,
            )
        deadline = time.monotonic() + 2.0  # CONSUMER_XFER_ACK_TIMEOUT_S
        try:
            self._sessions[addr].start_xfer(mm_hash, indices, deadline)
        except Exception:
            self._memory_context.region.free(self._blocks.pop(mm_hash))
            raise

    def _on_peer_down(self, addr) -> None:
        session = self._sessions.pop(addr, None)
        if session is None:
            return
        session.on_peer_down()
        self._process_session_results(session)
        session.close()
        logger.info("EC: peer down addr=%s", addr)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        feature = request.mm_features[index]
        mm_hash = feature.identifier
        if self._is_producer:
            self._try_offload(mm_hash, feature.mm_position.length)
        if self._is_consumer:
            self._try_reload(mm_hash)

    def _try_offload(self, mm_hash: str, feature_size: int) -> None:
        if mm_hash in self._encodings_pending_offload:
            return
        with self._shared_lock:
            if mm_hash in self._local_encodings:
                return
        size_bytes = (
            feature_size
            * self._memory_context.hidden_dim
            * self._memory_context.element_size
        )
        self._encodings_pending_offload[mm_hash] = size_bytes
        logger.debug("EC: save scheduled mm_hash=%s size_bytes=%d", mm_hash, size_bytes)

    def _try_reload(self, mm_hash: str) -> None:
        with self._shared_lock:
            if mm_hash in self._local_encodings:
                if mm_hash not in self._pending_reload:
                    self._memory_context.region.pin(self._blocks[mm_hash])
                self._pending_reload.add(mm_hash)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECCPUConnectorMetadata:
        meta = ECCPUConnectorMetadata()
        try:
            if self._is_producer:
                meta.saves.update(self._build_saves())
            if self._is_consumer:
                if self._nixl_enabled:
                    self._promote_completed_reads(meta)
                meta.loads.update(self._build_loads())
        except Exception:
            # Drop this step's reload pins so a failure mid-build does not
            # leak them (and block their eviction) until shutdown.
            if self._is_consumer:
                self._drop_reload_pins()
            raise
        finally:
            if self._nixl_enabled:
                self._first_in_batch = True
        return meta

    def _promote_completed_reads(self, meta: ECCPUConnectorMetadata) -> None:
        for mm_hash in self._step_completed:
            if mm_hash in self._blocks:
                meta.loads[mm_hash] = self._blocks[mm_hash]
                with self._shared_lock:
                    self._local_encodings[mm_hash] = None
        self._step_completed.clear()

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, "dict[str, Any] | None"]:
        if not (self._nixl_enabled and self._is_producer):
            return False, None
        params: dict[str, dict[str, Any]] = {}
        with self._shared_lock:
            local_snapshot = set(self._local_encodings)
        for feature in request.mm_features:
            mm_hash = feature.identifier
            if mm_hash not in local_snapshot:
                continue
            size_bytes = (
                feature.mm_position.length
                * self._memory_context.hidden_dim
                * self._memory_context.element_size
            )
            params[mm_hash] = {
                "peer_host": self._peer_host,
                "peer_port": self._peer_port,
                "size_bytes": size_bytes,
            }
        logger.debug(
            "EC: request_finished req_id=%s params=%s", request.request_id, params
        )
        return False, (params or None)

    def shutdown(self) -> None:
        if self._producer_session is not None:
            self._producer_session.stop()
        if self._is_consumer:
            self._drop_reload_pins()
        if self._nixl_enabled:
            self._shutdown_nixl_consumer()
            if self._data is not None:
                try:
                    self._data.deregister()
                except Exception:
                    logger.debug("ec: deregister failed", exc_info=True)
        try:
            self._memory_context.region.cleanup()
        except Exception:
            logger.debug("ec: region cleanup failed", exc_info=True)

    def _shutdown_nixl_consumer(self) -> None:
        for session in list(self._sessions.values()):
            session.close()
        self._sessions.clear()
        if self._transport is not None:
            self._transport.close()

    def _drop_reload_pins(self) -> None:
        """Unpin and forget every block pinned for this step's reloads."""
        with self._shared_lock:
            pending = list(self._pending_reload)
            self._pending_reload = set()
        # Pinned entries are stable; unpin outside the lock (region.unpin is
        # independently thread-safe via the region's own lock).
        for mm_hash in pending:
            blocks = self._blocks.get(mm_hash)
            if blocks is not None:
                self._memory_context.region.unpin(blocks)

    def _fifo_alloc(self, n_blocks: int) -> list[int]:
        try:
            return self._memory_context.region.alloc(n_blocks)
        except AllocationError:
            pass
        with self._shared_lock:
            result = _evict_and_alloc(
                n_blocks,
                self._local_encodings,
                self._blocks,
                self._memory_context.region,
                skip_pinned=True,
            )
        if result is not None:
            return result
        raise AllocationError(
            f"ECSharedRegion exhausted: cannot satisfy {n_blocks} blocks"
        )

    def _build_loads(self) -> dict[str, list[int]]:
        """Re-serve reloads and drop this step's pins."""
        with self._shared_lock:
            pending = list(self._pending_reload)
        # Each mm_hash here is pinned this step, so its cache entry cannot be
        # evicted concurrently — the reads below are safe outside the lock.
        loads: dict[str, list[int]] = {}
        for mm_hash in pending:
            if mm_hash in self._local_encodings:
                loads[mm_hash] = self._blocks[mm_hash]
                logger.debug("EC: CPU->GPU load mm_hash=%s", mm_hash)
        self._drop_reload_pins()
        return loads

    def _build_saves(self) -> dict[str, list[int]]:
        """Allocate blocks for pending encodings and promote them.

        Saves are best-effort: if the region cannot accommodate an encoding
        (even after FIFO eviction), the save is skipped and the encoding is
        simply not offloaded. It remains available in the engine's encoder
        cache for this request; a future request with the same mm_hash will
        schedule a fresh save attempt.
        """
        pending_offload = list(self._encodings_pending_offload.items())
        self._encodings_pending_offload = {}
        saves: dict[str, list[int]] = {}
        # Blocks allocated and promoted earlier in this loop are pinned until
        # the loop ends. Each save's _fifo_alloc may fall back to FIFO
        # eviction, which frees existing _local_encodings entries to make
        # room. Without the pin, a later save could evict and reuse the blocks
        # of an earlier save promoted in this same pass, so the worker would
        # copy two encodings into the same blocks (corruption) and the first
        # save would be silently dropped from the cache. Pinning makes those
        # blocks un-evictable (try_free skips them) for the pass; they are
        # unpinned at the end because the guard is only needed against sibling
        # saves here — afterward they are normal evictable cache entries.
        pinned_this_step: list[list[int]] = []
        try:
            for mm_hash, size_bytes in pending_offload:
                with self._shared_lock:
                    if mm_hash in self._local_encodings:
                        continue  # This mm_hash is already offloaded
                n_blocks = max(
                    1, ceil(size_bytes / self._memory_context.block_size_bytes)
                )
                try:
                    indices = self._fifo_alloc(n_blocks)
                except AllocationError:
                    logger.debug(
                        "EC: region full; skipping offload of mm_hash=%s "
                        "(%d blocks needed). Encoding is computed normally "
                        "and not cached.",
                        mm_hash,
                        n_blocks,
                    )
                    continue
                with self._shared_lock:
                    # Assume the worker's offload succeeds so the encoding is
                    # readable next step. Promote and pin atomically: pinning
                    # in the same critical section stops a concurrent evictor
                    # from reclaiming these blocks before they are pinned.
                    self._blocks[mm_hash] = indices
                    self._local_encodings[mm_hash] = None
                    self._memory_context.region.pin(indices)
                pinned_this_step.append(indices)
                saves[mm_hash] = indices
                logger.debug(
                    "EC: save allocated+promoted mm_hash=%s n_blocks=%d",
                    mm_hash,
                    n_blocks,
                )
        finally:
            # Drop the pass-scoped pins, whether the loop finished or raised:
            # the anti-eviction guard is no longer needed once allocation for
            # the step is done, and leaving blocks pinned would make them
            # permanently un-evictable (a region-space leak). The blocks stay
            # allocated and cached — unpin only clears the do-not-evict flag.
            # pinned_this_step is local and region.unpin is independently
            # thread-safe, so no _shared_lock is needed here.
            for indices in pinned_this_step:
                self._memory_context.region.unpin(indices)
        return saves


def _evict_and_alloc(
    n_blocks: int,
    cache: dict[str, None],
    blocks: dict[str, list[int]],
    region: ECSharedRegion,
    *,
    skip_pinned: bool = False,
) -> list[int] | None:
    """Evict `cache` entries in insertion order until `alloc` succeeds.

    ``cache`` is the ordered set of cached mm_hashes (``dict[str, None]``);
    ``blocks`` maps each mm_hash to its allocated block indices.

    ``skip_pinned=True`` uses ``try_free`` so that blocks held by an active
    NIXL READ pin or a ``_pending_reload`` pin are transparently skipped.
    Caller must hold the shared lock when calling this function.
    Returns allocated block list, or None if all candidates were exhausted.
    """
    for mm_hash in list(cache.keys()):
        indices = blocks[mm_hash]
        if skip_pinned:
            if not region.try_free(indices):
                continue
        else:
            region.free(indices)
        del cache[mm_hash]
        del blocks[mm_hash]
        try:
            return region.alloc(n_blocks)
        except AllocationError:
            continue
    return None
