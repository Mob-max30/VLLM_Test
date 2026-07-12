# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.v1.kv_offload.base import CanonicalKVCacheRef, KVHeadRegion
from vllm.v1.kv_offload.cpu.gpu_worker import _ref_copy_expansion


def _nhd_ref() -> CanonicalKVCacheRef:
    # block_size=4, 2 heads/worker, head_dim=64, int8: K/V of 512B, frag=128B
    return CanonicalKVCacheRef(
        tensor_idx=0,
        page_size_bytes=1024,
        head_regions=(
            KVHeadRegion(0, 128, 4, 2),
            KVHeadRegion(512, 128, 4, 2),
        ),
    )


def test_nhd_expansion_worker2():
    src, dst, sizes = _ref_copy_expansion(_nhd_ref(), True, num_slots=4, slot=2)
    k_dst = [256, 768, 1280, 1792]
    assert src.tolist() == [0, 128, 256, 384, 512, 640, 768, 896]
    assert dst.tolist() == k_dst + [2048 + o for o in k_dst]
    assert sizes.tolist() == [128] * 8
    assert sizes.sum() == 1024


def test_hnd_expansion():
    ref = CanonicalKVCacheRef(
        tensor_idx=0,
        page_size_bytes=1024,
        head_regions=(
            KVHeadRegion(0, 512, 1, 2),
            KVHeadRegion(512, 512, 1, 2),
        ),
    )
    src, dst, sizes = _ref_copy_expansion(ref, True, num_slots=4, slot=1)
    assert src.tolist() == [0, 512]
    assert dst.tolist() == [1 * 512, 4 * 512 + 1 * 512]
    assert sizes.tolist() == [512, 512]


def test_opaque_ref_single_slot():
    ref = CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=9728)
    src, dst, sizes = _ref_copy_expansion(ref, True, num_slots=4, slot=3)
    assert src.tolist() == [0]
    assert dst.tolist() == [3 * 9728]
    assert sizes.tolist() == [9728]


def test_load_direction_swaps_offsets():
    store = _ref_copy_expansion(_nhd_ref(), True, num_slots=4, slot=2)
    load = _ref_copy_expansion(_nhd_ref(), False, num_slots=4, slot=2)
    assert np.array_equal(store[0], load[1])
    assert np.array_equal(store[1], load[0])
    assert np.array_equal(store[2], load[2])


def test_slots_tile_region_without_overlap():
    # All slots' destinations together cover [0, num_slots * page) exactly once
    all_dst = []
    for slot in range(4):
        _, dst, sizes = _ref_copy_expansion(_nhd_ref(), True, num_slots=4, slot=slot)
        all_dst += [(d, d + s) for d, s in zip(dst.tolist(), sizes.tolist())]
    all_dst.sort()
    assert all_dst[0][0] == 0 and all_dst[-1][1] == 4 * 1024
    assert all(a[1] == b[0] for a, b in zip(all_dst, all_dst[1:]))
