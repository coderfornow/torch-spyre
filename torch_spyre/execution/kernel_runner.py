# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
from torch_spyre._C import launch_kernel, prepare_kernel, launch_jobplan
from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("kernel_runner")


_STICK_BYTES = 128  # DEVICE_ALIGNMENT: one stick = 128 bytes on Spyre hardware


def _normalize_storage_offset(t: torch.Tensor) -> torch.Tensor:
    # executeProgramAsync passes tensor.storage().data_ptr() (storage base) to
    # the hardware kernel, ignoring storage_offset by default.
    #
    # Two cases for non-zero storage_offset:
    #   Stick-aligned (byte_offset % 128 == 0): the C++ fix in executeProgramAsync
    #     passes the byte offset directly to the hardware via tensor_byte_offsets,
    #     so no CPU round-trip is needed.
    #   Non-stick-aligned: the hardware requires LogicalAddress.offset to be
    #     stick-aligned, so sub-stick offsets cannot be expressed natively.
    #     Fix: materialize a fresh zero-offset Spyre tensor via CPU round-trip.
    #     The D2H staging path in spyre_copy_from correctly applies storage_offset,
    #     so the result contains exactly the slice data at element 0.
    #
    #     TODO: Future optimization: use PR #1745's copy_from_d2d for on-device
    #     materialization once offset handling is improved to work with source offsets.
    if t.device.type != "spyre" or t.storage_offset() == 0:
        return t
    byte_offset = t.storage_offset() * t.element_size()
    if byte_offset % _STICK_BYTES == 0:
        return t  # stick-aligned: handled natively by executeProgramAsync
    return t.cpu().to(t.device)  # non-stick-aligned: CPU round-trip fallback


class SpyreUnimplementedRunner:
    def __init__(self, name: str, op: str):
        self.kernel_name = name
        self.op = op

    def run(self, *args, **kw_args):
        raise RuntimeError(
            f"Invoked {self.kernel_name} which contains unimplemented operation {self.op}"
        )


class SpyreSDSCKernelRunner:
    def __init__(self, name: str, code_dir: str):
        self.kernel_name = name
        self.code_dir = code_dir
        self.jobplan = None
        dump_spyre_code = os.environ.get("DUMP_SPYRE_CODE", "0")
        if dump_spyre_code.isdigit() and int(dump_spyre_code) != 0:
            self.jobplan = prepare_kernel(code_dir + "/spyreCodeDir")

    def run(self, *args, **kw_args):
        logger.info("RUN: %s %s", self.kernel_name, self.code_dir)

        with torch.profiler.record_function(f"launch_kernel:{self.kernel_name}"):
            args = tuple(_normalize_storage_offset(a) for a in args)
            if self.jobplan:
                launch_jobplan(self.jobplan, args)
            else:
                launch_kernel(self.code_dir, args)
