# Copyright 2023 NVIDIA CORPORATION
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

from itertools import chain
import random
import unittest

import torch
import torch.nn as nn

SKIP_TEST = None
try:
    from apex.contrib.openfold_triton.fused_adam_swa import AdamMathType, FusedAdamSWA
except ImportError as e:
    SKIP_TEST = e


# Stochastic weight average (SWA) reference code from
# https://github.com/mlcommons/hpc_results_v3.0/blob/350e46f7/NVIDIA/benchmarks/openfold/implementations/pytorch/openfold/swa.py#L21-L70
class AlphaFoldSWA(nn.Module):
    """AlphaFold SWA (Stochastic Weight Averaging) module wrapper."""

    def __init__(self, alphafold: nn.Module, enabled: bool, decay_rate: float) -> None:
        super(AlphaFoldSWA, self).__init__()
        if enabled:
            self.averaged_model = torch.optim.swa_utils.AveragedModel(
                model=alphafold,
                avg_fn=swa_avg_fn(decay_rate=decay_rate),
            )
            self.enabled = True
        else:
            self.averaged_model = None
            self.enabled = False

    def update(self, alphafold: nn.Module) -> None:
        if self.enabled:
            self.averaged_model.update_parameters(model=alphafold)

    def forward(self, batch):
        if not self.enabled:
            raise RuntimeError("AlphaFoldSWA is not enabled")
        return self.averaged_model(batch)


class swa_avg_fn:
    """Averaging function for EMA with configurable decay rate
    (Supplementary '1.11.7 Evaluator setup')."""

    def __init__(self, decay_rate: float) -> None:
        self._decay_rate = decay_rate

    def __call__(
        self,
        averaged_model_parameter: torch.Tensor,
        model_parameter: torch.Tensor,
        num_averaged: torch.Tensor,
    ) -> torch.Tensor:
        # for decay_rate = 0.999:
        # return averaged_model_parameter * 0.999 + model_parameter * 0.001
        # avg * 0.999 + m * 0.001
        # 999*avg/1000 + m/1000
        # (999*avg + avg - avg)/1000 + m/1000
        # (1000*avg - avg)/1000 + m/1000
        # 1000*avg/1000 - avg/1000 + m/1000
        # avg + (m - avg)/1000
        # avg + (m - avg)*0.001
        return averaged_model_parameter + (model_parameter - averaged_model_parameter) * (
            1.0 - self._decay_rate
        )


@unittest.skipIf(SKIP_TEST, f"Skip testing FusedAdamSWA: {SKIP_TEST}")
class FusedAdamSWATestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # Ensure deterministic tests.
        self.seed = 19260817
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        # Hyper parameters.
        self.device = torch.device("cuda:0")
        self.compute_dtype = torch.float32
        self.state_dtype = torch.float64
        self.atol = 1e-5  # Default: 1e-8, raise error at 1e-6 for FP32 compute and FP64 state.
        self.rtol = 1e-4  # Default: 1e-5
        self.lr = 1e-1
        self.bias_correction = True
        self.beta1, self.beta2 = 0.9, 0.999
        self.eps = 1e-6
        self.adam_math_mode = AdamMathType.PyTorchAdam
        self.weight_decay = 1e-3  # PyTorchAdam impl will fail non-zero weight decay.
        self.amsgrad = False
        self.adam_step = 1900
        self.swa_decay_rate = 0.9
        self.swa_n_averaged = 1

        # Tensors and gradients to be processed.
        self.num_param_tensors = 32
        self.state_params = [
            torch.empty(
                random.randint(128, 2048), device=self.device, dtype=self.state_dtype
            ).uniform_(-5, 5)
            for _ in range(self.num_param_tensors)
        ]
        self.compute_dtypes = [
            self.compute_dtype if random.uniform(0.0, 1.0) <= 0.5 else self.state_dtype
            for _ in range(self.num_param_tensors)
        ]
        self.grads = [
            torch.empty_like(p, dtype=d).uniform_(-5, 5)
            for d, p in zip(self.compute_dtypes, self.state_params)
        ]
        self.moments = [torch.empty_like(p).uniform_(-5, 5) for p in self.state_params]
        self.velocities = [torch.empty_like(p).uniform_(0, 10) for p in self.state_params]

        # Dummy model.
        self.dummy_model = torch.nn.Module()
        for i, p in enumerate(self.state_params):
            self.dummy_model.register_parameter(f"param_{i}", torch.nn.Parameter(p.clone()))

    def _get_ground_truth(self):
        # PyTorch Adam, MLPerf-HPC SWA implementation.
        compute_params_gt = [
            p.clone().to(d) for d, p in zip(self.compute_dtypes, self.state_params)
        ]
        state_params_gt = list(self.dummy_model.parameters())
        swa_model = AlphaFoldSWA(self.dummy_model, enabled=True, decay_rate=self.swa_decay_rate)
        swa_params_gt = list(swa_model.parameters())
        optimizer = torch.optim.Adam(
            state_params_gt,
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )
        moments_gt, velocities_gt, steps_gt = [], [], []
        for i, p in enumerate(optimizer.param_groups[0]["params"]):
            s = optimizer.state[p]
            self.assertTrue(self.moments[i].shape == self.velocities[i].shape == p.shape)
            s["step"] = torch.tensor(self.adam_step, dtype=self.state_dtype, device=self.device)
            s["exp_avg"] = self.moments[i].clone()
            s["exp_avg_sq"] = self.velocities[i].clone()
            steps_gt.append(s["step"])
            moments_gt.append(s["exp_avg"])
            velocities_gt.append(s["exp_avg_sq"])
        for p, g in zip(state_params_gt, self.grads):
            p.grad = g.clone().to(self.state_dtype)
        optimizer.step()
        swa_model.averaged_model.n_averaged.copy_(self.swa_n_averaged)
        swa_model.update(self.dummy_model)
        for c, s in zip(compute_params_gt, state_params_gt):
            c.detach().copy_(s.detach().to(c.dtype))
        swa_n_averaged = swa_model.averaged_model.n_averaged.item()

        return (
            state_params_gt,
            compute_params_gt,
            swa_params_gt,
            swa_n_averaged,
            moments_gt,
            velocities_gt,
            steps_gt,
        )

    def _get_eager_mode_results(self, return_optimizer=False, capturable=False,):
        # Fused AdamSWA, all at once.
        state_params_test = [torch.nn.Parameter(p.clone()) for p in self.state_params]
        compute_params_test = [
            p.clone().to(d) for d, p in zip(self.compute_dtypes, self.state_params)
        ]
        swa_params_test = [p.clone() for p in self.state_params]
        fused_optimizer = FusedAdamSWA(
            params=state_params_test,
            compute_params=compute_params_test,
            swa_params=swa_params_test,
            swa_decay_rate=self.swa_decay_rate,
            lr=self.lr,
            bias_correction=self.bias_correction,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            adam_math_mode=self.adam_math_mode,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
            capturable=capturable,
        )
        moments_test, velocities_test, steps_test, swa_n_averaged_test = [], [], [], []
        for i, p in enumerate(fused_optimizer.param_groups[0]["params"]):
            s = fused_optimizer.state[p]
            self.assertTrue(self.moments[i].shape == self.velocities[i].shape == p.shape)
            s["exp_avg"] = self.moments[i].clone()
            s["exp_avg_sq"] = self.velocities[i].clone()
            s["step"] = torch.tensor(self.adam_step, dtype=torch.int32, device=self.device)
            s["swa_n_averaged"] = torch.tensor(
                self.swa_n_averaged, dtype=torch.int32, device=self.device
            )
            moments_test.append(s["exp_avg"])
            velocities_test.append(s["exp_avg_sq"])
            steps_test.append(s["step"])
            swa_n_averaged_test.append(s["swa_n_averaged"])
        for c, g in zip(compute_params_test, self.grads):
            c.grad = g.clone()
        fused_optimizer.step()

        ret = [fused_optimizer, ] if return_optimizer else []
        ret += [
            state_params_test,
            compute_params_test,
            swa_params_test,
            swa_n_averaged_test,
            moments_test,
            velocities_test,
            steps_test,
        ]
        return tuple(ret)

    def _get_cuda_graph_mode_results(self):
        # Construct and warmup on a side CUDA stream.
        graph = torch.cuda.CUDAGraph()
        side_stream = torch.cuda.Stream()

        # TODO: Remove those debugs.
        # graph.enable_debug_mode()
        # debug_path = "/home/scratch.davidli_gpu_1/Projects/sandbox/cuda_graph_dumps/fused_adam_swa_prebuilt_buffer.dot"

        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            (
                fused_optimizer,
                state_params_test,
                compute_params_test,
                swa_params_test,
                swa_n_averaged_test,
                moments_test,
                velocities_test,
                steps_test,
            ) = self._get_eager_mode_results(return_optimizer=True, capturable=True)

            # Ensure warmup is compatible w/ DDP: run in eager mode for at least 11 steps.
            for _ in range(11):
                fused_optimizer.step()
        torch.cuda.current_stream().wait_stream(side_stream)

        # NOTE: We don't release pointer buffers here. Because we know there will be no new tensor
        # allocation in the capture region and previous tensor memories will be persistent. This
        # might not be the case in full OpenFold graph capturing.
        with torch.cuda.graph(graph):
            fused_optimizer.step()

        # Fill the static tensors with test data and collect results after a replay.
        for i in range(self.num_param_tensors):
            state_params_test[i].detach().copy_(self.state_params[i])
            compute_params_test[i].detach().copy_(self.state_params[i])
            compute_params_test[i].grad.detach().copy_(self.grads[i])
            swa_params_test[i].detach().copy_(self.state_params[i])
            swa_n_averaged_test[i].fill_(self.swa_n_averaged)
            moments_test[i].detach().copy_(self.moments[i])
            velocities_test[i].detach().copy_(self.velocities[i])
            steps_test[i].fill_(self.adam_step)

        graph.replay()
        # graph.debug_dump(debug_path)

        return (
            state_params_test,
            compute_params_test,
            swa_params_test,
            swa_n_averaged_test,
            moments_test,
            velocities_test,
            steps_test,
        )

    def _assert_param_and_moment_match_on_random_data(self, test_func, *args, **kwargs):
        (
            state_params_test,
            compute_params_test,
            swa_params_test,
            swa_n_averaged_test,
            moments_test,
            velocities_test,
            steps_test,
        ) = test_func(*args, **kwargs)
        (
            state_params_gt,
            compute_params_gt,
            swa_params_gt,
            swa_n_averaged_gt,
            moments_gt,
            velocities_gt,
            steps_gt,
        ) = self._get_ground_truth()

        # Ensure parameters are actually updated.
        for p_gt, p_test, p_origin in zip(state_params_gt, state_params_test, self.state_params):
            self.assertFalse(torch.allclose(p_gt, p_origin, rtol=self.rtol, atol=self.atol))
            self.assertFalse(torch.allclose(p_test, p_origin, rtol=self.rtol, atol=self.atol))
        # Ensure FusedAdamSWA correctness.
        for step, step_gt in zip(steps_test, steps_gt):
            self.assertEqual(step, step_gt)
        for swa_n_averaged in swa_n_averaged_test:
            self.assertEqual(swa_n_averaged, swa_n_averaged_gt)
        for p_test, p_gt in zip(
            chain(state_params_test, compute_params_test, swa_params_test),
            chain(state_params_gt, compute_params_gt, swa_params_gt),
        ):
            self.assertTrue(torch.allclose(p_test, p_gt, rtol=self.rtol, atol=self.atol))
        # Ensure moments are updated correctly.
        for m, m_gt in zip(chain(moments_test, velocities_test), chain(moments_gt, velocities_gt)):
            self.assertTrue(torch.allclose(m, m_gt, rtol=self.rtol, atol=self.atol))
        # Ensure gradients are set to zeros.
        for p in compute_params_test:
            self.assertTrue(p.grad.detach().eq(0).all())

    def test_eager_mode_correctness(self):
        self._assert_param_and_moment_match_on_random_data(self._get_eager_mode_results)

    def test_cuda_graph_mode_correctness(self):
        self._assert_param_and_moment_match_on_random_data(self._get_cuda_graph_mode_results)


if __name__ == "__main__":
    unittest.main()
