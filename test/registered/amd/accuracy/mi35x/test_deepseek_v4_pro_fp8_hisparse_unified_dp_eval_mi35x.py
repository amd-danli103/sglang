"""MI35x DeepSeek-V4-Pro FP8 unified-KV + HiSparse + DP-attention GSM8K test (8-GPU).

Same unified_kv + HiSparse path as the base test, but with TP=8 + DP=8
data-parallel attention (``--dp 8 --enable-dp-attention``). Verifies HiSparse
hot/cold pool + swap behaves correctly under DP attention and that GSM8K stays
aligned with the dense baseline.

Registry: nightly-amd-8-gpu-mi35x-deepseek-v4-pro-hisparse-unified-dp suite
"""

import os
import resource
import unittest
from types import SimpleNamespace

import requests

import sglang.test as sglang_test_pkg
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=7200,
    suite="nightly-amd-8-gpu-mi35x-deepseek-v4-pro-hisparse-unified-dp",
    nightly=True,
)

DEEPSEEK_V4_PRO_FP8_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V4_PRO_MODEL_PATH_FP8", "sgl-project/DeepSeek-V4-Pro-FP8"
)
SERVER_LAUNCH_TIMEOUT = 5400

# DeepSeek-V4 ROCm env with the unified_kv attention backend.
COMMON_ENV_VARS = {
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "false",
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_ROCM700A": "1",
    "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
    "SGLANG_OPT_USE_FUSED_COMPRESS_TRITON": "true",
    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
    "SGLANG_OPT_FP8_WO_A_GEMM": "false",
    "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "false",
    "SGLANG_OPT_USE_TOPK_V2": "false",
    "SGLANG_OPT_USE_AITER_INDEXER": "true",
    "SGLANG_OPT_USE_TILELANG_INDEXER": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "false",
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
    "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "false",
    "SGLANG_ROCM_USE_MULTI_STREAM": "false",
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "SGLANG_EAGER_INPUT_NO_COPY": "true",
}

# ratio=4 exercises the C4 offload/shrink path under DP (ratio=1 leaves it uncovered).
HISPARSE_CONFIG = (
    '{"top_k": 1024, "device_buffer_size": 2048, "host_to_device_ratio": 4}'
)


class TestDeepseekV4ProFp8HiSparseUnifiedDPMI35x(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(
            resource.RLIMIT_NOFILE, (max(_soft, min(_hard, 65536)), _hard)
        )

        cls.model = DEEPSEEK_V4_PRO_FP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        env = os.environ.copy()
        env.update(COMMON_ENV_VARS)

        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--dp",
            "8",
            "--enable-dp-attention",
            "--disable-radix-cache",
            "--attention-backend",
            "dsv4",
            "--max-running-requests",
            "256",
            "--page-size",
            "256",
            "--mem-fraction-static",
            "0.90",
            "--swa-full-tokens-ratio",
            "0.1",
            "--chunked-prefill-size",
            "8192",
            "--disable-shared-experts-fusion",
            "--tool-call-parser",
            "deepseekv4",
            "--reasoning-parser",
            "deepseek-v4",
            "--enable-hisparse",
            "--hisparse-config",
            HISPARSE_CONFIG,
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=1319,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v4-pro-fp8 unified_kv hisparse DP-attn)\n"
                f'{metrics["accuracy"]=:.3f}\n'
            )
        self.assertGreater(metrics["accuracy"], 0.90)

    def test_b_long_context_swap(self):
        """Exercise the host<->device swap path under DP attention.

        GSM8K prompts (~1k tokens) stay device-resident and never swap. A
        ~19k-token prompt overflows the shrunk on-device C4 hot region, forcing
        cold C4 tokens to stream in from the host pool; a passcode buried at the
        top must be retrieved, so the swap kernel is validated end-to-end.
        """
        dirpath = os.path.dirname(sglang_test_pkg.__file__)
        with open(os.path.join(dirpath, "long_prompt.txt")) as f:
            filler = f.read()
        filler = filler * 3

        passcode = "7G-ZULU-4419"
        prompt = (
            f"Remember this passcode exactly: {passcode}. "
            "You will be asked to recall it at the end.\n\n"
            f"{filler}\n\n"
            "Question: What is the exact passcode given at the very beginning of "
            "this message? Reply with only the passcode, nothing else."
        )

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 1024,
            },
            timeout=900,
        )
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        text = (
            (message.get("reasoning_content") or "")
            + "\n"
            + (message.get("content") or "")
        )
        print(f"long_context_swap completion={text!r}")

        retrieved = passcode in text
        if is_in_ci():
            write_github_step_summary(
                f"### test_long_context_swap (deepseek-v4-pro-fp8 unified_kv hisparse DP-attn)\n"
                f"{retrieved=}\n"
            )
        self.assertTrue(
            retrieved,
            f"passcode {passcode!r} not found in completion: {text!r}",
        )


if __name__ == "__main__":
    import sys

    sys.argv = [a for a in sys.argv if a not in ("-f", "--failfast")]
    unittest.main()
