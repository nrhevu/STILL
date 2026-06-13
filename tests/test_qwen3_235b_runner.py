import importlib.util
from pathlib import Path
from types import SimpleNamespace


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen3_235b_256k_benchmark.py"
    spec = importlib.util.spec_from_file_location("qwen3_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_config(tmp_path: Path) -> dict:
    return {
        "model": "Qwen/Qwen3-235B-A22B",
        "context_length": 262144,
        "rope_scaling_json": '{"rope_type":"yarn","factor":8.0}',
        "max_position_embeddings": 262144,
        "attn_implementation": "sdpa",
        "prefill_chunk_size": 2048,
        "device_map": "auto",
        "max_memory": "0=280GiB,cpu=512GiB",
        "dataset": {
            "task": "mixed",
            "depths": "0,50,100",
            "trials": 2,
            "case_context_token_margin": 512,
        },
        "compact_eval": {
            "compare_full_cache": True,
            "untrained_compactor": True,
            "num_latents": 1,
            "exact_tokens": 512,
            "exact_strategy": "lexical_linked",
            "beta_base": "zero",
        },
        "training_gate": {"min_relative_accuracy_to_full": 0.95},
        "outputs": {
            "records": str(tmp_path / "records.jsonl"),
            "summary": str(tmp_path / "summary.json"),
            "report": str(tmp_path / "report.md"),
        },
    }


def test_runner_builds_mixed_lexical_linked_command(tmp_path: Path) -> None:
    runner = load_runner()
    args = SimpleNamespace(max_new_tokens=24, max_storage="10TB")
    command = runner.benchmark_command(sample_config(tmp_path), args)
    text = runner.command_text(command)

    assert "--task mixed" in text
    assert "--exact-strategy lexical_linked" in text
    assert "--context-lengths 262144" in text
    assert "--prefill-chunk-size 2048" in text
    assert "--compare-full-cache" in command
    assert "--untrained-compactor" in command
    assert "'\"'" not in text


def test_runner_writes_blocked_report(tmp_path: Path) -> None:
    runner = load_runner()
    config = sample_config(tmp_path)
    args = SimpleNamespace(max_new_tokens=24, max_storage="10TB")
    benchmark = runner.benchmark_command(config, args)
    report = runner.report_command(config)
    path = runner.write_blocked_report(
        config=config,
        status={"torch_cuda_is_available": False, "dev_kfd_read_write": False},
        reason="preflight failed",
        benchmark=benchmark,
        report=report,
    )

    content = path.read_text(encoding="utf-8")
    assert "Blocked" in content
    assert "preflight failed" in content
    assert "overall_relative_success_to_full >= 0.95" in content


def test_runner_wraps_benchmark_for_rocm_docker(tmp_path: Path) -> None:
    runner = load_runner()
    args = SimpleNamespace(max_new_tokens=24, max_storage="10TB")
    command = runner.benchmark_command(sample_config(tmp_path), args)
    wrapped = runner.runtime_command(command, runtime="rocm-docker")

    assert wrapped[:2] == ["scripts/rocm_docker_run.sh", "python"]
    assert "scripts/evaluate_niah.py" in wrapped
    assert command[0] not in wrapped[:2]
