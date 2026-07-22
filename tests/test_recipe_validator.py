from pathlib import Path

from spark_doctor.recipes.validator import _looks_moe, load_recipe, validate_recipe

FIXTURES = Path(__file__).parent / "fixtures"


def test_tp_exceeds_gpu_count_fails():
    recipe = load_recipe(FIXTURES / "recipe_tp_too_high.yaml")
    result = validate_recipe(recipe, detected_gpu_count=1)
    assert result.status == "fail"
    ids = [i.id for i in result.issues]
    assert "recipe.tensor_parallel_exceeds_gpu_count" in ids


def test_ok_recipe_passes_or_warns():
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    result = validate_recipe(recipe, detected_gpu_count=1, detected_arch="aarch64")
    assert result.status in ("pass", "warn")
    assert not any(i.id == "recipe.tensor_parallel_exceeds_gpu_count" for i in result.issues)


def test_arm_arch_incompatibility():
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    # Override image with amd64 hint
    recipe.runtime.container_image = "docker.io/example/vllm-amd64:latest"
    result = validate_recipe(recipe, detected_gpu_count=1, detected_arch="aarch64")
    assert result.status == "fail"
    assert any(i.id == "recipe.arch_incompatible" for i in result.issues)


def test_insufficient_memory():
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    recipe.expectations.min_mem_available_gb_before_start = 64
    result = validate_recipe(recipe, detected_gpu_count=1, mem_available_gb=10)
    assert result.status == "fail"


def _ids(result) -> list:
    return [i.id for i in result.issues]


def test_enforce_eager_flagged():
    recipe = load_recipe(FIXTURES / "recipe_enforce_eager.yaml")
    result = validate_recipe(recipe, detected_gpu_count=1)
    assert "recipe.vllm_enforce_eager" in _ids(result)


def test_enforce_eager_absent_no_finding():
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    result = validate_recipe(recipe, detected_gpu_count=1)
    assert "recipe.vllm_enforce_eager" not in _ids(result)


def test_mxfp4_moe_explicit_is_critical():
    recipe = load_recipe(FIXTURES / "recipe_mxfp4_moe.yaml")
    result = validate_recipe(recipe, detected_gpu_count=1)
    ids = _ids(result)
    assert "recipe.mxfp4_moe_on_blackwell" in ids
    assert result.status == "fail"
    issue = next(i for i in result.issues if i.id == "recipe.mxfp4_moe_on_blackwell")
    assert issue.severity == "critical"


def test_mxfp4_moe_inferred_from_model_name():
    # No explicit is_moe; the a3b active-param name should trip _looks_moe.
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    recipe.is_moe = None
    recipe.runtime.quantization = "mxfp4"
    assert "a3b" in recipe.model.lower()
    result = validate_recipe(recipe, detected_gpu_count=1)
    assert "recipe.mxfp4_moe_on_blackwell" in _ids(result)


def test_mxfp4_dense_model_no_finding():
    # MXFP4 on a dense (non-MoE) model must NOT flag.
    recipe = load_recipe(FIXTURES / "recipe_ok.yaml")
    recipe.model = "meta-llama/Llama-3.1-8B"
    recipe.is_moe = None
    recipe.runtime.quantization = "mxfp4"
    result = validate_recipe(recipe, detected_gpu_count=1)
    assert "recipe.mxfp4_moe_on_blackwell" not in _ids(result)


def test_looks_moe_detection():
    assert _looks_moe("Qwen/Qwen3.6-35B-A3B")
    assert _looks_moe("mistralai/Mixtral-8x7B")
    assert _looks_moe("some-moe-model")
    assert _looks_moe("foo-8x22b")
    assert not _looks_moe("meta-llama/Llama-3.1-8B")
    assert not _looks_moe("google/gemma-2-27b")
