from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RecipeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Hardware(RecipeModel):
    nodes: int = Field(default=1, gt=0)
    gpus_per_node: int = Field(default=1, gt=0)


class Runtime(RecipeModel):
    container_image: str | None = None
    tensor_parallel_size: int | None = Field(default=None, gt=0)
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    max_model_len: int | None = Field(default=None, gt=0)
    kv_cache_dtype: str | None = None
    quantization: str | None = None
    command: str | None = None


class Expectations(RecipeModel):
    requires_docker: bool = True
    requires_connectx: bool = False
    min_mem_available_gb_before_start: float | None = Field(default=None, ge=0)


class Recipe(RecipeModel):
    name: str
    backend: str
    model: str
    is_moe: bool | None = None
    hardware: Hardware = Field(default_factory=Hardware)
    runtime: Runtime = Field(default_factory=Runtime)
    expectations: Expectations = Field(default_factory=Expectations)
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        return cls.model_validate(data)
