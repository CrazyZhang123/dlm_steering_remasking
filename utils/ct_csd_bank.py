from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class CTCSDBank:
    def __init__(self, state: dict[str, Any], device: torch.device, dtype: torch.dtype = torch.float32):
        self.format = state.get("format")
        if self.format != "ct_csd_v1":
            raise ValueError(f"Unsupported CT-CSD bank format: {self.format!r}")

        self.target_layer = int(state["target_layer"])
        self.config = dict(state.get("config", {}))
        self.cluster_sizes = state["cluster_sizes"].to(device="cpu", dtype=torch.long)
        raw_center_categories = state.get("center_categories")
        if raw_center_categories is None:
            self.center_categories = None
            self.categories = None
        else:
            self.center_categories = [str(category) for category in raw_center_categories]
            if len(self.center_categories) != self.cluster_sizes.shape[0]:
                raise ValueError("center_categories length must match number of centers")
            self.categories = [
                str(category)
                for category in state.get("categories", sorted(set(self.center_categories)))
            ]
        raw_centers = state.get("raw_centers", state["centers"])
        self.raw_centers = raw_centers.to(device=device, dtype=dtype)
        self.vectors_unit = self._unit(state["vectors"].to(device=device, dtype=dtype))

        if self.raw_centers.ndim != 2 or self.vectors_unit.ndim != 2:
            raise ValueError("centers and vectors must be rank-2 tensors")
        if self.raw_centers.shape != self.vectors_unit.shape:
            raise ValueError("raw centers and vectors must have identical shape")
        hidden_dim = int(self.vectors_unit.shape[1])
        self.preprocess = self._load_preprocess(state.get("preprocess"), device, dtype, hidden_dim)

        route_centers = state.get("route_centers")
        if route_centers is None:
            route_centers = state.get("centers_unit")
        if route_centers is None:
            route_centers = self._unit(state["centers"].to(device=device, dtype=dtype))
        self.route_centers = self._unit(route_centers.to(device=device, dtype=dtype))
        self.centers_unit = self.route_centers

        if self.route_centers.ndim != 2:
            raise ValueError("route_centers must be a rank-2 tensor")
        if self.route_centers.shape[0] != self.vectors_unit.shape[0]:
            raise ValueError("route_centers row count must match vectors")
        route_dim = self._route_feature_dim(hidden_dim)
        if self.route_centers.shape[1] != route_dim:
            raise ValueError("route_centers feature dimension does not match preprocess output")
        if self.cluster_sizes.shape[0] != self.route_centers.shape[0]:
            raise ValueError("cluster_sizes length must match number of centers")

        self.num_clusters = int(self.route_centers.shape[0])
        self.reset_diagnostics()

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "CTCSDBank":
        return cls(state, device=device, dtype=dtype)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "CTCSDBank":
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls.from_state_dict(state, device=device, dtype=dtype)

    @staticmethod
    def _unit(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=1e-8)

    def _load_preprocess(
        self,
        preprocess: dict[str, Any] | None,
        device: torch.device,
        dtype: torch.dtype,
        hidden_dim: int,
    ) -> dict[str, Any]:
        state = dict(preprocess or {"mode": "l2_only"})
        mode = str(state.get("mode", "l2_only"))
        if mode == "l2_only":
            return {"mode": "l2_only"}
        if mode not in {"center_l2", "center_pca128_l2", "center_pca256_l2"}:
            raise ValueError(f"Unsupported feature_preprocess: {mode!r}")
        if "mean" not in state or state["mean"] is None:
            raise ValueError(f"feature_preprocess={mode!r} requires a mean tensor")
        mean = state["mean"].to(device=device, dtype=dtype)
        if mean.ndim != 1 or int(mean.shape[0]) != hidden_dim:
            raise ValueError("preprocess mean dimension must match hidden dimension")
        loaded: dict[str, Any] = {"mode": mode, "mean": mean}
        if mode == "center_l2":
            return loaded
        if "pca_components" not in state or state["pca_components"] is None:
            raise ValueError(f"feature_preprocess={mode!r} requires pca_components")
        components = state["pca_components"].to(device=device, dtype=dtype)
        if components.ndim != 2 or int(components.shape[1]) != hidden_dim:
            raise ValueError("pca_components must have shape [pca_dim, hidden_dim]")
        loaded["pca_components"] = components
        loaded["pca_dim"] = int(state.get("pca_dim", components.shape[0]))
        return loaded

    def _route_feature_dim(self, hidden_dim: int) -> int:
        if self.preprocess["mode"] in {"l2_only", "center_l2"}:
            return hidden_dim
        return int(self.preprocess["pca_components"].shape[0])

    def _transform_route_features(self, hidden: torch.Tensor) -> torch.Tensor:
        mode = self.preprocess["mode"]
        if mode == "l2_only":
            return self._unit(hidden)
        centered = hidden - self.preprocess["mean"]
        if mode == "center_l2":
            return self._unit(centered)
        projected = centered @ self.preprocess["pca_components"].T
        return self._unit(projected)

    def reset_diagnostics(self) -> None:
        self.route_count = torch.zeros(self.num_clusters, dtype=torch.long)
        self.active_count = torch.zeros(self.num_clusters, dtype=torch.long)
        self.route_time_sec = 0.0

    def route(self, hidden: torch.Tensor) -> torch.Tensor:
        original_shape = hidden.shape[:-1]
        flat = hidden.reshape(-1, hidden.shape[-1])
        route_features = self._transform_route_features(
            flat.to(device=self.route_centers.device, dtype=self.route_centers.dtype)
        )
        sim = route_features @ self.route_centers.T
        return sim.argmax(dim=-1).reshape(original_shape)

    def alignment(
        self,
        hidden: torch.Tensor,
        theta: float | None = None,
        record: bool = False,
    ) -> torch.Tensor:
        start = time.perf_counter()
        route = self.route(hidden)
        flat_hidden = hidden.reshape(-1, hidden.shape[-1]).to(
            device=self.vectors_unit.device,
            dtype=self.vectors_unit.dtype,
        )
        flat_route = route.reshape(-1)
        local_vectors = self.vectors_unit[flat_route]
        score = (flat_hidden * local_vectors).sum(dim=-1).reshape(hidden.shape[:-1])
        elapsed = time.perf_counter() - start

        if record:
            self._record(route.reshape(-1), score.reshape(-1), theta)
            self.route_time_sec += elapsed

        return score.to(device=hidden.device, dtype=hidden.dtype)

    def steer(self, hidden: torch.Tensor, beta: float, theta: float, record: bool = False) -> torch.Tensor:
        start = time.perf_counter()
        route = self.route(hidden)
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_route = route.reshape(-1)
        local_vectors = self.vectors_unit[flat_route].to(device=hidden.device, dtype=hidden.dtype)
        score = (flat_hidden * local_vectors).sum(dim=-1)
        alpha = beta * (score - theta).clamp(min=0.0)
        steered = flat_hidden - alpha.unsqueeze(-1) * local_vectors
        if record:
            self._record(flat_route, score, theta)
            self.route_time_sec += time.perf_counter() - start
        return steered.reshape_as(hidden)

    def _record(self, route: torch.Tensor, score: torch.Tensor, theta: float | None) -> None:
        route_cpu = route.detach().to("cpu")
        self.route_count += torch.bincount(route_cpu, minlength=self.num_clusters)
        if theta is not None:
            active_route = route_cpu[score.detach().to("cpu") > theta]
            self.active_count += torch.bincount(active_route, minlength=self.num_clusters)

    def diagnostics(self) -> dict[str, Any]:
        total_routed = int(self.route_count.sum().item())
        total_active = int(self.active_count.sum().item())
        activation_rate = float(total_active / total_routed) if total_routed else 0.0
        result = {
            "format": self.format,
            "target_layer": self.target_layer,
            "num_clusters": self.num_clusters,
            "feature_preprocess": self.preprocess["mode"],
            "cluster_sizes": [int(x) for x in self.cluster_sizes.tolist()],
            "route_count": [int(x) for x in self.route_count.tolist()],
            "active_count": [int(x) for x in self.active_count.tolist()],
            "total_routed": total_routed,
            "total_active": total_active,
            "activation_rate": activation_rate,
            "route_time_sec": float(self.route_time_sec),
        }
        if self.center_categories is not None:
            route_by_category = {category: 0 for category in self.categories}
            active_by_category = {category: 0 for category in self.categories}
            for idx, category in enumerate(self.center_categories):
                route_by_category[category] = route_by_category.get(category, 0) + int(self.route_count[idx].item())
                active_by_category[category] = active_by_category.get(category, 0) + int(self.active_count[idx].item())
            result["center_categories"] = list(self.center_categories)
            result["category_route_count"] = route_by_category
            result["category_active_count"] = active_by_category
        return result
