import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import torch

from utils import make_ct_csd_llada as builder


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    mask_token_id = 126336
    all_special_ids = [0, 1, 2, 126336]

    def decode(self, token_ids, skip_special_tokens=False):
        value = token_ids[0]
        return {
            3: "word",
            4: " ",
            5: "\n",
            6: ".",
        }.get(value, "x")


class _SequenceTokenizer(_Tokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"):
        return torch.tensor([[10, 11]], dtype=torch.long)

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids_by_text = {
            "harmful1": [3, 4, 6, 2],
            "harmful2": [3, 4, 6, 2],
            "refusal1": [3, 4, 6, 2],
            "refusal2": [3, 4, 2],
        }
        return {"input_ids": torch.tensor([ids_by_text[text]], dtype=torch.long)}


class _Model:
    config = type("Config", (), {"hidden_size": 2})()


class _HookHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _HookBlock:
    def __init__(self):
        self.hook = None
        self.handle = _HookHandle()

    def register_forward_hook(self, hook):
        self.hook = hook
        return self.handle


class _HookModel:
    def __init__(self, hidden, fail=False):
        self.hidden = hidden
        self.fail = fail
        self.block = _HookBlock()
        self.model = type("Inner", (), {})()
        self.model.transformer = type("Transformer", (), {"blocks": [self.block]})()

    def __call__(self, input_ids):
        if self.fail:
            raise RuntimeError("forward failed")
        self.block.hook(None, None, self.hidden)
        return object()


class _FakeMiniBatchKMeans:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.partial_fit_batches = []
        _FakeMiniBatchKMeans.instances.append(self)

    def partial_fit(self, features):
        self.partial_fit_batches.append(features.copy())
        return self


class _NearestAxisKMeans:
    def predict(self, features):
        return [0 if row[0] > row[1] else 1 for row in features]


class _RecordingKMeans:
    def __init__(self):
        self.predicted_features = None

    def predict(self, features):
        self.predicted_features = features.copy()
        return [0 if row[0] >= row[1] else 1 for row in features]


class _RecordingCategoryKMeans:
    def __init__(self, category):
        self.category = category
        self.partial_fit_batches = []
        self.predicted_features = None

    def partial_fit(self, features):
        self.partial_fit_batches.append(features.copy())
        return self

    def predict(self, features):
        self.predicted_features = features.copy()
        if self.category == "fraud":
            return [0 for _row in features]
        return [0 if row[0] >= row[1] else 1 for row in features]


class BuilderHelpersTest(unittest.TestCase):
    def test_keep_response_token_filters_special_and_blank_tokens(self):
        tokenizer = _Tokenizer()

        self.assertFalse(builder.keep_response_token(tokenizer, 0))
        self.assertFalse(builder.keep_response_token(tokenizer, 2))
        self.assertFalse(builder.keep_response_token(tokenizer, 126336))
        self.assertFalse(builder.keep_response_token(tokenizer, 4))
        self.assertFalse(builder.keep_response_token(tokenizer, 5))
        self.assertTrue(builder.keep_response_token(tokenizer, 3))
        self.assertTrue(builder.keep_response_token(tokenizer, 6))

    def test_filter_response_hidden_states_reuses_token_filter(self):
        tokenizer = _Tokenizer()
        response_ids = torch.tensor([3, 4, 6, 2])
        hidden = torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ]
        )

        filtered = builder.filter_response_hidden_states(tokenizer, response_ids, hidden)

        self.assertTrue(torch.equal(filtered, torch.tensor([[1.0, 0.0], [3.0, 0.0]])))

    def test_filter_response_tokens_returns_kept_hidden_states_and_token_ids(self):
        tokenizer = _Tokenizer()
        response_ids = torch.tensor([3, 4, 6, 2])
        hidden = torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ]
        )

        filtered_hidden, filtered_ids = builder.filter_response_tokens(tokenizer, response_ids, hidden)

        self.assertTrue(torch.equal(filtered_hidden, torch.tensor([[1.0, 0.0], [3.0, 0.0]])))
        self.assertTrue(torch.equal(filtered_ids, torch.tensor([3, 6])))

    def test_apply_probe_threshold_keeps_only_high_scoring_tokens(self):
        hidden = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
        token_ids = torch.tensor([3, 6, 7])
        scores = torch.tensor([0.2, 0.7, 0.9])

        kept_hidden, kept_ids, kept_scores = builder.apply_probe_threshold(
            hidden,
            token_ids,
            scores,
            threshold=0.7,
        )

        self.assertTrue(torch.equal(kept_hidden, torch.tensor([[0.0, 2.0], [3.0, 0.0]])))
        self.assertTrue(torch.equal(kept_ids, torch.tensor([6, 7])))
        self.assertTrue(torch.equal(kept_scores, torch.tensor([0.7, 0.9])))

    def test_top_ratio_select_caps_ceil_ratio_and_keeps_token_alignment(self):
        hidden = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]])
        token_ids = torch.tensor([11, 12, 13, 14])
        scores = torch.tensor([0.2, 0.9, 0.8, 0.1])

        selected_hidden, selected_ids, selected_scores, selected_mask = builder.top_ratio_select(
            hidden,
            token_ids,
            scores,
            ratio=0.75,
            max_selected_tokens=2,
        )

        self.assertTrue(torch.equal(selected_hidden, torch.tensor([[0.0, 2.0], [3.0, 0.0]])))
        self.assertTrue(torch.equal(selected_ids, torch.tensor([12, 13])))
        self.assertTrue(torch.equal(selected_scores, torch.tensor([0.9, 0.8])))
        self.assertEqual(selected_mask.tolist(), [False, True, True, False])

    def test_direction_top_ratio_selects_tokens_by_category_direction_with_global_fallback(self):
        hidden = torch.tensor([[0.0, 3.0], [4.0, 0.0], [1.0, 0.0]])
        token_ids = torch.tensor([21, 22, 23])
        args = SimpleNamespace(
            token_selection="direction_top_ratio",
            selection_ratio=0.34,
            max_selected_tokens=2,
            category_key="semantic_category",
            coarse_direction_type="category",
            min_coarse_tokens=2,
            coarse_directions_by_category={"illegal": torch.tensor([1.0, 0.0])},
            coarse_direction_token_counts={"illegal": 1},
            global_coarse_direction=torch.tensor([0.0, 1.0]),
        )

        selected_hidden, selected_ids = builder.select_harmful_response_tokens(
            hidden,
            token_ids,
            {"semantic_category": "illegal"},
            args,
        )

        self.assertTrue(torch.equal(selected_hidden, torch.tensor([[0.0, 3.0], [4.0, 0.0]])))
        self.assertTrue(torch.equal(selected_ids, torch.tensor([21, 22])))

    def test_random_top_ratio_is_deterministic_per_sample_index(self):
        hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        token_ids = torch.tensor([31, 32, 33, 34, 35])
        args = SimpleNamespace(
            token_selection="random_top_ratio",
            selection_ratio=0.4,
            max_selected_tokens=4,
            seed=123,
            _sample_index=7,
        )

        first_hidden, first_ids = builder.select_harmful_response_tokens(hidden, token_ids, {}, args)
        second_hidden, second_ids = builder.select_harmful_response_tokens(hidden, token_ids, {}, args)

        self.assertEqual(first_ids.tolist(), second_ids.tolist())
        self.assertEqual(first_ids.numel(), 2)
        self.assertTrue(torch.equal(first_hidden, hidden[torch.isin(token_ids, first_ids)]))

    def test_extract_target_layer_tokens_uses_hook_and_slices_response(self):
        hidden = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        model = _HookModel(hidden)

        tokens = builder.extract_target_layer_tokens(
            model,
            torch.tensor([[10, 11, 12]]),
            response_start=1,
            target_layer=0,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(tokens, torch.tensor([[2.0, 0.0], [3.0, 0.0]])))
        self.assertTrue(model.block.handle.removed)

    def test_extract_target_layer_tokens_removes_hook_when_forward_raises(self):
        model = _HookModel(torch.zeros(1, 1, 2), fail=True)

        with self.assertRaises(RuntimeError):
            builder.extract_target_layer_tokens(
                model,
                torch.tensor([[10]]),
                response_start=0,
                target_layer=0,
                device=torch.device("cpu"),
            )

        self.assertTrue(model.block.handle.removed)

    def test_iter_valid_sample_tokens_filters_harmful_and_safe_hidden_states(self):
        args = SimpleNamespace(max_response_len=128, max_total_len=2048, target_layer=31)
        h_raw = torch.tensor([[1.0, 0.0], [99.0, 99.0], [0.0, 3.0], [88.0, 88.0]])
        s_raw = torch.tensor([[10.0, 0.0], [99.0, 99.0], [0.0, 30.0], [88.0, 88.0]])

        with patch.object(builder, "extract_target_layer_tokens", side_effect=[h_raw, s_raw]):
            h_tokens, safe_mean = builder.iter_valid_sample_tokens(
                _Model(),
                _SequenceTokenizer(),
                {"prompt": "prompt", "response": "harmful1"},
                "refusal1",
                args,
                torch.device("cpu"),
            )

        self.assertTrue(torch.equal(h_tokens, torch.tensor([[1.0, 0.0], [0.0, 3.0]])))
        self.assertTrue(torch.equal(safe_mean, torch.tensor([5.0, 15.0])))

    def test_transform_route_features_centers_projects_and_normalizes(self):
        preprocess = {
            "mode": "center_pca128_l2",
            "mean": torch.tensor([1.0, 1.0, 1.0]),
            "pca_components": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "pca_dim": 2,
        }
        hidden = torch.tensor([[2.0, 1.0, 1.0], [1.0, 3.0, 1.0]])

        features = builder.transform_route_features(hidden, preprocess)

        self.assertTrue(torch.allclose(features, torch.tensor([[1.0, 0.0], [0.0, 1.0]]), atol=1e-6))

    def test_fit_route_preprocess_uses_selected_harmful_and_safe_response_tokens(self):
        args = SimpleNamespace(
            max_response_len=128,
            max_total_len=2048,
            target_layer=31,
            feature_preprocess="center_l2",
            pca_dim=128,
            token_selection="direction_top_ratio",
            selection_ratio=0.5,
            max_selected_tokens=1,
            category_key="semantic_category",
            coarse_direction_type="global",
            min_coarse_tokens=1,
            global_coarse_direction=torch.tensor([1.0, 0.0]),
        )
        h_raw = torch.tensor([[10.0, 0.0], [0.0, 2.0], [1.0, 0.0], [88.0, 88.0]])
        s_raw = torch.tensor([[0.0, 4.0], [0.0, 6.0], [0.0, 6.0], [88.0, 88.0]])

        with patch.object(builder.random, "choice", return_value="refusal1"):
            with patch.object(builder, "extract_target_layer_tokens", side_effect=[h_raw, s_raw]):
                preprocess, skipped = builder.fit_route_preprocess(
                    _Model(),
                    _SequenceTokenizer(),
                    [{"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"}],
                    ["refusal1"],
                    args,
                    torch.device("cpu"),
                )

        self.assertEqual(skipped, 0)
        self.assertEqual(preprocess["mode"], "center_l2")
        self.assertTrue(torch.allclose(preprocess["mean"], torch.tensor([10.0 / 3.0, 10.0 / 3.0])))

    def test_fit_route_preprocess_pca_avoids_full_svd(self):
        args = SimpleNamespace(
            max_response_len=128,
            max_total_len=2048,
            target_layer=31,
            feature_preprocess="center_pca128_l2",
            pca_dim=128,
            token_selection="all",
            selection_ratio=1.0,
            max_selected_tokens=32,
            category_key="semantic_category",
            seed=42,
        )
        samples = [{"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"}]

        with patch.object(
            builder,
            "extract_sample_response_tokens",
            return_value=(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([3, 6]),
                torch.tensor([[1.0, 1.0]]),
            ),
        ):
            with patch.object(builder.torch.linalg, "svd", side_effect=AssertionError("full SVD should not be used")):
                preprocess, skipped = builder.fit_route_preprocess(
                    _Model(),
                    _SequenceTokenizer(),
                    samples,
                    ["refusal1"],
                    args,
                    torch.device("cpu"),
                )

        self.assertEqual(skipped, 0)
        self.assertEqual(preprocess["mode"], "center_pca128_l2")
        self.assertEqual(preprocess["pca_dim"], 2)
        self.assertEqual(preprocess["requested_pca_dim"], 128)
        self.assertEqual(preprocess["pca_components"].shape, (2, 2))

    def _preprocess_args(self, mode: str, cache: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            max_response_len=128,
            max_total_len=2048,
            target_layer=31,
            feature_preprocess=mode,
            pca_dim=128,
            token_selection="all",
            selection_ratio=1.0,
            max_selected_tokens=32,
            category_key="semantic_category",
            seed=42,
            preprocess_stats_cache=cache,
        )

    def test_fit_route_preprocess_saves_stats_cache_and_derives_other_dims_without_refit(self):
        samples = [{"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"}]
        tokens = (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([3, 6]),
            torch.tensor([[1.0, 1.0]]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache = str(Path(tmp) / "stats.pt")
            with patch.object(builder, "extract_sample_response_tokens", return_value=tokens):
                fresh, skipped = builder.fit_route_preprocess(
                    _Model(),
                    _SequenceTokenizer(),
                    samples,
                    ["refusal1"],
                    self._preprocess_args("center_l2", cache),
                    torch.device("cpu"),
                )
            self.assertEqual(skipped, 0)
            self.assertTrue(Path(cache).exists())
            stats = torch.load(cache, weights_only=True)
            # center_l2 自身不需要 gram，但指定缓存路径时也应累计，供后续任意 PCA 维度派生
            self.assertIsNotNone(stats["gram"])

            # 命中缓存后不允许再走前向拟合遍：extract 一旦被调用即失败
            with patch.object(
                builder,
                "extract_sample_response_tokens",
                side_effect=AssertionError("fit pass should be skipped when cache hits"),
            ):
                cached_pca, skipped_pca = builder.fit_route_preprocess(
                    _Model(),
                    _SequenceTokenizer(),
                    samples,
                    ["refusal1"],
                    self._preprocess_args("center_pca256_l2", cache),
                    torch.device("cpu"),
                )
            self.assertEqual(skipped_pca, 0)
            self.assertEqual(cached_pca["mode"], "center_pca256_l2")
            self.assertEqual(cached_pca["requested_pca_dim"], 256)
            self.assertEqual(cached_pca["pca_dim"], 2)
            self.assertTrue(torch.allclose(cached_pca["mean"], fresh["mean"]))

            # 缓存派生结果须与直接拟合逐项一致
            with patch.object(builder, "extract_sample_response_tokens", return_value=tokens):
                direct_pca, _ = builder.fit_route_preprocess(
                    _Model(),
                    _SequenceTokenizer(),
                    samples,
                    ["refusal1"],
                    self._preprocess_args("center_pca256_l2", None),
                    torch.device("cpu"),
                )
            self.assertTrue(torch.allclose(cached_pca["pca_components"], direct_pca["pca_components"], atol=1e-6))

    def test_fit_route_preprocess_rejects_stats_cache_with_mismatched_meta(self):
        samples = [{"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"}]
        tokens = (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([3, 6]),
            torch.tensor([[1.0, 1.0]]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache = str(Path(tmp) / "stats.pt")
            with patch.object(builder, "extract_sample_response_tokens", return_value=tokens):
                builder.fit_route_preprocess(
                    _Model(),
                    _SequenceTokenizer(),
                    samples,
                    ["refusal1"],
                    self._preprocess_args("center_l2", cache),
                    torch.device("cpu"),
                )
            mismatched = self._preprocess_args("center_l2", cache)
            mismatched.target_layer = 30
            with self.assertRaises(ValueError):
                builder.fit_route_preprocess(
                    _Model(),
                    _SequenceTokenizer(),
                    samples,
                    ["refusal1"],
                    mismatched,
                    torch.device("cpu"),
                )

    def test_extra_direction_and_preprocess_passes_do_not_consume_global_refusal_rng(self):
        args = SimpleNamespace(
            max_response_len=128,
            max_total_len=2048,
            target_layer=31,
            feature_preprocess="center_l2",
            pca_dim=128,
            token_selection="all",
            selection_ratio=1.0,
            max_selected_tokens=32,
            category_key="semantic_category",
            seed=42,
        )
        samples = [{"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"}]
        with patch.object(
            builder,
            "extract_sample_response_tokens",
            return_value=(
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([3]),
                torch.tensor([[0.0, 1.0]]),
            ),
        ):
            with patch.object(builder.random, "choice", side_effect=AssertionError("global RNG consumed")):
                _directions, _counts, _global_direction, skipped_direction = builder.fit_coarse_directions(
                    _Model(),
                    _SequenceTokenizer(),
                    samples,
                    ["refusal1"],
                    args,
                    torch.device("cpu"),
                )
                _preprocess, skipped_preprocess = builder.fit_route_preprocess(
                    _Model(),
                    _SequenceTokenizer(),
                    samples,
                    ["refusal1"],
                    args,
                    torch.device("cpu"),
                )

        self.assertEqual(skipped_direction, 0)
        self.assertEqual(skipped_preprocess, 0)

    def test_fit_coarse_directions_uses_sample_balanced_safe_mean(self):
        args = SimpleNamespace(
            max_response_len=128,
            max_total_len=2048,
            target_layer=31,
            category_key="semantic_category",
            seed=42,
        )
        samples = [
            {"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"},
            {"prompt": "prompt", "response": "harmful2", "semantic_category": "illegal"},
        ]
        with patch.object(
            builder,
            "extract_sample_response_tokens",
            side_effect=[
                (
                    torch.tensor([[10.0, 0.0], [10.0, 0.0]]),
                    torch.tensor([3, 6]),
                    torch.tensor([[0.0, 2.0]]),
                ),
                (
                    torch.tensor([[20.0, 0.0], [20.0, 0.0]]),
                    torch.tensor([3, 6]),
                    torch.tensor([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]),
                ),
            ],
        ):
            directions, counts, global_direction, skipped = builder.fit_coarse_directions(
                _Model(),
                _SequenceTokenizer(),
                samples,
                ["refusal1"],
                args,
                torch.device("cpu"),
            )

        self.assertEqual(skipped, 0)
        self.assertEqual(counts, {"illegal": 4})
        self.assertTrue(torch.allclose(directions["illegal"], torch.tensor([15.0, -6.0])))
        self.assertTrue(torch.allclose(global_direction, torch.tensor([15.0, -6.0])))

    def test_fit_minibatch_kmeans_uses_filtered_sample_balanced_safe_mean(self):
        args = SimpleNamespace(
            max_response_len=128,
            max_total_len=2048,
            target_layer=31,
            num_total_clusters=2,
            kmeans_batch_size=16,
            seed=42,
        )
        h1 = torch.tensor([[1.0, 0.0], [99.0, 99.0], [0.0, 3.0], [88.0, 88.0]])
        s1 = torch.tensor([[10.0, 0.0], [99.0, 99.0], [0.0, 30.0], [88.0, 88.0]])
        h2 = torch.tensor([[2.0, 0.0], [99.0, 99.0], [0.0, 4.0], [88.0, 88.0]])
        s2 = torch.tensor([[100.0, 0.0], [99.0, 99.0], [88.0, 88.0]])
        _FakeMiniBatchKMeans.instances = []

        with patch.object(builder, "MiniBatchKMeans", _FakeMiniBatchKMeans):
            with patch.object(builder.random, "choice", side_effect=["refusal1", "refusal2"]):
                with patch.object(builder, "extract_target_layer_tokens", side_effect=[h1, s1, h2, s2]):
                    _kmeans, safe_mean, skipped = builder.fit_minibatch_kmeans(
                        _Model(),
                        _SequenceTokenizer(),
                        [
                            {"prompt": "prompt", "response": "harmful1"},
                            {"prompt": "prompt", "response": "harmful2"},
                        ],
                        ["refusal1", "refusal2"],
                        args,
                        torch.device("cpu"),
                    )

        self.assertEqual(skipped, 0)
        self.assertTrue(torch.allclose(safe_mean, torch.tensor([52.5, 7.5])))
        self.assertEqual([batch.shape[0] for batch in _FakeMiniBatchKMeans.instances[0].partial_fit_batches], [2, 2])

    def test_accumulate_cluster_sums_uses_filtered_harmful_tokens(self):
        args = SimpleNamespace(max_response_len=128, max_total_len=2048, target_layer=31, num_total_clusters=2)
        h1 = torch.tensor([[1.0, 0.0], [99.0, 99.0], [0.0, 2.0], [88.0, 88.0]])
        s1 = torch.tensor([[10.0, 0.0], [99.0, 99.0], [0.0, 30.0], [88.0, 88.0]])
        h2 = torch.tensor([[3.0, 0.0], [99.0, 99.0], [0.0, 4.0], [88.0, 88.0]])
        s2 = torch.tensor([[100.0, 0.0], [99.0, 99.0], [88.0, 88.0]])

        with patch.object(builder.random, "choice", side_effect=["refusal1", "refusal2"]):
            with patch.object(builder, "extract_target_layer_tokens", side_effect=[h1, s1, h2, s2]):
                cluster_sums, cluster_counts, skipped = builder.accumulate_cluster_sums(
                    _Model(),
                    _SequenceTokenizer(),
                    [
                        {"prompt": "prompt", "response": "harmful1"},
                        {"prompt": "prompt", "response": "harmful2"},
                    ],
                    ["refusal1", "refusal2"],
                    _NearestAxisKMeans(),
                    args,
                    torch.device("cpu"),
                )

        self.assertEqual(skipped, 0)
        self.assertTrue(torch.equal(cluster_sums, torch.tensor([[4.0, 0.0], [0.0, 6.0]])))
        self.assertEqual(cluster_counts.tolist(), [2, 2])

    def test_accumulate_cluster_sums_predicts_with_route_features_but_sums_raw_hidden(self):
        args = SimpleNamespace(
            max_response_len=128,
            max_total_len=2048,
            target_layer=31,
            num_total_clusters=2,
            route_preprocess={
                "mode": "center_l2",
                "mean": torch.tensor([1.0, 1.0]),
            },
        )
        h_tokens = torch.tensor([[3.0, 1.0], [1.0, 4.0]])
        kmeans = _RecordingKMeans()

        with patch.object(builder.random, "choice", return_value="refusal1"):
            with patch.object(builder, "iter_valid_sample_tokens", return_value=(h_tokens, torch.zeros(2))):
                cluster_sums, cluster_counts, skipped = builder.accumulate_cluster_sums(
                    _Model(),
                    _SequenceTokenizer(),
                    [{"prompt": "prompt", "response": "harmful1"}],
                    ["refusal1"],
                    kmeans,
                    args,
                    torch.device("cpu"),
                )

        self.assertEqual(skipped, 0)
        self.assertTrue(torch.allclose(torch.tensor(kmeans.predicted_features), torch.eye(2), atol=1e-6))
        self.assertTrue(torch.equal(cluster_sums, torch.tensor([[3.0, 1.0], [1.0, 4.0]])))
        self.assertEqual(cluster_counts.tolist(), [1, 1])

    def test_main_writes_ct_csd_bank_with_cli_metadata(self):
        class _LoadableModel(_Model):
            def to(self, device):
                self.device = device
                return self

            def eval(self):
                self.is_eval = True
                return self

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            harmful_path = root / "harmful.json"
            refusals_path = root / "refusals.txt"
            output_dir = root / "out"
            harmful_path.write_text(
                '[{"prompt": "prompt", "response": "harmful1"}]',
                encoding="utf-8",
            )
            refusals_path.write_text("refusal1\n", encoding="utf-8")

            with patch.object(builder.AutoTokenizer, "from_pretrained", return_value=_SequenceTokenizer()):
                with patch.object(builder.AutoModel, "from_pretrained", return_value=_LoadableModel()):
                    with patch.object(
                        builder,
                        "fit_coarse_directions",
                        return_value=({}, {}, torch.tensor([1.0, 0.0]), 0),
                    ):
                        with patch.object(
                            builder,
                            "fit_route_preprocess",
                            return_value=(
                                {
                                    "mode": "center_pca128_l2",
                                    "mean": torch.zeros(2),
                                    "pca_components": torch.eye(2),
                                    "pca_dim": 2,
                                    "requested_pca_dim": 128,
                                },
                                0,
                            ),
                        ):
                            with patch.object(
                                builder,
                                "fit_minibatch_kmeans",
                                return_value=(object(), torch.tensor([0.5, 0.5]), 1),
                            ) as fit_mock:
                                with patch.object(
                                    builder,
                                    "accumulate_cluster_sums",
                                    return_value=(
                                        torch.tensor([[2.0, 0.0], [0.0, 6.0]]),
                                        torch.tensor([2, 3]),
                                        0,
                                    ),
                                ) as accumulate_mock:
                                    builder.main(
                                        [
                                            "--model_path",
                                            "dummy-model",
                                            "--harmful_json",
                                            str(harmful_path),
                                            "--refusals_txt",
                                            str(refusals_path),
                                            "--output_dir",
                                            str(output_dir),
                                            "--target_layer",
                                            "31",
                                            "--max_response_len",
                                            "128",
                                            "--max_total_len",
                                            "2048",
                                            "--method",
                                            "ct_csd",
                                            "--num_total_clusters",
                                            "2",
                                            "--kmeans_batch_size",
                                            "16",
                                            "--device",
                                            "cpu",
                                            "--seed",
                                            "42",
                                            "--token_selection",
                                            "direction_top_ratio",
                                            "--selection_ratio",
                                            "0.3",
                                            "--max_selected_tokens",
                                            "32",
                                            "--feature_preprocess",
                                            "center_pca128_l2",
                                        ]
                                    )

            out_path = output_dir / "ct_csd_bank.pt"
            state = torch.load(out_path, map_location="cpu", weights_only=True)

        self.assertTrue(fit_mock.called)
        self.assertTrue(accumulate_mock.called)
        self.assertEqual(state["format"], "ct_csd_v1")
        self.assertEqual(state["target_layer"], 31)
        self.assertEqual(state["config"]["method"], "ct_csd")
        self.assertEqual(state["config"]["num_total_clusters"], 2)
        self.assertEqual(state["config"]["harmful_json"], str(harmful_path))
        self.assertEqual(state["config"]["refusals_txt"], str(refusals_path))
        self.assertEqual(state["config"]["skipped_pass1"], 1)
        self.assertEqual(state["config"]["skipped_pass2"], 0)
        self.assertEqual(state["config"]["seed"], 42)
        self.assertEqual(state["config"]["model_path"], "dummy-model")
        self.assertEqual(state["config"]["max_samples"], None)
        self.assertEqual(state["config"]["kmeans_batch_size"], 16)
        self.assertEqual(state["config"]["token_selection"], "direction_top_ratio")
        self.assertEqual(state["config"]["selection_ratio"], 0.3)
        self.assertEqual(state["config"]["max_selected_tokens"], 32)
        self.assertEqual(state["config"]["feature_preprocess"], "center_pca128_l2")
        self.assertEqual(state["config"]["pca_dim"], 2)
        self.assertEqual(state["config"]["requested_pca_dim"], 128)
        self.assertEqual(state["preprocess"]["mode"], "center_pca128_l2")
        self.assertIn("route_centers", state)

    def test_main_writes_probe_ct_csd_bank_with_mil_metadata(self):
        class _LoadableModel(_Model):
            def to(self, device):
                self.device = device
                return self

            def eval(self):
                self.is_eval = True
                return self

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            harmful_path = root / "harmful.json"
            refusals_path = root / "refusals.txt"
            output_dir = root / "out"
            probe_path = root / "probe.pt"
            harmful_path.write_text(
                '[{"prompt": "prompt", "response": "harmful1"}]',
                encoding="utf-8",
            )
            refusals_path.write_text("refusal1\n", encoding="utf-8")
            torch.save(
                {
                    "format": "mil_token_probe_v1",
                    "model_family": "llada",
                    "target_layer": 31,
                    "input_dim": 2,
                    "top_q_ratio": 0.1,
                    "state_dict": {
                        "linear.weight": torch.tensor([[1.0, 0.0]]),
                        "linear.bias": torch.tensor([0.0]),
                    },
                },
                probe_path,
            )

            def fake_accumulate(_model, _tokenizer, _harmful, _refusals, _kmeans, args, _device):
                args.probe_empty_samples = 1
                return (
                    torch.tensor([[2.0, 0.0], [0.0, 6.0]]),
                    torch.tensor([2, 3]),
                    1,
                )

            with patch.object(builder.AutoTokenizer, "from_pretrained", return_value=_SequenceTokenizer()):
                with patch.object(builder.AutoModel, "from_pretrained", return_value=_LoadableModel()):
                    with patch.object(
                        builder,
                        "fit_minibatch_kmeans",
                        return_value=(object(), torch.tensor([0.5, 0.5]), 0),
                    ) as fit_mock:
                        with patch.object(
                            builder,
                            "accumulate_cluster_sums",
                            side_effect=fake_accumulate,
                        ) as accumulate_mock:
                            builder.main(
                                [
                                    "--model_path",
                                    "dummy-model",
                                    "--harmful_json",
                                    str(harmful_path),
                                    "--refusals_txt",
                                    str(refusals_path),
                                    "--output_dir",
                                    str(output_dir),
                                    "--target_layer",
                                    "31",
                                    "--max_response_len",
                                    "128",
                                    "--max_total_len",
                                    "2048",
                                    "--method",
                                    "probe_ct_csd",
                                    "--mil_probe_path",
                                    str(probe_path),
                                    "--probe_threshold",
                                    "0.7",
                                    "--num_total_clusters",
                                    "2",
                                    "--kmeans_batch_size",
                                    "16",
                                    "--device",
                                    "cpu",
                                    "--seed",
                                    "42",
                                ]
                            )

            state = torch.load(output_dir / "ct_csd_bank.pt", map_location="cpu", weights_only=True)
            summary = json.loads((output_dir / "ct_csd_bank_summary.json").read_text(encoding="utf-8"))

        self.assertTrue(fit_mock.called)
        self.assertTrue(accumulate_mock.called)
        self.assertEqual(state["config"]["method"], "probe_ct_csd")
        self.assertEqual(state["config"]["token_selection"], "mil_probe_threshold")
        self.assertEqual(state["config"]["probe_empty_samples"], 1)
        self.assertTrue(state["mil"]["enabled"])
        self.assertEqual(state["mil"]["probe_path"], str(probe_path))
        self.assertEqual(state["mil"]["probe_threshold"], 0.7)
        self.assertEqual(state["mil"]["top_q_ratio"], 0.1)
        self.assertTrue(summary["mil"]["enabled"])

    def test_resolve_path_prefers_cwd_relative_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "data.json"
            target.write_text("[]", encoding="utf-8")

            resolved = builder.resolve_path("data.json", root)

        self.assertEqual(resolved, target)

    def test_build_state_from_cluster_sums_uses_global_safe_mean(self):
        safe_mean = torch.tensor([0.5, 0.5])
        cluster_sums = torch.tensor([[3.0, 0.0], [0.0, 6.0]])
        cluster_counts = torch.tensor([3, 2])

        state = builder.build_bank_state_from_cluster_sums(
            safe_mean=safe_mean,
            cluster_sums=cluster_sums,
            cluster_counts=cluster_counts,
            target_layer=31,
            max_response_len=128,
            num_total_clusters=2,
        )

        self.assertEqual(state["format"], "ct_csd_v1")
        self.assertTrue(torch.allclose(state["centers"], torch.tensor([[1.0, 0.0], [0.0, 3.0]])))
        self.assertTrue(torch.allclose(state["vectors"], torch.tensor([[0.5, -0.5], [-0.5, 2.5]])))
        self.assertEqual(state["cluster_sizes"].tolist(), [3, 2])
        self.assertEqual(state["config"]["method"], "ct_csd")
        self.assertEqual(state["config"]["num_total_clusters"], 2)

    def test_build_state_from_cluster_sums_saves_preprocess_and_route_centers(self):
        safe_mean = torch.tensor([0.5, 0.5])
        cluster_sums = torch.tensor([[3.0, 0.0], [0.0, 6.0]])
        cluster_counts = torch.tensor([3, 2])
        route_preprocess = {"mode": "center_l2", "mean": torch.tensor([1.0, 1.0])}

        state = builder.build_bank_state_from_cluster_sums(
            safe_mean=safe_mean,
            cluster_sums=cluster_sums,
            cluster_counts=cluster_counts,
            target_layer=31,
            max_response_len=128,
            num_total_clusters=2,
            route_preprocess=route_preprocess,
            route_centers=torch.tensor([[2.0, 0.0], [0.0, 4.0]]),
        )

        self.assertTrue(torch.equal(state["raw_centers"], state["centers"]))
        self.assertEqual(state["preprocess"]["mode"], "center_l2")
        self.assertTrue(torch.equal(state["preprocess"]["mean"], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(state["route_centers"], torch.tensor([[1.0, 0.0], [0.0, 1.0]])))
        self.assertEqual(state["config"]["feature_preprocess"], "center_l2")

    def test_resolve_category_prefers_requested_key_then_fallbacks(self):
        sample = {
            "semantic_category": " illegal ",
            "functional_category": "standard",
            "category": "fallback",
        }

        self.assertEqual(builder.resolve_category(sample, "semantic_category"), "illegal")
        self.assertEqual(builder.resolve_category(sample, "missing_key"), "illegal")
        self.assertEqual(
            builder.resolve_category({"functional_category": " contextual "}, "semantic_category"),
            "contextual",
        )
        self.assertEqual(builder.resolve_category({}, "semantic_category"), "unknown")

    def test_count_response_tokens_by_category_uses_filtered_response_tokens(self):
        args = SimpleNamespace(max_response_len=128, category_key="semantic_category")
        harmful = [
            {"response": "harmful1", "semantic_category": "illegal"},
            {"response": "harmful2", "semantic_category": "fraud"},
            {"response": "   ", "semantic_category": "ignored"},
        ]

        counts = builder.count_response_tokens_by_category(_SequenceTokenizer(), harmful, args)

        self.assertEqual(counts, {"illegal": 2, "fraud": 2})

    def test_make_category_cluster_plan_allocates_total_budget_and_merges_tail(self):
        plan = builder.make_category_cluster_plan(
            {
                "illegal": 100,
                "cyber": 60,
                "fraud": 30,
                "privacy": 10,
            },
            num_total_clusters=3,
        )

        self.assertEqual(plan["categories"], ["cyber", "illegal", "other"])
        self.assertEqual(plan["category_token_counts"], {"cyber": 60, "illegal": 100, "other": 40})
        self.assertEqual(sum(plan["category_cluster_counts"].values()), 3)
        self.assertEqual(plan["category_cluster_counts"], {"cyber": 1, "illegal": 1, "other": 1})
        self.assertEqual(plan["raw_to_center_category"]["fraud"], "other")
        self.assertEqual(plan["raw_to_center_category"]["privacy"], "other")

    def test_make_category_cluster_plan_distributes_remaining_budget_by_fraction(self):
        plan = builder.make_category_cluster_plan(
            {
                "a": 70,
                "b": 20,
                "c": 10,
            },
            num_total_clusters=5,
        )

        self.assertEqual(plan["categories"], ["a", "b", "c"])
        self.assertEqual(plan["category_cluster_counts"], {"a": 3, "b": 1, "c": 1})
        self.assertEqual(sum(plan["category_cluster_counts"].values()), 5)

    def test_build_state_from_cluster_sums_adds_category_metadata(self):
        safe_mean = torch.tensor([0.5, 0.5])
        cluster_sums = torch.tensor([[3.0, 0.0], [0.0, 6.0], [8.0, 0.0]])
        cluster_counts = torch.tensor([3, 2, 4])
        category_plan = {
            "categories": ["cyber", "illegal"],
            "category_token_counts": {"cyber": 2, "illegal": 7},
            "category_cluster_counts": {"cyber": 1, "illegal": 2},
            "raw_to_center_category": {"cyber": "cyber", "illegal": "illegal"},
        }

        state = builder.build_bank_state_from_cluster_sums(
            safe_mean=safe_mean,
            cluster_sums=cluster_sums,
            cluster_counts=cluster_counts,
            target_layer=31,
            max_response_len=128,
            num_total_clusters=3,
            method="category_ct_csd",
            category_key="semantic_category",
            center_categories=["cyber", "illegal", "illegal"],
            center_cluster_ids=[0, 0, 1],
            category_plan=category_plan,
        )

        self.assertEqual(state["format"], "ct_csd_v1")
        self.assertEqual(state["config"]["method"], "category_ct_csd")
        self.assertEqual(state["config"]["category_key"], "semantic_category")
        self.assertEqual(state["categories"], ["cyber", "illegal"])
        self.assertEqual(state["center_categories"], ["cyber", "illegal", "illegal"])
        self.assertEqual(state["center_category_ids"].tolist(), [0, 1, 1])
        self.assertEqual(state["cluster_ids"].tolist(), [0, 0, 1])
        self.assertEqual(state["global_cluster_ids"].tolist(), [0, 1, 2])
        self.assertEqual(state["config"]["category_token_counts"], {"cyber": 2, "illegal": 7})
        self.assertEqual(state["config"]["category_cluster_counts"], {"cyber": 1, "illegal": 2})

    def test_fit_category_minibatch_kmeans_fits_each_category_and_safe_mean(self):
        args = SimpleNamespace(
            max_response_len=128,
            max_total_len=2048,
            target_layer=31,
            num_total_clusters=3,
            kmeans_batch_size=16,
            seed=42,
            category_key="semantic_category",
        )
        plan = {
            "categories": ["fraud", "illegal"],
            "category_token_counts": {"fraud": 2, "illegal": 2},
            "category_cluster_counts": {"fraud": 1, "illegal": 2},
            "raw_to_center_category": {"fraud": "fraud", "illegal": "illegal"},
        }
        h1 = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        s1 = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        h2 = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        s2 = torch.tensor([[4.0, 0.0], [0.0, 4.0]])

        with patch.object(builder.random, "choice", side_effect=["refusal1", "refusal2"]):
            with patch.object(builder, "iter_valid_sample_tokens", side_effect=[(h1, s1.mean(dim=0)), (h2, s2.mean(dim=0))]):
                kmeans_by_category, safe_mean, skipped = builder.fit_category_minibatch_kmeans(
                    _Model(),
                    _SequenceTokenizer(),
                    [
                        {"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"},
                        {"prompt": "prompt", "response": "harmful2", "semantic_category": "fraud"},
                    ],
                    ["refusal1", "refusal2"],
                    args,
                    torch.device("cpu"),
                    plan,
                    kmeans_factory=lambda category, n_clusters: _RecordingCategoryKMeans(category),
                )

        self.assertEqual(skipped, 0)
        self.assertTrue(torch.allclose(safe_mean, torch.tensor([1.5, 1.5])))
        self.assertEqual(sorted(kmeans_by_category), ["fraud", "illegal"])
        self.assertEqual(len(kmeans_by_category["illegal"].partial_fit_batches), 1)
        self.assertEqual(len(kmeans_by_category["fraud"].partial_fit_batches), 1)

    def test_fit_category_minibatch_kmeans_uses_route_features(self):
        args = SimpleNamespace(
            max_response_len=128,
            max_total_len=2048,
            target_layer=31,
            num_total_clusters=2,
            kmeans_batch_size=16,
            seed=42,
            category_key="semantic_category",
            route_preprocess={"mode": "center_l2", "mean": torch.tensor([1.0, 1.0])},
        )
        plan = {
            "categories": ["illegal"],
            "category_token_counts": {"illegal": 2},
            "category_cluster_counts": {"illegal": 1},
            "raw_to_center_category": {"illegal": "illegal"},
        }
        h_tokens = torch.tensor([[3.0, 1.0], [1.0, 4.0]])

        with patch.object(builder.random, "choice", return_value="refusal1"):
            with patch.object(builder, "iter_valid_sample_tokens", return_value=(h_tokens, torch.zeros(2))):
                kmeans_by_category, _safe_mean, skipped = builder.fit_category_minibatch_kmeans(
                    _Model(),
                    _SequenceTokenizer(),
                    [{"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"}],
                    ["refusal1"],
                    args,
                    torch.device("cpu"),
                    plan,
                    kmeans_factory=lambda category, n_clusters: _RecordingCategoryKMeans(category),
                )

        self.assertEqual(skipped, 0)
        batch = torch.tensor(kmeans_by_category["illegal"].partial_fit_batches[0])
        self.assertTrue(torch.allclose(batch, torch.eye(2), atol=1e-6))

    def test_accumulate_category_cluster_sums_uses_global_offsets(self):
        args = SimpleNamespace(
            max_response_len=128,
            max_total_len=2048,
            target_layer=31,
            num_total_clusters=3,
            category_key="semantic_category",
        )
        plan = {
            "categories": ["fraud", "illegal"],
            "category_token_counts": {"fraud": 2, "illegal": 2},
            "category_cluster_counts": {"fraud": 1, "illegal": 2},
            "raw_to_center_category": {"fraud": "fraud", "illegal": "illegal"},
        }
        h1 = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        h2 = torch.tensor([[3.0, 0.0], [0.0, 4.0]])

        with patch.object(builder.random, "choice", side_effect=["refusal1", "refusal2"]):
            with patch.object(builder, "iter_valid_sample_tokens", side_effect=[(h1, torch.zeros(2)), (h2, torch.zeros(2))]):
                cluster_sums, cluster_counts, skipped, center_categories, center_cluster_ids, cluster_category_counts = (
                    builder.accumulate_category_cluster_sums(
                        _Model(),
                        _SequenceTokenizer(),
                        [
                            {"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"},
                            {"prompt": "prompt", "response": "harmful2", "semantic_category": "fraud"},
                        ],
                        ["refusal1", "refusal2"],
                        {
                            "fraud": _RecordingCategoryKMeans("fraud"),
                            "illegal": _NearestAxisKMeans(),
                        },
                        args,
                        torch.device("cpu"),
                        plan,
                    )
                )

        self.assertEqual(skipped, 0)
        self.assertEqual(center_categories, ["fraud", "illegal", "illegal"])
        self.assertEqual(center_cluster_ids, [0, 0, 1])
        self.assertTrue(torch.equal(cluster_sums, torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])))
        self.assertEqual(cluster_counts.tolist(), [2, 1, 1])
        self.assertEqual(cluster_category_counts, {"fraud": [2], "illegal": [1, 1]})

    def test_main_writes_category_ct_csd_bank_with_cli_metadata(self):
        class _LoadableModel(_Model):
            def to(self, device):
                self.device = device
                return self

            def eval(self):
                self.is_eval = True
                return self

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            harmful_path = root / "harmful.json"
            refusals_path = root / "refusals.txt"
            output_dir = root / "out"
            harmful_path.write_text(
                '['
                '{"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"},'
                '{"prompt": "prompt", "response": "harmful2", "semantic_category": "fraud"}'
                ']',
                encoding="utf-8",
            )
            refusals_path.write_text("refusal1\n", encoding="utf-8")
            category_plan = {
                "categories": ["fraud", "illegal"],
                "category_token_counts": {"fraud": 2, "illegal": 2},
                "category_cluster_counts": {"fraud": 1, "illegal": 1},
                "raw_to_center_category": {"fraud": "fraud", "illegal": "illegal"},
            }

            with patch.object(builder.AutoTokenizer, "from_pretrained", return_value=_SequenceTokenizer()):
                with patch.object(builder.AutoModel, "from_pretrained", return_value=_LoadableModel()):
                    with patch.object(builder, "count_response_tokens_by_category", return_value={"illegal": 2, "fraud": 2}) as count_mock:
                        with patch.object(builder, "make_category_cluster_plan", return_value=category_plan) as plan_mock:
                            with patch.object(
                                builder,
                                "fit_category_minibatch_kmeans",
                                return_value=({"fraud": object(), "illegal": object()}, torch.tensor([0.5, 0.5]), 1),
                            ) as fit_mock:
                                with patch.object(
                                    builder,
                                    "accumulate_category_cluster_sums",
                                    return_value=(
                                        torch.tensor([[2.0, 0.0], [0.0, 6.0]]),
                                        torch.tensor([2, 3]),
                                        0,
                                        ["fraud", "illegal"],
                                        [0, 0],
                                        {"fraud": [2], "illegal": [3]},
                                    ),
                                ) as accumulate_mock:
                                    builder.main(
                                        [
                                            "--model_path",
                                            "dummy-model",
                                            "--harmful_json",
                                            str(harmful_path),
                                            "--refusals_txt",
                                            str(refusals_path),
                                            "--output_dir",
                                            str(output_dir),
                                            "--target_layer",
                                            "31",
                                            "--max_response_len",
                                            "128",
                                            "--max_total_len",
                                            "2048",
                                            "--method",
                                            "category_ct_csd",
                                            "--num_total_clusters",
                                            "2",
                                            "--kmeans_batch_size",
                                            "16",
                                            "--category_key",
                                            "semantic_category",
                                            "--device",
                                            "cpu",
                                            "--seed",
                                            "42",
                                        ]
                                    )

            state = torch.load(output_dir / "ct_csd_bank.pt", map_location="cpu", weights_only=True)

        self.assertTrue(count_mock.called)
        self.assertTrue(plan_mock.called)
        self.assertTrue(fit_mock.called)
        self.assertTrue(accumulate_mock.called)
        self.assertEqual(state["format"], "ct_csd_v1")
        self.assertEqual(state["config"]["method"], "category_ct_csd")
        self.assertEqual(state["config"]["category_key"], "semantic_category")
        self.assertEqual(state["center_categories"], ["fraud", "illegal"])
        self.assertEqual(state["cluster_ids"].tolist(), [0, 0])
        self.assertEqual(state["config"]["skipped_pass1"], 1)
        self.assertEqual(state["config"]["skipped_pass2"], 0)

    def test_main_writes_probe_category_ct_csd_bank_with_mil_metadata(self):
        class _LoadableModel(_Model):
            def to(self, device):
                self.device = device
                return self

            def eval(self):
                self.is_eval = True
                return self

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            harmful_path = root / "harmful.json"
            refusals_path = root / "refusals.txt"
            output_dir = root / "out"
            probe_path = root / "probe.pt"
            harmful_path.write_text(
                '['
                '{"prompt": "prompt", "response": "harmful1", "semantic_category": "illegal"},'
                '{"prompt": "prompt", "response": "harmful2", "semantic_category": "fraud"}'
                ']',
                encoding="utf-8",
            )
            refusals_path.write_text("refusal1\n", encoding="utf-8")
            torch.save(
                {
                    "format": "mil_token_probe_v1",
                    "model_family": "llada",
                    "target_layer": 31,
                    "input_dim": 2,
                    "top_q_ratio": 0.1,
                    "state_dict": {
                        "linear.weight": torch.tensor([[1.0, 0.0]]),
                        "linear.bias": torch.tensor([0.0]),
                    },
                },
                probe_path,
            )
            category_plan = {
                "categories": ["fraud", "illegal"],
                "category_token_counts": {"fraud": 2, "illegal": 2},
                "category_cluster_counts": {"fraud": 1, "illegal": 1},
                "raw_to_center_category": {"fraud": "fraud", "illegal": "illegal"},
            }

            with patch.object(builder.AutoTokenizer, "from_pretrained", return_value=_SequenceTokenizer()):
                with patch.object(builder.AutoModel, "from_pretrained", return_value=_LoadableModel()):
                    with patch.object(builder, "count_response_tokens_by_category", return_value={"illegal": 2, "fraud": 2}):
                        with patch.object(builder, "make_category_cluster_plan", return_value=category_plan):
                            with patch.object(
                                builder,
                                "fit_category_minibatch_kmeans",
                                return_value=({"fraud": object(), "illegal": object()}, torch.tensor([0.5, 0.5]), 0),
                            ) as fit_mock:
                                with patch.object(
                                    builder,
                                    "accumulate_category_cluster_sums",
                                    return_value=(
                                        torch.tensor([[2.0, 0.0], [0.0, 6.0]]),
                                        torch.tensor([2, 3]),
                                        0,
                                        ["fraud", "illegal"],
                                        [0, 0],
                                        {"fraud": [2], "illegal": [3]},
                                    ),
                                ) as accumulate_mock:
                                    builder.main(
                                        [
                                            "--model_path",
                                            "dummy-model",
                                            "--harmful_json",
                                            str(harmful_path),
                                            "--refusals_txt",
                                            str(refusals_path),
                                            "--output_dir",
                                            str(output_dir),
                                            "--target_layer",
                                            "31",
                                            "--max_response_len",
                                            "128",
                                            "--max_total_len",
                                            "2048",
                                            "--method",
                                            "probe_category_ct_csd",
                                            "--mil_probe_path",
                                            str(probe_path),
                                            "--probe_threshold",
                                            "0.7",
                                            "--num_total_clusters",
                                            "2",
                                            "--kmeans_batch_size",
                                            "16",
                                            "--category_key",
                                            "semantic_category",
                                            "--device",
                                            "cpu",
                                            "--seed",
                                            "42",
                                        ]
                                    )

            state = torch.load(output_dir / "ct_csd_bank.pt", map_location="cpu", weights_only=True)
            selection_summary_exists = (output_dir / "mil_token_selection_summary.json").exists()
            cluster_terms_exists = (output_dir / "cluster_token_top_terms.md").exists()

        self.assertTrue(fit_mock.called)
        self.assertTrue(accumulate_mock.called)
        self.assertEqual(state["config"]["method"], "probe_category_ct_csd")
        self.assertEqual(state["center_categories"], ["fraud", "illegal"])
        self.assertTrue(state["mil"]["enabled"])
        self.assertEqual(state["mil"]["probe_path"], str(probe_path))
        self.assertTrue(selection_summary_exists)
        self.assertTrue(cluster_terms_exists)

    def test_write_bank_summary_outputs_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            state = {
                "config": {
                    "method": "category_ct_csd",
                    "num_total_clusters": 2,
                    "category_cluster_counts": {"fraud": 1, "illegal": 1},
                    "category_token_counts": {"fraud": 2, "illegal": 3},
                    "cluster_category_counts": {"fraud": [2], "illegal": [3]},
                },
                "cluster_sizes": torch.tensor([2, 3]),
                "center_categories": ["fraud", "illegal"],
                "cluster_ids": torch.tensor([0, 0]),
            }

            builder.write_bank_summary(output_dir, state)

            summary = json.loads((output_dir / "ct_csd_bank_summary.json").read_text(encoding="utf-8"))
            markdown = (output_dir / "cluster_category_distribution.md").read_text(encoding="utf-8")

        self.assertEqual(summary["method"], "category_ct_csd")
        self.assertEqual(summary["num_total_clusters"], 2)
        self.assertEqual(summary["cluster_sizes"], [2, 3])
        self.assertEqual(summary["token_selection"], None)
        self.assertEqual(summary["feature_preprocess"], None)
        self.assertEqual(summary["requested_pca_dim"], None)
        self.assertIn("| fraud | 0 | 2 |", markdown)
        self.assertIn("| illegal | 0 | 3 |", markdown)

    def test_write_bank_summary_outputs_stage6_selection_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            state = {
                "config": {
                    "method": "ct_csd",
                    "num_total_clusters": 2,
                    "token_selection": "direction_top_ratio",
                    "selection_ratio": 0.3,
                    "max_selected_tokens": 32,
                    "coarse_direction_type": "category",
                    "min_coarse_tokens": 1024,
                    "feature_preprocess": "center_pca128_l2",
                    "pca_dim": 2,
                    "requested_pca_dim": 128,
                    "harmful_json": "harmful.json",
                    "refusals_txt": "refusals.txt",
                    "model_path": "dummy-model",
                    "max_samples": 64,
                    "kmeans_batch_size": 16,
                    "skipped_pass1": 1,
                    "skipped_pass2": 2,
                    "skipped_coarse_direction": 3,
                    "skipped_preprocess": 4,
                    "seed": 42,
                },
                "cluster_sizes": torch.tensor([2, 3]),
            }

            builder.write_bank_summary(output_dir, state)

            summary = json.loads((output_dir / "ct_csd_bank_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["token_selection"], "direction_top_ratio")
        self.assertEqual(summary["selection_ratio"], 0.3)
        self.assertEqual(summary["max_selected_tokens"], 32)
        self.assertEqual(summary["coarse_direction_type"], "category")
        self.assertEqual(summary["min_coarse_tokens"], 1024)
        self.assertEqual(summary["feature_preprocess"], "center_pca128_l2")
        self.assertEqual(summary["pca_dim"], 2)
        self.assertEqual(summary["requested_pca_dim"], 128)
        self.assertEqual(summary["harmful_json"], "harmful.json")
        self.assertEqual(summary["refusals_txt"], "refusals.txt")
        self.assertEqual(summary["model_path"], "dummy-model")
        self.assertEqual(summary["max_samples"], 64)
        self.assertEqual(summary["kmeans_batch_size"], 16)
        self.assertEqual(summary["skipped_pass1"], 1)
        self.assertEqual(summary["skipped_pass2"], 2)
        self.assertEqual(summary["skipped_coarse_direction"], 3)
        self.assertEqual(summary["skipped_preprocess"], 4)
        self.assertEqual(summary["seed"], 42)

    def test_write_probe_diagnostics_outputs_summary_and_high_score_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            diagnostics = builder.new_probe_diagnostics()
            builder.record_probe_selection(
                diagnostics,
                _Tokenizer(),
                sample_index=0,
                category="illegal",
                token_ids=torch.tensor([3, 6]),
                scores=torch.tensor([0.9, 0.4]),
                selected_mask=torch.tensor([True, False]),
            )

            builder.write_probe_diagnostics(
                output_dir,
                diagnostics,
                probe_path=Path("outputs/mil_token_probe_llada.pt"),
                probe_threshold=0.7,
                top_q_ratio=0.1,
            )

            summary = json.loads((output_dir / "mil_token_selection_summary.json").read_text(encoding="utf-8"))
            markdown = (output_dir / "mil_high_score_tokens.md").read_text(encoding="utf-8")

        self.assertEqual(summary["total_harmful_tokens_before_probe"], 2)
        self.assertEqual(summary["total_harmful_tokens_after_probe"], 1)
        self.assertEqual(summary["per_category_tokens_before_probe"], {"illegal": 2})
        self.assertEqual(summary["per_category_tokens_after_probe"], {"illegal": 1})
        self.assertIn("| illegal | word | 0.900000 | yes |", markdown)


class KnnLabelCleanTest(unittest.TestCase):
    def _toy_pool(self):
        # 前 4 行为有害：3 个纯有害（x 轴正向）+ 1 个边界有害（落入安全簇）；后 3 行为安全（x 轴负向）。
        harmful = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.95, 0.05], [-0.9, 0.1]])
        safe = torch.tensor([[-1.0, 0.0], [-0.95, 0.05], [-0.92, 0.08]])
        return torch.cat([harmful, safe], dim=0), 4

    def test_knn_keep_decisions_removes_boundary_token(self):
        features, n_h = self._toy_pool()
        keep = builder.knn_keep_decisions(features, n_h, k=4, keep_ratio=0.5, backend="sklearn", metric="cosine")
        self.assertTrue(bool(keep[:3].all()))  # 纯有害 token 邻居多为有害 → 保留
        self.assertFalse(bool(keep[3]))  # 边界有害 token 邻居多为安全 → 剔除

    def test_knn_keep_decisions_threshold_controls_retention(self):
        features, n_h = self._toy_pool()
        keep_low = builder.knn_keep_decisions(features, n_h, k=4, keep_ratio=0.2, backend="sklearn", metric="cosine")
        keep_high = builder.knn_keep_decisions(features, n_h, k=4, keep_ratio=0.5, backend="sklearn", metric="cosine")
        # 边界 token 有害邻居占比约 0.25：阈值 0.2 保留，阈值 0.5 剔除
        self.assertTrue(bool(keep_low[3]))
        self.assertFalse(bool(keep_high[3]))

    def _imbalanced_pool(self):
        # 7 纯有害(x+) + 1 功能词有害(靠安全簇, idx7) + 2 安全(x-)；N_h=8 : N_s=2 = 4:1 不平衡
        features = torch.tensor(
            [
                [1.0, 0.0], [0.95, 0.05], [0.9, 0.1], [0.92, 0.0],
                [0.97, 0.03], [0.88, 0.12], [0.91, 0.06],
                [-0.8, 0.2],
                [-1.0, 0.0], [-0.95, 0.05],
            ]
        )
        return features, 8

    def test_knn_balanced_corrects_majority_class_bias(self):
        features, n_h = self._imbalanced_pool()
        std = builder.knn_keep_decisions(features, n_h, k=4, keep_ratio=0.5, backend="sklearn", balanced=False)
        bal = builder.knn_keep_decisions(features, n_h, k=4, keep_ratio=0.5, backend="sklearn", balanced=True)
        # 多数类(有害)主导下，标准投票保留靠安全簇的功能词；per-class 加权将其剔除
        self.assertTrue(bool(std[7]))
        self.assertFalse(bool(bal[7]))
        # 纯有害 token 在两种模式下都保留
        self.assertTrue(bool(std[:7].all()))
        self.assertTrue(bool(bal[:7].all()))

    def test_knn_balanced_default_false_matches_explicit_standard(self):
        features, n_h = self._imbalanced_pool()
        default = builder.knn_keep_decisions(features, n_h, k=4, keep_ratio=0.5, backend="sklearn")
        explicit = builder.knn_keep_decisions(features, n_h, k=4, keep_ratio=0.5, backend="sklearn", balanced=False)
        self.assertTrue(torch.equal(default, explicit))

    def test_knn_balanced_all_harmful_keeps_all(self):
        # 安全池为空(n_safe=0)：balanced 分支 w_s=0 → ratio=1.0，退化为全部保留，且不除零
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]])
        keep = builder.knn_keep_decisions(
            features, n_harmful=3, k=2, keep_ratio=0.5, backend="sklearn", balanced=True
        )
        self.assertTrue(bool(keep.all()))

    def test_knn_nearest_indices_auto_falls_back_to_sklearn(self):
        features, n_h = self._toy_pool()
        with patch.object(builder, "_faiss_available", return_value=False):
            idx_auto = builder._knn_nearest_indices(features, n_h, k=3, backend="auto", metric="cosine")
        idx_sklearn = builder._knn_nearest_indices(features, n_h, k=3, backend="sklearn", metric="cosine")
        self.assertEqual(tuple(idx_auto.shape), (n_h, 3))
        self.assertTrue(torch.equal(idx_auto, idx_sklearn))

    def test_select_knn_label_clean_equivalent_to_all_when_cache_missing(self):
        args = SimpleNamespace(token_selection="knn_label_clean", _sample_index=0)
        hidden = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        token_ids = torch.tensor([3, 4, 6])
        out_hidden, out_ids = builder.select_harmful_response_tokens(hidden, token_ids, {}, args)
        self.assertTrue(torch.equal(out_hidden, hidden))
        self.assertTrue(torch.equal(out_ids, token_ids))

    def test_select_knn_label_clean_applies_mask_and_detects_misalignment(self):
        hidden = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        token_ids = torch.tensor([3, 4, 6])
        args = SimpleNamespace(
            token_selection="knn_label_clean",
            _sample_index=0,
            _knn_keep_masks={0: torch.tensor([True, False, True])},
        )
        out_hidden, out_ids = builder.select_harmful_response_tokens(hidden, token_ids, {}, args)
        self.assertTrue(torch.equal(out_hidden, torch.tensor([[1.0, 1.0], [3.0, 3.0]])))
        self.assertTrue(torch.equal(out_ids, torch.tensor([3, 6])))
        # mask 与 token 序列长度不一致时必须报错，不得静默错位
        args._knn_keep_masks = {0: torch.tensor([True, False])}
        with self.assertRaises(RuntimeError):
            builder.select_harmful_response_tokens(hidden, token_ids, {}, args)

    def _knn_build_args(self):
        return SimpleNamespace(
            max_response_len=128,
            max_total_len=2048,
            target_layer=31,
            seed=42,
            knn_k=2,
            knn_keep_ratio=0.5,
            knn_metric="cosine",
            knn_backend="sklearn",
            knn_safe_pool_cap=0,
            knn_balanced=False,
        )

    def test_build_knn_keep_masks_does_not_consume_global_rng(self):
        args = self._knn_build_args()
        samples = [
            {"prompt": "prompt", "response": "harmful1"},
            {"prompt": "prompt", "response": "harmful2"},
        ]
        with patch.object(builder, "extract_target_layer_tokens", return_value=torch.zeros(4, 2)), patch.object(
            builder, "filter_response_tokens", return_value=(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([3, 6]))
        ), patch.object(
            builder, "filter_response_hidden_states", return_value=torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
        ), patch.object(
            builder.random, "choice", side_effect=AssertionError("global RNG consumed")
        ):
            masks = builder.build_knn_keep_masks(
                _Model(), _SequenceTokenizer(), samples, ["refusal1"], args, torch.device("cpu")
            )
        self.assertEqual(set(masks.keys()), {0, 1})

    def test_build_knn_keep_masks_aligns_masks_by_sample(self):
        args = self._knn_build_args()
        samples = [
            {"prompt": "prompt", "response": "harmful1"},
            {"prompt": "prompt", "response": "harmful2"},
        ]
        with patch.object(builder, "extract_target_layer_tokens", return_value=torch.zeros(4, 2)), patch.object(
            builder, "filter_response_tokens", return_value=(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([3, 6]))
        ), patch.object(
            builder, "filter_response_hidden_states", return_value=torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
        ):
            masks = builder.build_knn_keep_masks(
                _Model(), _SequenceTokenizer(), samples, ["refusal1"], args, torch.device("cpu")
            )
        # 每个样本的 mask 长度 == 过滤后有害 token 数（2），可被 select 无错位应用
        for idx in (0, 1):
            self.assertEqual(int(masks[idx].shape[0]), 2)
            apply_args = SimpleNamespace(
                token_selection="knn_label_clean", _sample_index=idx, _knn_keep_masks=masks
            )
            builder.select_harmful_response_tokens(
                torch.zeros(2, 2), torch.tensor([3, 6]), {}, apply_args
            )
        # 诊断统计已写入 args
        self.assertEqual(args._knn_stats["total_harmful_tokens"], 4)
        self.assertIn("retention", args._knn_stats)

    def test_main_writes_knn_ct_csd_bank_with_cli_metadata(self):
        class _LoadableModel(_Model):
            def to(self, device):
                self.device = device
                return self

            def eval(self):
                self.is_eval = True
                return self

        def fake_build(model, tokenizer, harmful, refusals, args, device):
            args._knn_stats = {
                "total_harmful_tokens": 2,
                "kept_harmful_tokens": 1,
                "removed_harmful_tokens": 1,
                "retention": 0.5,
                "safe_pool_tokens": 2,
                "degenerate": False,
                "knn_k": int(args.knn_k),
                "knn_keep_ratio": float(args.knn_keep_ratio),
                "knn_metric": str(args.knn_metric),
                "knn_backend": str(args.knn_backend),
                "knn_safe_pool_cap": int(args.knn_safe_pool_cap),
                "knn_balanced": bool(getattr(args, "knn_balanced", False)),
                "removed_top_terms": [],
                "kept_top_terms": [],
            }
            return {0: torch.tensor([True, False])}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            harmful_path = root / "harmful.json"
            refusals_path = root / "refusals.txt"
            output_dir = root / "out"
            harmful_path.write_text('[{"prompt": "prompt", "response": "harmful1"}]', encoding="utf-8")
            refusals_path.write_text("refusal1\n", encoding="utf-8")

            with patch.object(builder.AutoTokenizer, "from_pretrained", return_value=_SequenceTokenizer()), patch.object(
                builder.AutoModel, "from_pretrained", return_value=_LoadableModel()
            ), patch.object(
                builder,
                "fit_route_preprocess",
                return_value=(
                    {
                        "mode": "center_pca128_l2",
                        "mean": torch.zeros(2),
                        "pca_components": torch.eye(2),
                        "pca_dim": 2,
                        "requested_pca_dim": 128,
                    },
                    0,
                ),
            ), patch.object(builder, "build_knn_keep_masks", side_effect=fake_build) as build_mock, patch.object(
                builder, "fit_minibatch_kmeans", return_value=(object(), torch.tensor([0.5, 0.5]), 0)
            ), patch.object(
                builder,
                "accumulate_cluster_sums",
                return_value=(torch.tensor([[2.0, 0.0], [0.0, 6.0]]), torch.tensor([2, 3]), 0),
            ):
                builder.main(
                    [
                        "--model_path", "dummy-model",
                        "--harmful_json", str(harmful_path),
                        "--refusals_txt", str(refusals_path),
                        "--output_dir", str(output_dir),
                        "--method", "ct_csd",
                        "--num_total_clusters", "2",
                        "--device", "cpu",
                        "--token_selection", "knn_label_clean",
                        "--knn_k", "6",
                        "--knn_keep_ratio", "0.5",
                        "--knn_metric", "cosine",
                        "--knn_backend", "sklearn",
                        "--knn_balanced",
                        "--feature_preprocess", "center_pca128_l2",
                    ]
                )

            state = torch.load(output_dir / "ct_csd_bank.pt", map_location="cpu", weights_only=True)
            summary = json.loads((output_dir / "knn_label_clean_summary.json").read_text(encoding="utf-8"))

        self.assertTrue(build_mock.called)
        self.assertEqual(state["config"]["token_selection"], "knn_label_clean")
        self.assertEqual(state["config"]["knn_k"], 6)
        self.assertEqual(state["config"]["knn_keep_ratio"], 0.5)
        self.assertEqual(state["config"]["knn_metric"], "cosine")
        self.assertEqual(state["config"]["knn_backend"], "sklearn")
        self.assertEqual(state["config"]["knn_retention_ratio"], 0.5)
        self.assertTrue(state["config"]["knn_balanced"])
        self.assertEqual(summary["total_harmful_tokens"], 2)
        self.assertFalse(summary["degenerate"])
        self.assertTrue(summary["knn_balanced"])


if __name__ == "__main__":
    unittest.main()
