import random
import unittest
import weakref

from nrs_experiment.core import (
    CVRPDecision,
    CvrpPerturbResult,
    CvrpRefinedCandidate,
    Instance,
    STARStrategy,
    SparseEdgeMemory,
    choose_tsp_perturb_start,
    flip_tsp_section,
    perturb_cvrp,
    perturb_tsp,
    refine_tsp_local,
    select_cvrp_neural_decision_with_memory,
    tsp_cost,
    validate_cvrp,
    validate_tsp,
)


class SequencePolicy:
    policy_id = "sequence"

    def __init__(self, picks):
        self.picks = list(picks)

    def select_next(self, instance, current, candidates, rng, prefix, *, repair=False):
        del instance, current, rng, prefix, repair
        if self.picks:
            picked = self.picks.pop(0)
            if picked in candidates:
                return picked
        return candidates[0]


class RecordingPolicy:
    policy_id = "recording"

    def __init__(self):
        self.calls = []

    def select_next(self, instance, current, candidates, rng, prefix, *, repair=False):
        del rng, repair
        self.calls.append((instance, current, list(candidates), list(prefix)))
        return candidates[0]


class BreakPolicy:
    policy_id = "break"

    def __init__(self, decisions):
        self.decisions = list(decisions)


class FakeNeuralPolicy:
    policy_id = "sil"

    def action_candidate_probabilities_batch(
        self,
        instance,
        prefixes_batch,
        candidates_batch,
        *,
        repair=False,
        allowed_candidates=None,
    ):
        del instance, repair, allowed_candidates
        rows = []
        for prefix, candidates in zip(prefixes_batch, candidates_batch):
            current = prefix[-1]
            weights = []
            for node in candidates:
                if current == 1 and node == 2:
                    weights.append(0.01)
                elif current == 1:
                    weights.append(0.99)
                else:
                    weights.append(0.01)
            total = sum(weights)
            rows.append([weight / total for weight in weights])
        return rows


class FakeCVRPNeuralPolicy:
    policy_id = "sil"

    def __init__(self, direct, via_depot):
        self.direct = dict(direct)
        self.via_depot = dict(via_depot)

    def action_probabilities(self, instance, prefix, remaining_capacity):
        del instance, prefix, remaining_capacity
        return dict(self.direct), dict(self.via_depot)


def tiny_tsp():
    return Instance(
        name="tiny-tsp",
        problem="tsp",
        coords={1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)},
        demands={},
        capacity=None,
        bks_cost=None,
    )


def tiny_cvrp():
    return Instance(
        name="tiny-cvrp",
        problem="cvrp",
        coords={1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (0, 1), 5: (0, 2)},
        demands={1: 0, 2: 1, 3: 1, 4: 1, 5: 1},
        capacity=2,
        bks_cost=None,
    )


class STARStrategyTests(unittest.TestCase):
    def test_tsp_successor_pick_is_noop(self):
        instance = tiny_tsp()
        route = [1, 2, 3, 4]
        policy = SequencePolicy([3, 4, 1])
        candidate, changed = perturb_tsp(instance, route, policy, random.Random(1), min_new_edges=3)
        self.assertEqual(candidate, route)
        self.assertTrue(changed.issubset(set(route)))

    def test_tsp_relocation_preserves_tour_and_budget(self):
        instance = tiny_tsp()
        route = [1, 2, 3, 4]
        policy = SequencePolicy([4, 1])
        candidate, changed = perturb_tsp(instance, route, policy, random.Random(1), min_new_edges=1)
        self.assertTrue(validate_tsp(instance, candidate))
        self.assertEqual(len(candidate), 4)
        self.assertTrue({2, 4}.issubset(changed))

    def test_tsp_local_refine_can_fix_crossing(self):
        instance = tiny_tsp()
        route = [1, 3, 2, 4]
        refined = refine_tsp_local(instance, route, {1, 2, 3, 4}, refine_k=3)
        self.assertTrue(validate_tsp(instance, refined))
        self.assertLessEqual(tsp_cost(instance, refined), tsp_cost(instance, route))

    def test_neighbor_order_purges_stale_instance_cache(self):
        import nrs_experiment.core as core

        stale_instance = Instance(
            name="stale-tsp",
            problem="tsp",
            coords={65: (0, 0), 66: (1, 0), 67: (2, 0), 68: (3, 0)},
            demands={},
            capacity=None,
            bks_cost=None,
        )
        instance = tiny_tsp()
        instance_key = id(instance)
        core._INSTANCE_CACHE_REFS[instance_key] = weakref.ref(stale_instance)
        core._STAR_CONTEXT_CACHE[instance_key] = object()

        neighbor_order = core.STAR_tsp_neighbor_order(instance, 3)

        self.assertEqual(len(neighbor_order), 4)
        self.assertNotEqual(neighbor_order[0], [65, 66, 67])
        self.assertTrue(set(neighbor_order[0]).issubset(instance.coords))

    def test_neighbor_order_reuses_persistent_cpp_context(self):
        import nrs_experiment.core as core

        instance = tiny_tsp()
        core._purge_instance_caches(id(instance))
        original_ext = core.STAR

        class CountingSTAR:
            build_count = 0

            def __init__(self, nodes, coords, demands, depot, capacity, edge_weight_type):
                CountingSTAR.build_count += 1
                self.inner = original_ext.STAR(nodes, coords, demands, depot, capacity, edge_weight_type)

            def neighbor_order(self, total):
                return self.inner.neighbor_order(total)

        class CountingExt:
            STAR = CountingSTAR

        core.STAR = CountingExt
        try:
            order3 = core.STAR_tsp_neighbor_order(instance, 3)
            order2 = core.STAR_tsp_neighbor_order(instance, 2)
        finally:
            core.STAR = original_ext

        self.assertEqual(CountingSTAR.build_count, 1)
        self.assertEqual(order2, [row[:2] for row in order3])

    def test_no_memory_ignores_memory_update_mode(self):
        instance = tiny_tsp()
        strategy = STARStrategy(iterations=0, memory=False)

        initial, final, valid = strategy.run(instance, SequencePolicy([]), random.Random(0))

        self.assertTrue(valid)
        self.assertEqual(initial, final)

    def test_tsp_cost_start_prefers_long_edge(self):
        instance = Instance(
            name="long-edge-tsp",
            problem="tsp",
            coords={1: (0, 0), 2: (100, 0), 3: (101, 0), 4: (102, 0)},
            demands={},
            capacity=None,
            bks_cost=None,
        )
        route = [1, 2, 3, 4]
        positions = {node: index for index, node in enumerate(route)}

        picked, info = choose_tsp_perturb_start(
            instance,
            route,
            positions,
            SequencePolicy([]),
            random.Random(0),
            edge_memory=None,
            neural_knn_k=3,
            neural_knn_mask=True,
            start_mode="cost",
            start_probes=4,
            cost_weight=1.0,
            policy_weight=1.0,
            memory_weight=1.0,
        )

        self.assertEqual(picked, 4)
        self.assertEqual(info.mode, "cost")

    def test_tsp_policy_disagreement_requires_neural_policy(self):
        instance = tiny_tsp()
        route = [1, 2, 3, 4]
        positions = {node: index for index, node in enumerate(route)}

        with self.assertRaises(ValueError):
            choose_tsp_perturb_start(
                instance,
                route,
                positions,
                SequencePolicy([]),
                random.Random(0),
                edge_memory=None,
                neural_knn_k=3,
                neural_knn_mask=True,
                start_mode="policy-disagreement",
                start_probes=4,
                cost_weight=1.0,
                policy_weight=1.0,
                memory_weight=1.0,
            )

    def test_tsp_hybrid_start_uses_policy_disagreement(self):
        import nrs_experiment.core as core

        instance = tiny_tsp()
        route = [1, 2, 3, 4]
        positions = {node: index for index, node in enumerate(route)}
        original = core.NativeTSPNeuralPolicy
        core.NativeTSPNeuralPolicy = FakeNeuralPolicy
        try:
            picked, info = choose_tsp_perturb_start(
                instance,
                route,
                positions,
                FakeNeuralPolicy(),
                random.Random(0),
                edge_memory=SparseEdgeMemory(instance, k=3),
                neural_knn_k=3,
                neural_knn_mask=True,
                start_mode="hybrid",
                start_probes=4,
                cost_weight=0.0,
                policy_weight=100.0,
                memory_weight=0.0,
            )
        finally:
            core.NativeTSPNeuralPolicy = original

        self.assertEqual(picked, 1)
        self.assertGreater(info.policy_score, 0.0)
        self.assertLess(info.successor_prob or 0.0, info.best_alt_prob or 0.0)

    def test_advantage_memory_reinforces_only_sparse_introduced_edges(self):
        instance = tiny_tsp()
        memory = SparseEdgeMemory(instance, k=3, rho=1.0, tau_min=0.1, tau_max=1.0)

        reinforced = memory.update_from_advantage_edges({(1, 3), (3, 1)}, strength=1.0)

        self.assertGreaterEqual(reinforced, 1)
        self.assertEqual(memory.values[(1, 3)], 1.0)
        self.assertEqual(memory.values[(1, 2)], 0.1)

    def test_tsp_flip_section_matches_reference_half_open_segment(self):
        route = [1, 2, 3, 4, 5]
        flip_tsp_section(route, 2, 4)
        self.assertEqual(route, [1, 3, 2, 4, 5])

        route = [1, 2, 3, 4, 5]
        flip_tsp_section(route, 2, 5)
        self.assertEqual(route, [5, 2, 3, 4, 1])

    def test_cvrp_route_break_preserves_feasibility(self):
        instance = tiny_cvrp()
        routes = [[2, 3], [4, 5]]
        policy = BreakPolicy([])

        def select_decision(instance, policy, current, candidates, rng, prefix, remaining_capacity):
            del instance, policy, current, rng, prefix, remaining_capacity
            return CVRPDecision(candidates[0], True)

        import nrs_experiment.core as core

        original = core.select_cvrp_decision
        core.select_cvrp_decision = select_decision
        try:
            candidate, changed = perturb_cvrp(instance, routes, policy, random.Random(1), min_new_edges=1)
        finally:
            core.select_cvrp_decision = original

        self.assertTrue(validate_cvrp(instance, candidate))
        self.assertIn(4, changed)

    def test_cvrp_sil_probability_memory_selects_route_break_action(self):
        instance = tiny_cvrp()
        memory = SparseEdgeMemory(instance, k=4, rho=1.0)
        policy = FakeCVRPNeuralPolicy(
            direct={2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01},
            via_depot={2: 0.01, 3: 0.99, 4: 0.01, 5: 0.01},
        )

        decision = select_cvrp_neural_decision_with_memory(
            instance,
            policy,
            current=2,
            feasible_now=[4],
            feasible_full=[3, 4],
            rng=random.Random(0),
            prefix=[2],
            remaining_capacity=1,
            edge_memory=memory,
        )

        self.assertEqual(decision, CVRPDecision(3, True))

    def test_cvrp_advantage_introduced_reinforces_introduced_edges(self):
        instance = tiny_cvrp()
        memory = SparseEdgeMemory(instance, k=4, rho=1.0, tau_min=0.1, tau_max=1.0)
        strategy = STARStrategy(memory_update_mode="advantage-introduced")
        candidate = CvrpRefinedCandidate(
            routes=[[2, 4], [3, 5]],
            cost=9.0,
            perturb=CvrpPerturbResult(
                routes=[[2, 4], [3, 5]],
                changed={2, 3, 4, 5},
                introduced_edges={(2, 4), (4, 2)},
            ),
        )

        self.assertEqual(strategy.effective_memory_update_mode(instance), "advantage-introduced")

        strategy.update_cvrp_memory_from_candidates(memory, source_cost=10.0, refined_candidates=[candidate])

        self.assertEqual(memory.values[(2, 4)], 1.0)
        self.assertEqual(memory.values[(1, 2)], 0.1)

    def test_strategy_reports_valid_cvrp_with_stub_policy(self):
        instance = tiny_cvrp()
        strategy = STARStrategy(iterations=1, min_new_edges=1, refine_k=2)
        initial, final, valid = strategy.run(instance, SequencePolicy([]), random.Random(0))
        self.assertTrue(valid)
        self.assertLessEqual(final, initial)


if __name__ == "__main__":
    unittest.main()
