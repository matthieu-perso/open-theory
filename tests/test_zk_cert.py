import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from zk_cert_tool import (  # noqa: E402
    DEFAULT_OPENECONOMICS_AGENT_PASSPORT_PRIVATE,
    DEFAULT_OPENECONOMICS_AGENT_PASSPORT_PUBLIC,
    DEFAULT_OPENECONOMICS_PRIVATE,
    DEFAULT_OPENECONOMICS_PUBLIC,
    DEFAULT_OPENTHEORY_AGENT_PASSPORT_PRIVATE,
    DEFAULT_OPENTHEORY_AGENT_PASSPORT_PUBLIC,
    DEFAULT_OPENTHEORY_PRIVATE,
    DEFAULT_OPENTHEORY_PUBLIC,
    HomomorphicMultiStepTrajectoryAccumulator,
    PRIME_P,
    extract_witness_from_forked_transcripts,
    generate_zk_certificate,
    mod_p,
    pack_zk_WireHeader_binary,
    pedersen_linear_commit_fp,
    prove_bit_range_fp,
    simulate_hvzk_transcript_fp,
    sigma_prove_fp,
    verify_bit_range_fp,
    verify_zk_WireHeader_binary,
    verify_zk_certificate,
)
from zk_economic_escrow_tool import (  # noqa: E402
    DEFAULT_PRIVATE_ARROW_ALPHA,
    DEFAULT_PRIVATE_BALANCE_SHEET,
    DEFAULT_PRIVATE_EXECUTION_HISTORY,
    DEFAULT_PUBLIC_ARROW_BUYER_OFFER,
    DEFAULT_PUBLIC_EXECUTION_AUDIT,
    DEFAULT_PUBLIC_SOLVENCY_ENVELOPE,
    generate_zk_agent_passport_and_escrow,
    verify_zk_agent_passport_and_escrow,
)
from pytorch_telemetry_bridge import (  # noqa: E402
    EmpiricalTwoBatchConstantEstimator,
    RealMatrixNeuralTrainer,
    build_zero_sweep_arbitrage_payload,
    compute_two_batch_hilbert_polarization,
)
from market_macro_telemetry_bridge import (  # noqa: E402
    build_combinatorial_and_regime_payload,
    build_live_market_macro_payload,
)


class ZeroKnowledgeCertTests(unittest.TestCase):
    def test_opentheory_zk_certificate_valid_and_private(self):
        cert = generate_zk_certificate("OpenTheory", DEFAULT_OPENTHEORY_PRIVATE, DEFAULT_OPENTHEORY_PUBLIC)
        res = verify_zk_certificate(cert)
        self.assertTrue(res["valid"])
        self.assertEqual(res["domain"], "OpenTheory")
        self.assertTrue(res["zero_knowledge_verified"])
        self.assertEqual(res["slack_proofs_verified"], 3)

        serialized = json.dumps(cert)
        for secret_key in ("sr_active", "md_dense", "Lambda_raw_num", "eps_token_num", "blindingSalt"):
            self.assertNotIn(secret_key, serialized)

    def test_openeconomics_zk_certificate_valid_and_private(self):
        cert = generate_zk_certificate("OpenEconomics", DEFAULT_OPENECONOMICS_PRIVATE, DEFAULT_OPENECONOMICS_PUBLIC)
        res = verify_zk_certificate(cert)
        self.assertTrue(res["valid"])
        self.assertEqual(res["domain"], "OpenEconomics")
        self.assertTrue(res["zero_knowledge_verified"])
        self.assertEqual(res["slack_proofs_verified"], 6)

        serialized = json.dumps(cert)
        for secret_key in ("v_priv_num", "s_stress_num", "cov_ZY_num", "delta_IV_num", "blindingSalt"):
            self.assertNotIn(secret_key, serialized)

    def test_three_pillar_zk_agent_passport_and_arrow_escrow(self):
        passport = generate_zk_agent_passport_and_escrow(
            DEFAULT_PRIVATE_BALANCE_SHEET,
            DEFAULT_PUBLIC_SOLVENCY_ENVELOPE,
            DEFAULT_PRIVATE_ARROW_ALPHA,
            DEFAULT_PUBLIC_ARROW_BUYER_OFFER,
            DEFAULT_PRIVATE_EXECUTION_HISTORY,
            DEFAULT_PUBLIC_EXECUTION_AUDIT,
            escrow_amount_usd=250000,
        )
        res = verify_zk_agent_passport_and_escrow(passport)
        self.assertTrue(res["valid"])
        self.assertEqual(res["escrowSettlementStatus"], "ReleasedToAgent")
        self.assertEqual(res["escrowReleasedAmountUSD"], 250000)
        self.assertTrue(res["pillar_1_has_solvency_verified"])
        self.assertTrue(res["pillar_2_can_do_arrow_paradox_resolved"])
        self.assertTrue(res["pillar_3_did_almgren_chriss_and_anti_collusion_verified"])
        self.assertEqual(res["primary_proofs_verified"], 3)
        self.assertEqual(res["r1cs_slack_proofs_verified"], 8)

        # Confirm zero private witness leakage
        raw = json.dumps(passport)
        for forbidden in (
            "q_asset1",
            "cash_reserves",
            "cov_ZY_num",
            "d_matched_dim",
            "v_executed_num",
            "pnl_period1",
            "balanceSheetSalt",
            "alphaSalt",
            "historySalt",
        ):
            self.assertNotIn(f'"{forbidden}"', raw)

        # Tampering with public ask price must reject and refund principal
        tampered = json.loads(raw)
        tampered["public_envelopes"]["pillar_2_can_do_arrow_offer"]["price_info_ask"] = 100
        res_tampered = verify_zk_agent_passport_and_escrow(tampered)
        self.assertFalse(res_tampered["valid"])
        self.assertEqual(res_tampered["escrowSettlementStatus"], "RefundedToPrincipal")

    def test_two_batch_hilbert_zero_sweep_arbitrage_polarization(self):
        pol = compute_two_batch_hilbert_polarization(
            [6.0, 4.0, 2.0, 2.0],
            [2.0, 4.0, -2.0, 2.0],
        )
        self.assertEqual(pol["g1_proj"], 6)
        self.assertEqual(pol["g2_proj"], 2)
        self.assertEqual(pol["hilbert_signal_sq_est"], 28.0)
        self.assertEqual(pol["hilbert_noise_var_est"], 16.0)
        self.assertAlmostEqual(pol["direct_wiener_filter_float"], 8.0 / 40.0)

        payload = build_zero_sweep_arbitrage_payload("unit-test-2batch-001")
        self.assertEqual(payload["runId"], "unit-test-2batch-001")
        self.assertEqual(payload["d_dense_dim"] // payload["d_matched_dim"], 8)
        self.assertEqual(payload["B_sgd_num"] // payload["B_rpm_num"], 4)
        self.assertEqual(payload["d_kv_full"] // payload["r_kv_latent"], 8)
        self.assertEqual(payload["K_rlvr_search"], 8)

    def test_combinatorial_clearing_and_eprocess_regime_break_payloads(self):
        normal = build_combinatorial_and_regime_payload("normal-regime", crisis_mode=False)
        crisis = build_combinatorial_and_regime_payload("crisis-regime", crisis_mode=True)
        self.assertLessEqual(normal["e_process_milli"], normal["ville_threshold_milli"])
        self.assertGreater(crisis["e_process_milli"], crisis["ville_threshold_milli"])

        live_mm = build_live_market_macro_payload("live-mm-001")
        self.assertTrue(live_mm["usesFrequentBatchAuction"])
        self.assertGreater(
            live_mm["gross_exposure_long"] + live_mm["gross_exposure_short"],
            live_mm["posted_zk_margin"] * 10,
        )

    def test_tampering_with_public_envelope_or_proof_is_rejected(self):
        cert = generate_zk_certificate("OpenTheory", DEFAULT_OPENTHEORY_PRIVATE, DEFAULT_OPENTHEORY_PUBLIC)

        # 1. Tamper with public safety bound B_safe_num
        tampered_env = json.loads(json.dumps(cert))
        tampered_env["public_envelope"]["B_safe_num"] = 999
        res_env = verify_zk_certificate(tampered_env)
        self.assertFalse(res_env["valid"])
        self.assertIn("challenge mismatch", res_env["reason"])

        # 2. Tamper with Sigma-protocol response z_w
        tampered_z = json.loads(json.dumps(cert))
        orig_z = int(tampered_z["cryptographic_proof"]["primary_witness_proof"]["response_z_hex"], 16)
        tampered_z["cryptographic_proof"]["primary_witness_proof"]["response_z_hex"] = hex((orig_z + 1) % PRIME_P)
        res_z = verify_zk_certificate(tampered_z)
        self.assertFalse(res_z["valid"])
        self.assertIn("Sigma-protocol algebraic verification", res_z["reason"])

        # 3. Tamper with R1CS slack proof
        tampered_slack = json.loads(json.dumps(cert))
        slack_entry = tampered_slack["cryptographic_proof"]["r1cs_slack_proofs"]["barrier_margin_slack"]
        orig_sz = int(slack_entry["response_z_hex"], 16)
        slack_entry["response_z_hex"] = hex((orig_sz + 5) % PRIME_P)
        res_slack = verify_zk_certificate(tampered_slack)
        self.assertFalse(res_slack["valid"])
        self.assertIn("R1CS slack proof", res_slack["reason"])

    def test_invalid_private_telemetry_refuses_to_generate_certificate(self):
        bad_priv = dict(DEFAULT_OPENTHEORY_PRIVATE)
        bad_priv["sr_active"] = 1000
        with self.assertRaises(ValueError) as ctx:
            generate_zk_certificate("OpenTheory", bad_priv, DEFAULT_OPENTHEORY_PUBLIC)
        self.assertIn("Cannot issue ZK certificate", str(ctx.exception))

    def test_two_special_soundness_extractor_and_hvzk_simulator(self):
        w_secret = 42424242
        r_secret = 99999999
        tr1 = sigma_prove_fp("TestDomain", w_secret, r_secret, {"id": 1}, nonce_kw=111, nonce_kr=222)
        tr2 = sigma_prove_fp("TestDomain", w_secret, r_secret, {"id": 2}, nonce_kw=111, nonce_kr=222)
        w_ext, r_ext = extract_witness_from_forked_transcripts(tr1, tr2)
        self.assertEqual(w_ext, mod_p(w_secret))
        self.assertEqual(r_ext, mod_p(r_secret))

        C = pedersen_linear_commit_fp(w_secret, r_secret)
        sim_tr = simulate_hvzk_transcript_fp(C, challenge_c=12345)
        lhs = (
            int(sim_tr["generator_g_hex"], 16) * int(sim_tr["response_z_hex"], 16)
            + int(sim_tr["generator_h_hex"], 16) * int(sim_tr["response_r_hex"], 16)
        ) % PRIME_P
        rhs = (int(sim_tr["announce_hex"], 16) + int(sim_tr["challenge_hex"], 16) * C) % PRIME_P
        self.assertEqual(lhs, rhs)

    def test_cli_demo_roundtrip(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "zk_cert_tool.py"), "demo"],
            capture_output=True,
            text=True,
            check=True,
        )
        out = json.loads(proc.stdout)
        self.assertTrue(out["Bilateral_Handshake_Success"])

        proc_escrow = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "zk_economic_escrow_tool.py"), "demo"],
            capture_output=True,
            text=True,
            check=True,
        )
        out_escrow = json.loads(proc_escrow.stdout)
        self.assertTrue(out_escrow["valid"])
        self.assertEqual(out_escrow["escrowSettlementStatus"], "ReleasedToAgent")

    def test_homomorphic_bit_range_proof_fp(self):
        slack_val = 42
        slack_r = 987654321
        pub_env = {"max_bound": 100}
        c_slack = pedersen_linear_commit_fp(slack_val, slack_r)
        br_proof = prove_bit_range_fp("TestBitRange", slack_val, slack_r, pub_env, num_bits=8)
        ok, msg = verify_bit_range_fp("TestBitRange", c_slack, br_proof, pub_env)
        self.assertTrue(ok, msg)
        self.assertIn("bit_or_disjunctive_proofs", br_proof)
        self.assertEqual(len(br_proof["bit_or_disjunctive_proofs"]), 8)
        self.assertIn("schwartz_zippel_batch", br_proof)

        # Tampering with a single bit commitment breaks the homomorphic sum 2^i * C_{b_i} == C_slack
        tampered = json.loads(json.dumps(br_proof))
        orig_c0 = int(tampered["bit_sigma_proofs"][0]["commitment_fp_hex"], 16)
        tampered["bit_sigma_proofs"][0]["commitment_fp_hex"] = hex((orig_c0 + 1) % PRIME_P)
        ok_bad, _ = verify_bit_range_fp("TestBitRange", c_slack, tampered, pub_env)
        self.assertFalse(ok_bad)

        # Tampering with a CDS94 1-of-2 Bit OR-proof branch response z0 is rejected
        tampered_or = json.loads(json.dumps(br_proof))
        orig_z0 = int(tampered_or["bit_or_disjunctive_proofs"][0]["z0_hex"], 16)
        tampered_or["bit_or_disjunctive_proofs"][0]["z0_hex"] = hex((orig_z0 + 1) % PRIME_P)
        ok_bad_or, msg_bad_or = verify_bit_range_fp("TestBitRange", c_slack, tampered_or, pub_env)
        self.assertFalse(ok_bad_or)
        self.assertIn("OR-proof", msg_bad_or)

        # Tampering with Schwartz-Zippel batch aggregate is rejected
        tampered_sz = json.loads(json.dumps(br_proof))
        orig_zw_agg = int(tampered_sz["schwartz_zippel_batch"]["z_w_agg_hex"], 16)
        tampered_sz["schwartz_zippel_batch"]["z_w_agg_hex"] = hex((orig_zw_agg + 1) % PRIME_P)
        ok_bad_sz, msg_bad_sz = verify_bit_range_fp("TestBitRange", c_slack, tampered_sz, pub_env)
        self.assertFalse(ok_bad_sz)
        self.assertIn("Schwartz-Zippel", msg_bad_sz)

    def test_opentheory_and_openeconomics_agent_passports_with_bit_range_proofs(self):
        ot_pass = generate_zk_certificate(
            "OpenTheoryAgentPassport",
            DEFAULT_OPENTHEORY_AGENT_PASSPORT_PRIVATE,
            DEFAULT_OPENTHEORY_AGENT_PASSPORT_PUBLIC,
        )
        res_ot = verify_zk_certificate(ot_pass)
        self.assertTrue(res_ot["valid"])
        self.assertTrue(res_ot["dlp_binding_verified"])
        self.assertEqual(res_ot["domain"], "OpenTheoryAgentPassport")
        self.assertGreater(res_ot["bit_range_proofs_verified"], 0)
        raw_ot = json.dumps(ot_pass)
        for secret_key in ("cot_trace_hash", "sys_prompt_hash", "C_avail_flops"):
            self.assertNotIn(f'"{secret_key}"', raw_ot)

        # Tampering with 2048-bit DLP-binding proof is rejected
        tampered_dlp = json.loads(raw_ot)
        orig_dlp_z = int(tampered_dlp["cryptographic_proof"]["primary_dlp_binding_proof"]["response_z_w_hex"], 16)
        tampered_dlp["cryptographic_proof"]["primary_dlp_binding_proof"]["response_z_w_hex"] = hex(orig_dlp_z + 1)
        res_bad_dlp = verify_zk_certificate(tampered_dlp)
        self.assertFalse(res_bad_dlp["valid"])
        self.assertIn("DLP", res_bad_dlp["reason"])

        oe_pass = generate_zk_certificate(
            "OpenEconomicsAgentPassport",
            DEFAULT_OPENECONOMICS_AGENT_PASSPORT_PRIVATE,
            DEFAULT_OPENECONOMICS_AGENT_PASSPORT_PUBLIC,
        )
        res_oe = verify_zk_certificate(oe_pass)
        self.assertTrue(res_oe["valid"])
        self.assertTrue(res_oe["dlp_binding_verified"])
        self.assertEqual(res_oe["domain"], "OpenEconomicsAgentPassport")
        self.assertEqual(res_oe["escrow_settlement_status"], "ReleasedToAgent")
        self.assertGreater(res_oe["bit_range_proofs_verified"], 0)
        raw_oe = json.dumps(oe_pass)
        for secret_key in ("q_asset1", "cash_reserves", "pnl_period1"):
            self.assertNotIn(f'"{secret_key}"', raw_oe)

    def test_tier1_closed_loop_autopilot_and_crypto_tradfi_execution_engine(self):
        from market_macro_telemetry_bridge import (
            build_high_roi_verticals_payload,
            compute_crypto_tradfi_live_execution_decision,
        )
        from pytorch_telemetry_bridge import run_closed_loop_autopilot_demo

        # 1. Verify Tier 1.1 Closed-Loop Zero-Sweep Training & Grid Arbitrage Autopilot
        autopilot_report = run_closed_loop_autopilot_demo()
        self.assertEqual(autopilot_report["autopilot_status"], "ACTIVE_CLOSED_LOOP")
        self.assertEqual(
            autopilot_report["final_param_group_state"]["optimizer_family"],
            "WienerDampedSpectralMuon",
        )
        self.assertAlmostEqual(
            autopilot_report["identified_3point_trajectory"]["contraction_gamma"],
            0.6,
            places=5,
        )
        self.assertAlmostEqual(
            autopilot_report["identified_3point_trajectory"]["asymptotic_floor_L_inf"],
            1.2,
            places=5,
        )
        # During normal LMP ($150/MWh) -> 50 MW AI compute; during spike ($300/MWh) -> 0 MW AI compute
        self.assertEqual(autopilot_report["step_execution_logs"][0]["allocated_ai_mw"], 50.0)
        self.assertEqual(autopilot_report["step_execution_logs"][-1]["allocated_ai_mw"], 0.0)
        self.assertEqual(
            autopilot_report["step_execution_logs"][-1]["curtailed_grid_export_mw"],
            100.0,
        )

        # 2. Verify Tier 1.2 Live Crypto/TradFi Execution, SOR, Stat-Arb, Capital Relief & Grid Arbitrage
        calm_payload = build_high_roi_verticals_payload(crisis_mode=False)
        calm_decision = compute_crypto_tradfi_live_execution_decision(calm_payload)
        self.assertFalse(calm_decision["ou_stat_arb_engine"]["is_cointegration_break_detected"])
        self.assertFalse(calm_decision["ou_stat_arb_engine"]["should_halt_and_liquidate"])
        self.assertEqual(calm_decision["ou_stat_arb_engine"]["action"], "DEPLOY_MEAN_REVERSION_SPREAD")
        self.assertGreater(calm_decision["cross_venue_smart_order_router"]["savings_vs_lit_only"], 0.0)
        self.assertGreater(calm_decision["cross_venue_smart_order_router"]["savings_vs_dark_amm_only"], 0.0)
        self.assertEqual(
            calm_decision["basel_and_prime_broker_capital_relief"]["freed_collateral_dollars"],
            9_600_000.0,
        )
        self.assertTrue(
            calm_decision["basel_and_prime_broker_capital_relief"]["is_zk_solvency_verified"]
        )
        self.assertEqual(
            calm_decision["ai_compute_power_grid_arbitrage"]["optimal_ai_compute_mw"],
            50.0,
        )

        crisis_payload = build_high_roi_verticals_payload(crisis_mode=True)
        crisis_decision = compute_crypto_tradfi_live_execution_decision(crisis_payload)
        self.assertTrue(crisis_decision["ou_stat_arb_engine"]["is_cointegration_break_detected"])
        self.assertTrue(crisis_decision["ou_stat_arb_engine"]["should_halt_and_liquidate"])
        self.assertEqual(crisis_decision["ou_stat_arb_engine"]["optimal_stat_arb_weight"], 0.0)
        self.assertEqual(crisis_decision["ou_stat_arb_engine"]["action"], "EMERGENCY_LIQUIDATE_DEPEG")
        self.assertEqual(
            crisis_decision["ai_compute_power_grid_arbitrage"]["optimal_ai_compute_mw"],
            0.0,
        )
        self.assertGreater(
            crisis_decision["ai_compute_power_grid_arbitrage"]["hourly_grid_arbitrage_gain_dollars"],
            0.0,
        )

    def test_empirical_constant_estimator_and_real_matrix_neural_trainer(self):
        # 1. Verify EmpiricalTwoBatchConstantEstimator measures real constants without hardcoded flags
        estimator = EmpiricalTwoBatchConstantEstimator(seed=42)
        probe = estimator.probe_dataset_and_architecture(
            d_in=8,
            d_hidden=8,
            d_out=4,
            batch_size=16,
            noise_std=0.35,
            target_rank=3,
        )
        # Must measure positive gradient signal and batch noise variance from real forward/backward passes
        self.assertGreater(probe["measured_signal_sq"], 0.0)
        self.assertGreater(probe["measured_noise_var"], 0.0)
        self.assertGreater(probe["measured_c_comp"], 0.0)
        self.assertGreater(probe["measured_c_prop"], 0.0)
        self.assertGreaterEqual(probe["prescribed_optimal_depth_L"], 1)
        self.assertGreater(probe["prescribed_terminal_lr_eta_end"], 0.0)
        self.assertIn(
            probe["inferred_optimizer_family"],
            ("WienerDampedSpectralMuon", "PureNewtonSchulzMuon", "DiagonalAdamW", "IsotropicSGD"),
        )

        # 2. Verify RealMatrixNeuralTrainer executes genuine forward/backward passes and weight updates
        trainer = RealMatrixNeuralTrainer(
            d_in=8,
            d_hidden=8,
            d_out=4,
            seed=42,
            noise_std=0.45,
        )
        benchmark = trainer.benchmark_optimizers(steps=25, batch_size=12)

        # Must record actual validation loss trajectories for all 4 optimizers
        for opt_key in ("wiener_spectral_muon", "pure_newton_schulz_muon", "diagonal_adamw", "isotropic_sgd"):
            traj = benchmark["trajectories"][opt_key]
            self.assertEqual(len(traj), 26)  # step 0 .. 25

        # Wiener-Damped Spectral Muon + Token Curation + WSD must strictly reduce validation loss
        # from initialization AND strictly beat Isotropic SGD, Diagonal AdamW, and Undamped Pure Muon
        wiener_init = benchmark["trajectories"]["wiener_spectral_muon"][0]
        wiener_final = benchmark["trajectories"]["wiener_spectral_muon"][-1]
        pure_muon_final = benchmark["trajectories"]["pure_newton_schulz_muon"][-1]
        adamw_final = benchmark["trajectories"]["diagonal_adamw"][-1]
        sgd_final = benchmark["trajectories"]["isotropic_sgd"][-1]

        self.assertLess(wiener_final, wiener_init * 0.5)
        self.assertLess(wiener_final, pure_muon_final)
        self.assertLess(wiener_final, adamw_final)
        self.assertLess(wiener_final, sgd_final)
        self.assertTrue(benchmark["wiener_strictly_dominates_all_baselines"])

    def test_hyper_fast_256byte_binary_wire_packet_and_trajectory_folding(self):
        # 1. 256-byte binary X-ZK-Passport header roundtrip & tamper rejection
        pkt = pack_zk_WireHeader_binary(
            prompt_output_hash=12345678,
            K_search_executed=8,
            verifier_score_num=95,
            step_error_num=1,
        )
        self.assertEqual(len(pkt), 256)
        ok, msg = verify_zk_WireHeader_binary(pkt, use_native_c_fastpath=True)
        self.assertTrue(ok, msg)

        # Tampering with 1 byte in the 256-byte binary packet must reject
        bad_pkt = bytearray(pkt)
        bad_pkt[100] ^= 0x01
        ok_bad, _ = verify_zk_WireHeader_binary(bytes(bad_pkt), use_native_c_fastpath=True)
        self.assertFalse(ok_bad)

        # 2. Homomorphic Multi-Step Trajectory Accumulator (O(1) verification for 50 agent steps)
        acc = HomomorphicMultiStepTrajectoryAccumulator("agent-session-test")
        for step in range(50):
            acc.record_inference_step(42 + step, 999 + step, 111 + step, 222 + step)
        folded = acc.export_folded_proof()
        self.assertEqual(folded["steps_folded"], 50)
        self.assertTrue(HomomorphicMultiStepTrajectoryAccumulator.verify_folded_trajectory(folded))

        # Tampering with folded response must reject
        folded["z_w_agg_hex"] = hex((int(folded["z_w_agg_hex"], 16) + 1) % PRIME_P)
        self.assertFalse(HomomorphicMultiStepTrajectoryAccumulator.verify_folded_trajectory(folded))


if __name__ == "__main__":
    unittest.main()
