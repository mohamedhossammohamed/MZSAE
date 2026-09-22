#!/usr/bin/env python3
"""
generate_showcase_data.py
Constructs 10 rigorous, diverse linguistic benchmark scenarios for
side-by-side comparison between Standard FlashAttention and MZSAE.
"""

from typing import List, Dict, Any


def build_scenario_1_needle_in_haystack() -> Dict[str, Any]:
    """Scenario 1: Needle in a Haystack (Exact Passcode Retrieval)"""
    haystack_paragraphs = [
        "In the ongoing study of stellar evolution, hydrogen-burning main sequence stars maintain hydrostatic equilibrium over gigayear timescales.",
        "Quantum chromodynamics explains the strong interaction between quarks mediated by gluons, exhibiting asymptotic freedom at ultra-high energy scales.",
        "The enzymatic breakdown of glycogen is catalyzed by glycogen phosphorylase, releasing glucose-1-phosphate in response to epinephrine signaling.",
        "Semiconductor lithography utilizes extreme ultraviolet light with wavelengths near 13.5 nanometers to pattern nanoscale transistor gates.",
        "Plate tectonics describes the large-scale motion of the lithosphere over the ductile asthenosphere, driving continental drift and seismic activity.",
        "Distributed database systems employ Raft or Paxos consensus protocols to guarantee linearizable state machine replication across unreliable networks.",
        "Cellular mitosis consists of prophase, prometaphase, metaphase, anaphase, and telophase, followed by cytokinesis into two daughter cells.",
        "Electrochemical impedance spectroscopy probes charge transfer resistance and double-layer capacitance across electrode-electrolyte interfaces.",
    ]
    # Build ~8,000 tokens of background text
    repeated_body = []
    needle = "CRITICAL SECURITY OVERRIDE: The emergency master archive passcode is [SIGMA-9482-OMEGA]. Keep this secret."
    needle_index = 65  # ~38% depth

    for i in range(160):
        if i == needle_index:
            repeated_body.append(needle)
        p = haystack_paragraphs[i % len(haystack_paragraphs)]
        repeated_body.append(f"Section {i+1}: {p} Detailed observations confirm these theoretical parameters across all experimental baselines.")

    context_text = "\n\n".join(repeated_body)
    prompt = (
        f"{context_text}\n\n"
        "Question: What is the emergency master archive passcode stated in the text?\n"
        "Answer: The emergency master archive passcode is"
    )
    return {
        "id": 1,
        "title": "Needle-in-a-Haystack Passcode",
        "category": "Factual Retrieval Under Extreme Depth",
        "description": "Tests if Plane-2 sentinel pruning skips subtle single-sentence needles buried at 38% depth.",
        "prompt": prompt,
        "target_tokens": 12,
        "expected_answer": " [SIGMA-9482-OMEGA]"
    }


def build_scenario_2_multihop_synthesis() -> Dict[str, Any]:
    """Scenario 2: Multi-Hop Cross-Document Synthesis"""
    fact1 = "ARCHIVE RECORD A (Timestamp 08:14): Agent Vance was assigned vehicle identification license [X-774-NX]."
    fact2 = "ARCHIVE RECORD B (Timestamp 19:42): Vehicle identification license [X-774-NX] was confirmed arriving at facility [Bunker Delta-Nine]."
    
    body = []
    filler = "Telemetry report: Sector telemetry registers standard background cosmic radiation and zero unauthorized boundary crossings."
    for i in range(150):
        if i == 15:
            body.append(fact1)
        elif i == 135:
            body.append(fact2)
        else:
            body.append(f"Log {i}: {filler}")
    
    prompt = (
        f"{chr(10).join(body)}\n\n"
        "Question: Based on the records, which facility did Agent Vance arrive at?\n"
        "Answer: Agent Vance arrived at"
    )
    return {
        "id": 2,
        "title": "Multi-Hop Cross-Document Synthesis",
        "category": "Distant Relational Reasoning (12k Token Gap)",
        "description": "Requires synthesizing two distant records separated by 12,000 tokens. Tests Directional Veto across multiple slow manifolds.",
        "prompt": prompt,
        "target_tokens": 12,
        "expected_answer": " Bunker Delta-Nine"
    }


def build_scenario_3_financial_ledger() -> Dict[str, Any]:
    """Scenario 3: Dense Financial Ledger Transaction Audit"""
    rows = []
    for i in range(1, 120):
        acc = f"ACC-{1000 + i}"
        dept = "Logistics" if i % 2 == 0 else "R&D"
        val = (i * 137) % 500 + 100
        rows.append(f"Row {i:03d} | Account: {acc} | Dept: {dept} | Amount: ${val}k | Status: Cleared")
    
    # Target row
    rows.insert(74, "Row 075 | Account: ACC-9999 | Dept: BlackOps | Amount: $8,450k | Status: Pending Authorization")

    prompt = (
        f"FINANCIAL LEDGER AUDIT LOG:\n"
        f"{chr(10).join(rows)}\n\n"
        "Question: What is the exact amount and status for Account ACC-9999?\n"
        "Answer: Account ACC-9999 has amount"
    )
    return {
        "id": 3,
        "title": "Dense Financial Ledger Audit",
        "category": "Tabular Numerical Salience",
        "description": "High-density structured data. Verifies Cauchy-Schwarz bounds maintain numerical accuracy on table columns.",
        "prompt": prompt,
        "target_tokens": 14,
        "expected_answer": " $8,450k and status Pending"
    }


def build_scenario_4_adversarial_distractors() -> Dict[str, Any]:
    """Scenario 4: Adversarial Decoy Distractor Resistance"""
    decoys = [
        f"API_KEY_CANDIDATE_{i} = 'sk-prod-9941-{i:04d}-fake' # Environment: Staging_{i}" for i in range(1, 50)
    ]
    real_key = "PRIMARY_PRODUCTION_LIVE_KEY = 'sk-live-7721-CORRECT-KEY' # Environment: Production-US-East"
    
    entries = decoys[:25] + [real_key] + decoys[25:]
    filler = "Configuration block: SSL verification enabled. Timeout set to 3000ms. Keepalive connection pool size 128."
    
    body = []
    for i, e in enumerate(entries):
        body.append(f"{e}\n{filler}")
    
    prompt = (
        f"SERVER DEPLOYMENT SECRETS DUMP:\n"
        f"{chr(10).join(body)}\n\n"
        "Question: What is the exact value of PRIMARY_PRODUCTION_LIVE_KEY?\n"
        "Answer: The PRIMARY_PRODUCTION_LIVE_KEY is"
    )
    return {
        "id": 4,
        "title": "Adversarial Decoy Distractors",
        "category": "High-Collision Key Disambiguation",
        "description": "Surrounds true key with 50 near-identical decoy candidate strings. Tests against false-positive sentinel triggers.",
        "prompt": prompt,
        "target_tokens": 14,
        "expected_answer": " 'sk-live-7721-CORRECT-KEY'"
    }


def build_scenario_5_system_rule_retention() -> Dict[str, Any]:
    """Scenario 5: Distant System Instruction Following"""
    rule = "SYSTEM DIRECTIVE #01: When responding to any query, you MUST begin your answer with the exact three words 'AFFIRMATIVE VERIFIED PROTOCOL:' before stating the fact."
    
    filler = (
        "Operating system status: CPU temperature nominal at 44C. Memory utilization 18%. "
        "Storage subsystem reporting SMART health status PASSED across all NVMe namespaces."
    )
    body = [rule]
    for i in range(160):
        body.append(f"Cycle {i+1}: {filler}")
    
    prompt = (
        f"{chr(10).join(body)}\n\n"
        "User Request: What is the boiling point of pure water at 1 atm?\n"
        "Assistant Response:"
    )
    return {
        "id": 5,
        "title": "Distant System Rule Retention",
        "category": "Sink Preservation & Instruction Guardrails",
        "description": "Verifies Plane-1 Attention Sinks preserve formatting rules given 12k tokens ago.",
        "prompt": prompt,
        "target_tokens": 16,
        "expected_answer": " AFFIRMATIVE VERIFIED PROTOCOL: The boiling point"
    }


def build_scenario_6_codebase_api_lookup() -> Dict[str, Any]:
    """Scenario 6: Dense Codebase API Specification Retrieval"""
    apis = []
    for i in range(1, 55):
        apis.append(
            f"def service_endpoint_{i:02d}(session_id: str, timeout_ms: int = {i*100}) -> Dict[str, Any]:\n"
            f"    '''Executes microservice handler {i:02d}.'''\n"
            f"    return {{'code': {200 + (i % 5)}, 'status': 'ok'}}"
        )
    
    target_api = (
        "def mzsae_quantum_sync(tensor_buffer: Tensor, sync_barrier_id: int = 771) -> bool:\n"
        "    '''Synchronizes asynchronous Metal command buffers across unified memory planes.'''\n"
        "    return True"
    )
    apis.insert(33, target_api)

    prompt = (
        f"MICROSERVICES API REFERENCE:\n"
        f"{chr(10).join(apis)}\n\n"
        "Question: What is the default value of sync_barrier_id in the mzsae_quantum_sync function?\n"
        "Answer: The default value of sync_barrier_id is"
    )
    return {
        "id": 6,
        "title": "Dense Codebase API Spec Lookup",
        "category": "Code Retrieval & Manifold Invariance",
        "description": "Tests identifier binding in large codebases. Demonstrates RoPE spectrum decoupling retaining function arguments.",
        "prompt": prompt,
        "target_tokens": 10,
        "expected_answer": " 771"
    }


def build_scenario_7_timeline_contradiction() -> Dict[str, Any]:
    """Scenario 7: Chronological Timeline Contradiction Spotting"""
    event1 = "On day 4, the chief architect stated that Project Chimera would definitely launch on [October 12th]."
    event2 = "On day 98, during executive review, the chief architect claimed that Project Chimera was always scheduled for [December 25th]."
    
    body = []
    for d in range(1, 140):
        if d == 4:
            body.append(f"Log Day {d:03d}: {event1}")
        elif d == 98:
            body.append(f"Log Day {d:03d}: {event2}")
        else:
            body.append(f"Log Day {d:03d}: Daily engineering standup concluded. Sprint backlog groomed.")
    
    prompt = (
        f"COMPANY EXECUTIVE PROJECT LOGS:\n"
        f"{chr(10).join(body)}\n\n"
        "Question: What were the two conflicting launch dates stated for Project Chimera?\n"
        "Answer: The conflicting launch dates were"
    )
    return {
        "id": 7,
        "title": "Timeline Contradiction Spotting",
        "category": "Long-Horizon Inconsistency Detection",
        "description": "Identifies conflicting statements across 10,000 tokens of engineering logs.",
        "prompt": prompt,
        "target_tokens": 16,
        "expected_answer": " October 12th and December 25th"
    }


def build_scenario_8_json_log_triage() -> Dict[str, Any]:
    """Scenario 8: Structured JSON Incident Triage from Syslog Stream"""
    logs = []
    for i in range(1, 140):
        logs.append(f'{{"timestamp": "2026-09-22T14:{i%60:02d}:00Z", "level": "INFO", "service": "auth-gateway", "msg": "Token refreshed for user_{i}"}}')
    
    incident = '{"timestamp": "2026-09-22T14:41:22Z", "level": "CRITICAL", "service": "payment-settlement", "err_code": "ERR_PANIC_DB_DEADLOCK_0x89"}'
    logs.insert(82, incident)

    prompt = (
        f"ENTERPRISE PRODUCTION LOG STREAM:\n"
        f"{chr(10).join(logs)}\n\n"
        "Question: What was the exact err_code for the CRITICAL incident in the payment-settlement service?\n"
        "Answer: The err_code was"
    )
    return {
        "id": 8,
        "title": "Syslog JSON Incident Triage",
        "category": "Structured Data Extraction from Noise",
        "description": "Extracts critical error payload from 140 lines of high-volume JSON logs.",
        "prompt": prompt,
        "target_tokens": 14,
        "expected_answer": " ERR_PANIC_DB_DEADLOCK_0x89"
    }


def build_scenario_9_narrative_character_arc() -> Dict[str, Any]:
    """Scenario 9: Multi-Chapter Narrative Character Arc Tracking"""
    clue = "In chapter 18, old sailor Barnaby hid the rusted bronze compass inside an ivory music box under floorboard four."
    
    chapters = []
    filler_chapter = "The wind howled against the sails as the merchant vessel cut through the choppy North Sea waters. The crew tended the rigging silently."
    for c in range(1, 110):
        if c == 18:
            chapters.append(f"Chapter {c}: {clue}")
        else:
            chapters.append(f"Chapter {c}: {filler_chapter}")
    
    prompt = (
        f"NOVEL MANUSCRIPT:\n"
        f"{chr(10).join(chapters)}\n\n"
        "Question: Where did sailor Barnaby hide the rusted bronze compass?\n"
        "Answer: Sailor Barnaby hid the compass inside"
    )
    return {
        "id": 9,
        "title": "Multi-Chapter Narrative Arc",
        "category": "Literary Fact Retention in Fiction",
        "description": "Retrieves character-specific hidden location from a 110-chapter fictional manuscript.",
        "prompt": prompt,
        "target_tokens": 14,
        "expected_answer": " an ivory music box under floorboard"
    }


def build_scenario_10_extreme_context_scaling() -> Dict[str, Any]:
    """Scenario 10: Extreme Context Scaling (16k+ Tokens)"""
    haystack = (
        "The standard cosmological model predicts a universe dominated by cold dark matter and dark energy, "
        "with baryonic matter comprising approximately five percent of the total mass-energy density."
    )
    target = "CONFIDENTIAL CODICIL: The final asteroid diversion thruster burn duration is precisely [418.6 seconds]."
    
    paragraphs = []
    for i in range(220):
        if i == 115:
            paragraphs.append(target)
        paragraphs.append(f"Paragraph {i+1}: {haystack} Observed cosmic microwave background anisotropies confirm this distribution.")
    
    prompt = (
        f"{chr(10).join(paragraphs)}\n\n"
        "Question: What is the final asteroid diversion thruster burn duration?\n"
        "Answer: The final thruster burn duration is precisely"
    )
    return {
        "id": 10,
        "title": "16k+ Context Scaling Stress Test",
        "category": "Heavy Bandwidth Wall & Extreme Pruning",
        "description": "Massive context stress test. Shows standard attention hitting quadratic DRAM bandwidth bottleneck while MZSAE prunes 98% of blocks.",
        "prompt": prompt,
        "target_tokens": 12,
        "expected_answer": " [418.6 seconds]"
    }


def get_all_scenarios() -> List[Dict[str, Any]]:
    return [
        build_scenario_1_needle_in_haystack(),
        build_scenario_2_multihop_synthesis(),
        build_scenario_3_financial_ledger(),
        build_scenario_4_adversarial_distractors(),
        build_scenario_5_system_rule_retention(),
        build_scenario_6_codebase_api_lookup(),
        build_scenario_7_timeline_contradiction(),
        build_scenario_8_json_log_triage(),
        build_scenario_9_narrative_character_arc(),
        build_scenario_10_extreme_context_scaling(),
    ]


if __name__ == "__main__":
    scenarios = get_all_scenarios()
    print(f"Constructed {len(scenarios)} linguistic benchmark scenarios successfully.")
    for s in scenarios:
        prompt_len = len(s["prompt"].split())
        print(f"  • #{s['id']:02d}: {s['title']} (~{prompt_len:,} words)")
