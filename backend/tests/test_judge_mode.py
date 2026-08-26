import pytest
from app.judge.schemas import JudgeScenario, DifficultyProfile, ScenarioStatePhase
from app.judge.scenario_runner import ScenarioOrchestrator
from pydantic import ValidationError

def test_1_scenario_schema():
    # Valid scenario
    scen = JudgeScenario(
        scenario_id="test_1",
        seed=42,
        attack_family="micro_structuring",
        attack_scale=100,
        difficulty=DifficultyProfile.HARD,
        adaptive_red_team_enabled=True,
        zero_day_radar_enabled=False,
        defense_compiler_enabled=True,
        human_approval_required=True,
        evolution_generations=3
    )
    assert scen.scenario_id == "test_1"

def test_3_invalid_parameter_rejection():
    # Scale too large
    with pytest.raises(ValidationError):
        JudgeScenario(
            scenario_id="test_bound",
            seed=42,
            attack_family="micro_structuring",
            attack_scale=600, # Max is 500
            difficulty=DifficultyProfile.HARD,
            adaptive_red_team_enabled=True,
            zero_day_radar_enabled=False,
            defense_compiler_enabled=True,
            human_approval_required=True,
            evolution_generations=3
        )
    # Generations too large
    with pytest.raises(ValidationError):
        JudgeScenario(
            scenario_id="test_bound_2",
            seed=42,
            attack_family="micro_structuring",
            attack_scale=100,
            difficulty=DifficultyProfile.HARD,
            adaptive_red_team_enabled=True,
            zero_day_radar_enabled=False,
            defense_compiler_enabled=True,
            human_approval_required=True,
            evolution_generations=10 # Max is 5
        )

def test_4_scenario_state_transitions_and_rest():
    # Creates scenario
    scen = JudgeScenario(
        scenario_id="test_transitions",
        seed=42,
        attack_family="micro_structuring",
        attack_scale=50,
        difficulty=DifficultyProfile.EASY,
        adaptive_red_team_enabled=False,
        zero_day_radar_enabled=False,
        defense_compiler_enabled=False,
        human_approval_required=False,
        evolution_generations=0
    )
    state = ScenarioOrchestrator.create_scenario(scen)
    assert state.current_phase == ScenarioStatePhase.PREPARE
    assert not state.is_running
    assert not state.is_completed

    # Run scenario
    ScenarioOrchestrator.run_scenario("test_transitions")
    state = ScenarioOrchestrator.get_state("test_transitions")
    assert state.is_completed
    assert state.current_phase == ScenarioStatePhase.SCORE
    assert state.scorecard is not None
    assert state.scorecard.attack_family == "micro_structuring"
    
    # 13 Scorecard correctness
    assert state.scorecard.initial_evasion >= 0.0
    assert state.scorecard.defense_readiness_score >= 0.0
    
    # 14 Reset behavior
    ScenarioOrchestrator.reset("test_transitions")
    assert ScenarioOrchestrator.get_state("test_transitions") is None

def test_9_approval_state():
    from unittest.mock import patch
    from app.blue_team.defense_compiler import DefensePolicy
    scen = JudgeScenario(
        scenario_id="test_approval",
        seed=42,
        attack_family="micro_structuring",
        attack_scale=50,
        difficulty=DifficultyProfile.HARD,
        adaptive_red_team_enabled=False,
        zero_day_radar_enabled=False,
        defense_compiler_enabled=True,
        human_approval_required=True,
        evolution_generations=0
    )
    state = ScenarioOrchestrator.create_scenario(scen)
    
    with patch('app.judge.scenario_runner.compile_policy') as mock_compile:
        mock_compile.return_value = [DefensePolicy(
            policy_id="mock_pol", version=1, source_attack_id="a", source_attack_family="b",
            root_cause="c", policy_type="d", conditions={}, action="BLOCK", severity="HIGH",
            confidence=0.8, status="CANDIDATE", provenance="mock"
        )]
        ScenarioOrchestrator.run_scenario("test_approval")
    state = ScenarioOrchestrator.get_state("test_approval")
    
    # Should halt at APPROVE
    assert state.current_phase == ScenarioStatePhase.APPROVE
    assert not state.is_completed
    
    # Trigger approve
    ScenarioOrchestrator.approve_and_continue("test_approval")
    assert state.current_phase == ScenarioStatePhase.SCORE
    assert state.is_completed
    assert state.policy_status == "ACTIVE"

def test_5_to_8_execution_components():
    # Test adaptive execution, radar, etc
    scen = JudgeScenario(
        scenario_id="test_extreme",
        seed=123,
        attack_family="micro_structuring",
        attack_scale=50,
        difficulty=DifficultyProfile.EXTREME,
        adaptive_red_team_enabled=True,
        zero_day_radar_enabled=True,
        defense_compiler_enabled=True,
        human_approval_required=False,
        evolution_generations=1
    )
    ScenarioOrchestrator.create_scenario(scen)
    ScenarioOrchestrator.run_scenario("test_extreme")
    state = ScenarioOrchestrator.get_state("test_extreme")
    assert state.is_completed
    assert state.scorecard.best_evolved_evasion >= state.scorecard.initial_evasion
    assert state.scorecard.cluster_count >= 0
