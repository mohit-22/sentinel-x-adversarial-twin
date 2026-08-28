import copy
import random
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from app.red_team.immune_memory import MemoryRecord
from app.red_team.attack_injector import ATTACK_GENERATORS
from app.blue_team.features import engineer_features, combine_clean_and_injected
from app.blue_team.graph_engine import apply_graph_features
from app.blue_team.detector import FEATURE_COLUMNS
from app.blue_team.zero_day import compute_novelty_score

def mutate_genome(genome: Dict, mutation_prob: float = 0.5, generation: int = 0, seed: int = 42, parent_id: Optional[str] = None) -> Dict:
    """Mutate a genome's parameters deterministically based on seed/random state.

    genome_id is derived from the already-seeded `random` module state
    (seeded once via random.seed(seed) in run_evolutionary_search), not
    uuid.uuid4() (which is os.urandom-backed and not reproducible -- CLAUDE.md
    rule 6, fixed seeds everywhere). A literal f"genome_{family}_{generation}_
    {seed}" alone would collide whenever more than one genome is bred within
    the same generation (this loop does that), so a random.randint draw from
    the same seeded stream disambiguates while staying fully reproducible:
    same seed -> same full call sequence -> same ids, every run.
    """
    mutated = copy.deepcopy(genome)
    # Ensure it gets a new genome_id if it's a mutation
    if "genome_id" in mutated:
        unique_component = random.randint(0, 999_999)
        mutated["genome_id"] = f"MUT-genome_{genome['family']}_{generation}_{seed}_{unique_component}"
        
    if parent_id is not None:
        mutated["_parent_id"] = parent_id
        
    if "parameters" not in mutated:
        return mutated
        
    for k, v in mutated["parameters"].items():
        if random.random() < mutation_prob:
            if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                # Range mutation
                shift_low = v[0] * random.uniform(-0.2, 0.2)
                shift_high = v[1] * random.uniform(-0.2, 0.2)
                
                # Ensure we don't drop below 0 if it was positive
                new_low = max(0.01 if isinstance(v[0], float) else 1, v[0] + shift_low)
                new_high = max(new_low + (0.01 if isinstance(v[1], float) else 1), v[1] + shift_high)
                
                if isinstance(v[0], int):
                    mutated["parameters"][k] = [int(new_low), int(new_high)]
                else:
                    mutated["parameters"][k] = [new_low, new_high]
            
            elif isinstance(v, bool):
                # Flip bool
                mutated["parameters"][k] = not v
                
            elif isinstance(v, (int, float)):
                # Scalar mutation
                factor = random.uniform(0.5, 2.0)
                new_val = v * factor
                
                # Special cases for bounded values
                if "ratio" in k or "score" in k:
                    new_val = min(max(new_val, 0.0), 1.0)
                    
                if isinstance(v, int):
                    new_val = int(max(1, new_val))
                    
                mutated["parameters"][k] = new_val
                
    return mutated

def calculate_fitness(
    evasion_rate: float,
    novelty_score: float,
    target_amount: float,
    max_amount: float = 100000.0,
    validity_penalty: float = 0.0
) -> float:
    """
    Weighted fitness calculation.
    """
    evasion_comp = 0.5 * evasion_rate
    novelty_comp = 0.2 * novelty_score
    impact_comp = 0.2 * min(target_amount / max_amount, 1.0)
    realism_comp = 0.1 * 1.0 # High if generated successfully
    
    return max(0.0, evasion_comp + novelty_comp + impact_comp + realism_comp - validity_penalty)

def run_evolutionary_search(
    base_genome: Dict,
    model,
    radar_state: Dict,
    customers: pd.DataFrame,
    clean_history: pd.DataFrame,
    merchants: pd.DataFrame,
    graph_features: Dict,
    feature_columns: List[str] = FEATURE_COLUMNS,
    population_size: int = 5,
    generations: int = 3,
    elite_count: int = 1,
    mutation_probability: float = 0.5,
    n_instances: int = 100,
    seed: int = 42
) -> Dict:
    """
    Runs an evolutionary search loop to discover high-fitness mutated attacks.
    """
    random.seed(seed)
    
    population = [base_genome]
    base_genome_id = base_genome.get("genome_id")
    for _ in range(population_size - 1):
        population.append(mutate_genome(base_genome, mutation_probability, generation=0, seed=seed, parent_id=base_genome_id))
        
    lineage = []
    
    for gen in range(generations):
        gen_results = []
        for i, genome in enumerate(population):
            family = genome["family"]
            gen_fn = ATTACK_GENERATORS[family]["attacks_fn"]
            
            try:
                # Generate transactions
                attacks_df = gen_fn(genome, customers, merchants, n_instances, seed=seed + gen*100 + i)
                
                # Feature engineering
                combined = combine_clean_and_injected(clean_history, attacks_df)
                combined = combined.drop_duplicates(subset="transaction_id", keep="last")
                featured = engineer_features(combined, customers)
                featured = apply_graph_features(featured, graph_features)
                
                fraud_rows = featured[featured["is_fraud"] == 1]

                # Evaluate Evasion. fraud_rows is all is_fraud==1 by
                # construction, so evaluate_detector's confusion-matrix
                # approach is the wrong tool here (no negative samples
                # exist -- confusion_matrix().ravel() can return fewer than
                # 4 values whenever predictions are also all one class,
                # e.g. a genome that gets fully caught, which is a real and
                # desirable outcome, not an error). Compute evasion
                # directly, same formula and same model.predict() call as
                # arena.py's own run_attack.
                import inspect
                if len(fraud_rows) > 0:
                    sig = inspect.signature(model.predict)
                    if 'context' in sig.parameters:
                        y_pred = model.predict(fraud_rows[feature_columns], context={'eval_df': fraud_rows, 'featured_df': featured})
                    else:
                        y_pred = model.predict(fraud_rows[feature_columns])
                    evasion_rate = float(1.0 - y_pred.mean())
                else:
                    evasion_rate = 0.0
                
                # Evaluate Novelty
                novelty_scores = compute_novelty_score(radar_state, fraud_rows)
                avg_novelty = float(novelty_scores.mean()) if len(novelty_scores) > 0 else 0.0
                
                # Evaluate Impact
                avg_amount = float(fraud_rows["amount"].sum() / max(1, n_instances))
                
                fitness = calculate_fitness(
                    evasion_rate=evasion_rate,
                    novelty_score=avg_novelty,
                    target_amount=avg_amount
                )
                
                result = {
                    "generation": gen,
                    "genome": genome,
                    "evasion_rate": evasion_rate,
                    "novelty_score": avg_novelty,
                    "impact_score": avg_amount,
                    "realism_score": 1.0, # generated successfully
                    "total_fitness": fitness,
                    "validity_status": "VALID",
                    "parent_attack_id": genome.get("_parent_id", base_genome_id) if gen == 0 else genome.get("_parent_id")
                }
                
            except Exception as e:
                # Penalize invalid mutations
                fitness = calculate_fitness(0, 0, 0, validity_penalty=1.0)
                result = {
                    "generation": gen,
                    "genome": genome,
                    "evasion_rate": 0.0,
                    "novelty_score": 0.0,
                    "impact_score": 0.0,
                    "realism_score": 0.0,
                    "total_fitness": fitness,
                    "validity_status": f"INVALID: {str(e)}",
                    "parent_attack_id": genome.get("_parent_id", base_genome_id) if gen == 0 else genome.get("_parent_id")
                }
                
            gen_results.append(result)
            lineage.append(result)
            
        # Select top elite
        gen_results.sort(key=lambda x: x["total_fitness"], reverse=True)
        elites = gen_results[:elite_count]
        
        # Mark elite status directly on the dictionaries
        for i, res in enumerate(gen_results):
            res["is_elite"] = (i < elite_count)
            res["is_best"] = False  # Placeholder to be updated later
        
        # Breed next generation
        if gen < generations - 1:
            population = [e["genome"] for e in elites]
            while len(population) < population_size:
                parent = random.choice(elites)["genome"]
                population.append(mutate_genome(parent, mutation_probability, generation=gen + 1, seed=seed, parent_id=parent["genome_id"]))
                
    # Evaluate best overall and mark it
    best_result = sorted(lineage, key=lambda x: x["total_fitness"], reverse=True)[0]
    for res in lineage:
        if res["genome"]["genome_id"] == best_result["genome"]["genome_id"]:
            res["is_best"] = True
            
    return {
        "best_attack": best_result,
        "lineage": lineage
    }
