"""Orchestration for KV-budget grid runs.

Coordinates running inference across multiple KV budgets.
"""

from .. import paths
from ..db import connect, schema
from ..engines import endpoints
from . import runner


def orchestrate(settings: dict) -> None:
    """Orchestrate KV-budget grid run.
    
    Creates one run record per budget and invokes the runner for each setting.
    
    Args:
        settings: Settings dict (from load_settings).
    
    Raises:
        ValueError: If required settings are missing.
    """
    # Resolve paths
    exp_group_id = settings["output"]["exp_group_id"]
    db_path_cfg = settings.get("db", {}).get("path")
    db_path = paths.db_path(exp_group_id, db_path_cfg)
    
    # Open DB connection and initialize schema
    conn = connect.connect(db_path)
    schema.init_schema(conn)
    
    # Resolve engine and model
    engine, model_name = endpoints.build_engine(settings)
    
    # Get KV budgets
    kv_budgets = settings.get("kv", {}).get("budgets", [])
    if not kv_budgets:
        raise ValueError("settings.kv.budgets must be non-empty")
    
    # Get KV policy
    kv_policy = settings.get("kv", {}).get("policy", "unknown")
    
    # Run for each KV budget
    for kv_budget in kv_budgets:
        print(f"Starting run for KV budget: {kv_budget}")
        
        # Generate run_id
        run_id = f"{exp_group_id}_budget_{kv_budget}"
        
        # Create run record (placeholder - will be implemented in dao.py)
        # TODO: Call dao.create_run(conn, run_id, exp_group_id, kv_policy, kv_budget, engine_name, base_url, model_name)
        # This will be implemented when dao.py is extended with run creation functions
        # For now, runner.run_one_setting may handle run creation
        
        # Run inference for this budget
        runner.run_one_setting(
            conn=conn,
            settings=settings,
            engine=engine,
            model_name=model_name,
            run_id=run_id,
            kv_budget=kv_budget
        )
        
        print(f"Completed run for KV budget: {kv_budget}")
    
    conn.close()
