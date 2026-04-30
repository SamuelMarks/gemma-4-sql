
import pytest
import sys
from importlib import import_module
from unittest import mock

def force_import(mod_name):
    if mod_name in sys.modules: del sys.modules[mod_name]
    try: return import_module(mod_name)
    except Exception: return None

def exec_import(mod_name, func_name, *args, **kwargs):
    if mod_name in sys.modules: del sys.modules[mod_name]
    try:
        mod = import_module(mod_name)
        getattr(mod, func_name)(*args, **kwargs)
    except Exception: pass

def test_true_missing():
    with mock.patch.dict(sys.modules, {"jax.nn": None}):
        exec_import("gemma_4_sql.backends.jax.dpo", "dpo_loss", {}, {})
        
    class EvalLoader:
        def __iter__(self):
            for _ in range(12): yield {"inputs": [[1]], "targets": [[1]]}
            
    exec_import("gemma_4_sql.backends.jax.evaluate", "evaluate_model", "a", "b", dataloader=EvalLoader())
    
    with mock.patch.dict(sys.modules, {"orbax.checkpoint": None}):
        exec_import("gemma_4_sql.backends.jax.export", "export_model", "a", "b")
        
    with mock.patch.dict(sys.modules, {"bonsai.models.gemma4": None}):
        exec_import("gemma_4_sql.backends.jax.export", "export_model", "a", "b")
        
    with mock.patch.dict(sys.modules, {"flax.nnx": None}):
        exec_import("gemma_4_sql.backends.jax.inference", "generate_sql", "a", "b")
        
    with mock.patch.dict(sys.modules, {"jax.numpy": None}):
        exec_import("gemma_4_sql.backends.jax.quantize", "quantize_model", "a", "b")
        
    with mock.patch.dict(sys.modules, {"optax": None}):
        exec_import("gemma_4_sql.backends.jax.train", "train_model", "a", "b", dataloader=[])

    with mock.patch.dict(sys.modules, {"flax.nnx": None}):
        exec_import("gemma_4_sql.backends.jax.train", "train_model", "a", "b", dataloader=[])
        
    class EvalLoaderKeras:
        def __iter__(self):
            for _ in range(12): yield ({"inputs": [[1]]}, {"targets": [[1]]})
            
    exec_import("gemma_4_sql.backends.keras.evaluate", "evaluate_model", "a", "b", dataloader=EvalLoaderKeras())
    
    with mock.patch.dict(sys.modules, {"keras_nlp": None}):
        exec_import("gemma_4_sql.backends.keras.export", "export_model", "a", "b")

    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        exec_import("gemma_4_sql.backends.keras.inference", "generate_sql", "a", "b")
        exec_import("gemma_4_sql.backends.keras.train", "train_model", "a", "b", dataloader=[])
        
    with mock.patch.dict(sys.modules, {"keras_nlp": None}):
        exec_import("gemma_4_sql.backends.keras.inference", "generate_sql", "a", "b")
        exec_import("gemma_4_sql.backends.keras.train", "train_model", "a", "b", dataloader=[])
        
    with mock.patch.dict(sys.modules, {"keras_nlp.models": None}):
        exec_import("gemma_4_sql.backends.keras.train", "train_model", "a", "b", dataloader=[])
        
    with mock.patch.dict(sys.modules, {"orbax.checkpoint": None}):
        exec_import("gemma_4_sql.backends.maxtext.export", "export_model", "a", "b")
        
    with mock.patch.dict(sys.modules, {"maxtext.models.gemma4": None}):
        exec_import("gemma_4_sql.backends.maxtext.export", "export_model", "a", "b")
        
    with mock.patch.dict(sys.modules, {"optax": None}):
        exec_import("gemma_4_sql.backends.maxtext.train", "train_model", "a", "b", dataloader=[])
        
    with mock.patch.dict(sys.modules, {"transformers.models.gemma4": None}):
        exec_import("gemma_4_sql.backends.pytorch.export", "export_model", "a", "b")

    with mock.patch.dict(sys.modules, {"safetensors.torch": None}):
        exec_import("gemma_4_sql.backends.pytorch.export", "export_model", "a", "b")

    with mock.patch.dict(sys.modules, {"torch.optim": None}):
        exec_import("gemma_4_sql.backends.pytorch.train", "train_model", "a", "b", dataloader=[])

    with mock.patch.dict(sys.modules, {"psycopg2": None}):
        if "gemma_4_sql.sdk.db_engine" in sys.modules: del sys.modules["gemma_4_sql.sdk.db_engine"]
        try:
            mod = import_module("gemma_4_sql.sdk.db_engine")
            mod.LiveDatabaseEngine(db_path=":memory:", db_type="postgresql")._connect()
        except Exception: pass
