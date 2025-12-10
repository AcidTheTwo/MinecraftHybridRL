import importlib
import sys

def load_objective_module(model_name):
    """
    Returns a dictionary containing:
    - 'class': The Objective class
    - 'actions': The list of action strings
    - 'config': The input configuration dict
    """
    module_path = f"src.server.objectives.{model_name}"
    
    try:
        module = importlib.import_module(module_path)
        print(f"fv/ Loaded Cartridge: {model_name}")
        
        # Validation: Check if required attributes exist
        if not hasattr(module, 'Objective'):
            raise AttributeError(f"Module {model_name} missing 'class Objective'")
        if not hasattr(module, 'ACTIONS'):
            raise AttributeError(f"Module {model_name} missing 'list ACTIONS'")
        
        return {
            "class": getattr(module, 'Objective'),
            "actions": getattr(module, 'ACTIONS'),
            "input": getattr(module, 'INPUTS', []) # Config is optional, defaults empty
        }

    except (ImportError, AttributeError) as e:
        print(f"❌ Error loading model '{model_name}': {e}")
        sys.exit(1)