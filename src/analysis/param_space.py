"""
Parameter space management for sweeps
"""

import numpy as np
from typing import Dict, List, Iterator, Union, Any
from itertools import product
import yaml


class ParamRange:
    """represents a single parameter's range/values"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.type = config['type']
        self._values = None
        self._validate_config()
        
    def _validate_config(self):
        """validate parameter configuration"""
        valid_types = {'linspace', 'logspace', 'grid', 'random', 'fixed', 'values'}
        if self.type not in valid_types:
            raise ValueError(f"invalid param type '{self.type}', must be one of {valid_types}")
            
        # type-specific validation
        if self.type in ['linspace', 'logspace']:
            required = {'start', 'stop', 'num'}
            missing = required - set(self.config.keys())
            if missing:
                raise ValueError(f"param '{self.name}' missing required fields: {missing}")
                
        elif self.type == 'random':
            required = {'start', 'stop', 'num'}
            missing = required - set(self.config.keys())
            if missing:
                raise ValueError(f"param '{self.name}' missing required fields: {missing}")
                
        elif self.type == 'fixed':
            if 'value' not in self.config:
                raise ValueError(f"param '{self.name}' missing 'value' field")
                
        elif self.type == 'values':
            if 'values' not in self.config:
                raise ValueError(f"param '{self.name}' missing 'values' field")
            if not isinstance(self.config['values'], (list, tuple)):
                raise ValueError(f"param '{self.name}' 'values' must be a list/tuple")
                
    def get_values(self) -> np.ndarray:
        """get array of parameter values"""
        if self._values is not None:
            return self._values
            
        if self.type == 'linspace':
            self._values = np.linspace(
                self.config['start'], 
                self.config['stop'], 
                self.config['num']
            )
        elif self.type == 'logspace':
            self._values = np.logspace(
                np.log10(self.config['start']),
                np.log10(self.config['stop']),
                self.config['num']
            )
        elif self.type == 'random':
            # reproducible random sampling
            rng = np.random.RandomState(self.config.get('seed', 42))
            self._values = rng.uniform(
                self.config['start'],
                self.config['stop'], 
                self.config['num']
            )
        elif self.type == 'fixed':
            self._values = np.array([self.config['value']])
        elif self.type == 'values':
            self._values = np.array(self.config['values'])
            
        return self._values
        
    def size(self) -> int:
        """number of values in this parameter range"""
        return len(self.get_values())
        
    def __repr__(self) -> str:
        return f"ParamRange({self.name}, {self.type}, size={self.size()})"


class ParamSpace:
    """manages parameter space sampling and iteration"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        initialize parameter space from config
        
        config should contain 'parameters' key with param definitions
        """
        self.config = config
        self.param_ranges: Dict[str, ParamRange] = {}
        
        if 'parameters' not in config:
            raise ValueError("config missing 'parameters' section")
            
        # build parameter ranges
        for param_name, param_config in config['parameters'].items():
            self.param_ranges[param_name] = ParamRange(param_name, param_config)
            
        # precompute some properties
        self._param_names = list(self.param_ranges.keys())
        self._total_size = None
        
    def iter_params(self) -> Iterator[Dict[str, float]]:
        """yield parameter combinations as dicts"""
        # get all parameter value arrays
        param_arrays = [self.param_ranges[name].get_values() for name in self._param_names]
        
        # use itertools.product for cartesian product
        for param_values in product(*param_arrays):
            yield dict(zip(self._param_names, param_values))
            
    def size(self) -> int:
        """total number of parameter combinations"""
        if self._total_size is None:
            self._total_size = 1
            for param_range in self.param_ranges.values():
                self._total_size *= param_range.size()
        return self._total_size
        
    def get_param_info(self) -> Dict[str, Dict]:
        """get info about each parameter"""
        info = {}
        for name, param_range in self.param_ranges.items():
            values = param_range.get_values()
            info[name] = {
                'type': param_range.type,
                'size': param_range.size(),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values.tolist() if param_range.size() <= 10 else f"[{values[0]:.3f}, ..., {values[-1]:.3f}]"
            }
        return info
        
    def validate(self) -> bool:
        """validate parameter space configuration"""
        try:
            # check that we can generate at least one param combination
            first_params = next(self.iter_params())
            
            # check that all parameters have reasonable ranges
            for name, param_range in self.param_ranges.items():
                values = param_range.get_values()
                if len(values) == 0:
                    raise ValueError(f"parameter '{name}' has no values")
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"parameter '{name}' contains non-finite values")
                    
            return True
        except Exception as e:
            print(f"parameter space validation failed: {e}")
            return False
            
    def __repr__(self) -> str:
        param_info = ", ".join([f"{name}({pr.size()})" for name, pr in self.param_ranges.items()])
        return f"ParamSpace(size={self.size():,}, params=[{param_info}])"
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ParamSpace':
        """load parameter space from yaml file"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)
        
    def to_yaml(self, yaml_path: str):
        """save parameter space config to yaml file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


def test_param_space():
    """basic tests for ParamSpace functionality"""
    
    # test config
    config = {
        'parameters': {
            'a': {
                'type': 'linspace',
                'start': 1.0,
                'stop': 2.0,
                'num': 5
            },
            'b': {
                'type': 'fixed', 
                'value': 0.3
            },
            'noise_std': {
                'type': 'logspace',
                'start': 0.01,
                'stop': 0.1,
                'num': 3
            }
        }
    }
    
    # create param space
    param_space = ParamSpace(config)
    print(f"param space: {param_space}")
    
    # validate
    assert param_space.validate()
    assert param_space.size() == 5 * 1 * 3  # 15 combinations
    
    # test iteration
    params_list = list(param_space.iter_params())
    assert len(params_list) == 15
    
    # check first few combinations
    first_params = params_list[0]
    assert 'a' in first_params
    assert 'b' in first_params  
    assert 'noise_std' in first_params
    assert first_params['b'] == 0.3  # fixed value
    
    print("param space tests passed")
    return param_space


if __name__ == "__main__":
    test_param_space() 