"""
we look at the following:
- confidence intervals: statistical bounds (95% confidence, tight)
- deterministic intervals: guaranteed bounds (100% certain, wide) 
- regime classification: actionable decisions from tightest valid bounds
- kaplan-yorke dim
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings

@dataclass
class RegimeConfig:
    """configuration for regime analysis"""
    # confidence interval settings
    min_trajectories: int = 50
    bias_fit_points: int = 5
    confidence_level: float = 0.95
    min_time_points: int = 99
    
    # deterministic intervals settings
    attractor_bound: float = 2.5  # |x|,|y| ≤ R for henon
    
    # kaplan-yorke tracking
    enable_ky_tracking: bool = True
    ky_convergence_threshold: float = 0.01  # fractional vs integer detection
    
    def get_z_score(self) -> float:
        """z-score for confidence level"""
        if self.confidence_level == 0.95:
            return 1.96
        elif self.confidence_level == 0.99:
            return 2.576
        else:
            from scipy.stats import norm
            return norm.ppf((1 + self.confidence_level) / 2)


@dataclass
class RegimeClassification:
    """regime classification result with confidence levels"""
    regime_type: str  # sync/chaos/hyperchaos/uncertain
    confidence_source: str  # confidence_interval/deterministic/statistical_uncertainty
    lambda1_bounds: Tuple[float, float]
    lambda2_bounds: Optional[Tuple[float, float]] = None
    ky_dimension: Optional[float] = None
    confidence_level: float = 0.95
    
    @property
    def is_chaotic(self) -> bool:
        return "chaos" in self.regime_type.lower()
    
    @property
    def is_synchronized(self) -> bool:
        return "sync" in self.regime_type.lower()
    
    @property
    def is_hyperchaotic(self) -> bool:
        return "hyperchaos" in self.regime_type.lower()
    
    @property
    def has_strange_attractor(self) -> bool:
        if self.ky_dimension is None:
            return False
        return not np.isclose(self.ky_dimension, round(self.ky_dimension), atol=0.01)


class RegimeAnalyzer:
    """regime analysis using confidence intervals + deterministic intervals"""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.trajectory_data: List[List[Dict]] = []
        self.analysis_history: List[Dict] = []
        
    def add_trajectory(self, lyap_history: List[Dict]) -> None:
        """add independent trajectory for confidence interval analysis"""
        if len(lyap_history) >= self.config.min_time_points:
            self.trajectory_data.append(lyap_history)
        else:
            warnings.warn(f"trajectory too short ({len(lyap_history)} < {self.config.min_time_points})")
    
    def reset_trajectories(self) -> None:
        """clear trajectory data"""
        self.trajectory_data = []
        self.analysis_history = []
    
    def analyze_regime(self, 
                      attractor_params: Dict[str, float],
                      noise_amplitude: float,
                      target_times: Optional[List[int]] = None,
                      use_deterministic_heuristic: bool = True) -> Dict[str, Any]:
        """
        comprehensive regime analysis with deterministic heuristic optimization
        
        if use_deterministic_heuristic=True:
        1. compute cheap deterministic intervals first
        2. if bounds are clearly positive/negative, conclude regime immediately
        3. only do expensive confidence interval analysis if bounds cross zero (uncertain)
        """
        
        # deterministic intervals analysis (always cheap, ~milliseconds)
        deterministic_analysis = self._analyze_deterministic_bounds(
            attractor_params, noise_amplitude
        )
        
        # deterministic heuristic filtering
        if use_deterministic_heuristic and not deterministic_analysis.get('error'):
            det_bounds = deterministic_analysis.get('lambda1_bounds')
            
            if det_bounds:
                bounds_min, bounds_max = det_bounds
                margin = 0.01  # require clear separation from zero
                
                # clear regime from deterministic intervals alone?
                if bounds_max < -margin:
                    # clearly synchronized - skip expensive analysis
                    print(f"  deterministic heuristic: clearly synchronized (λ₁ ∈ [{bounds_min:.4f}, {bounds_max:.4f}])")
                    return self._quick_deterministic_result(
                        attractor_params, noise_amplitude, deterministic_analysis, "synchronized"
                    )
                elif bounds_min > margin:
                    # clearly chaotic - skip expensive analysis  
                    print(f"  deterministic heuristic: clearly chaotic (λ₁ ∈ [{bounds_min:.4f}, {bounds_max:.4f}])")
                    return self._quick_deterministic_result(
                        attractor_params, noise_amplitude, deterministic_analysis, "chaotic"
                    )
                else:
                    # bounds cross zero - need expensive confidence interval analysis
                    bounds_width = bounds_max - bounds_min
                    print(f"  deterministic intervals uncertain (λ₁ ∈ [{bounds_min:.4f}, {bounds_max:.4f}], width={bounds_width:.4f})")
                    print(f"  proceeding with confidence interval analysis...")
        
        # expensive confidence interval analysis (for uncertain cases)
        confidence_analysis = self._analyze_confidence_intervals(target_times)
        
        # regime classification using both methods
        regime_classification = self._classify_regime(
            confidence_analysis, deterministic_analysis
        )
        
        # kaplan-yorke dimension tracking
        ky_analysis = self._analyze_kaplan_yorke(confidence_analysis)
        
        # set ky_dimension in regime classification if available
        if ky_analysis.get('ky_dimension') is not None:
            regime_classification.ky_dimension = ky_analysis['ky_dimension']
        
        # package comprehensive results
        analysis = {
            'timestamp': confidence_analysis.get('timestamp', 0),
            'noise_amplitude': noise_amplitude,
            'attractor_params': attractor_params,
            'confidence_analysis': confidence_analysis,
            'deterministic_analysis': deterministic_analysis,
            'regime_classification': regime_classification,
            'kaplan_yorke_analysis': ky_analysis,
            'scaling_comparison': self._compare_bound_scaling(
                confidence_analysis, deterministic_analysis
            ),
            'used_deterministic_heuristic': use_deterministic_heuristic
        }
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _quick_deterministic_result(self, 
                                   attractor_params: Dict[str, float],
                                   noise_amplitude: float,
                                   deterministic_analysis: Dict[str, Any],
                                   regime_type: str) -> Dict[str, Any]:
        """create quick result using only deterministic intervals"""
        
        det_bounds = deterministic_analysis['lambda1_bounds']
        
        classification = RegimeClassification(
            regime_type=regime_type,
            confidence_source="deterministic_heuristic", 
            lambda1_bounds=det_bounds,
            confidence_level=1.0  # 100% certain from deterministic intervals
        )
        
        # quick kaplan-yorke estimate (if possible)
        ky_analysis = {'enabled': False, 'reason': 'deterministic_heuristic_used'}
        
        return {
            'timestamp': 0,
            'noise_amplitude': noise_amplitude,
            'attractor_params': attractor_params,
            'confidence_analysis': {'skipped': 'deterministic_heuristic_sufficient'},
            'deterministic_analysis': deterministic_analysis,
            'regime_classification': classification,
            'kaplan_yorke_analysis': ky_analysis,
            'scaling_comparison': {
                'deterministic_width': det_bounds[1] - det_bounds[0],
                'confidence_interval_width': 'not_computed',
                'heuristic_used': True
            },
            'used_deterministic_heuristic': True
        }
    
    def _analyze_confidence_intervals(self, target_times: Optional[List[int]] = None) -> Dict[str, Any]:
        """confidence interval analysis"""
        print(f"    DEBUG: starting confidence interval analysis with {len(self.trajectory_data)} trajectories")
        
        if len(self.trajectory_data) < self.config.min_trajectories:
            return {'error': f"need {self.config.min_trajectories} trajectories, got {len(self.trajectory_data)}"}
        
        # determine analysis time points
        if target_times is None:
            target_times = self._auto_determine_times()
        
        if len(target_times) < self.config.bias_fit_points:
            return {'error': f"need {self.config.bias_fit_points} time points for bias fitting"}
        
        # extract time series data across trajectories
        time_series_data = self._extract_time_series(target_times)
        if not time_series_data:
            return {'error': "failed to extract time series data"}
        
        # fit scaling laws: bias ~ B/T, fluctuations ~ σ/√T
        bias_analysis = self._fit_bias_scaling(time_series_data)
        fluctuation_analysis = self._fit_fluctuation_scaling(time_series_data)
        
        # construct confidence intervals
        interval_analysis = self._construct_confidence_intervals(
            time_series_data, bias_analysis, fluctuation_analysis
        )
        
        result = {
            'num_trajectories': len(self.trajectory_data),
            'target_times': target_times,
            'bias_analysis': bias_analysis,
            'fluctuation_analysis': fluctuation_analysis,
            'confidence_intervals': interval_analysis,
            'scaling_diagnostics': self._compute_scaling_diagnostics(bias_analysis, fluctuation_analysis)
        }
        
        print(f"    DEBUG: confidence interval final width: {interval_analysis.get('final_interval', {}).get('width', 'N/A')}")
        return result
    
    def _analyze_deterministic_bounds(self, 
                                    attractor_params: Dict[str, float],
                                    noise_amplitude: float) -> Dict[str, Any]:
        """deterministic envelope bounds (from deterministic_bounds.py)"""
        
        # currently only henon implemented - generalize later
        if 'a' in attractor_params and 'b' in attractor_params:
            return self._henon_deterministic_bounds(
                attractor_params['a'], attractor_params['b'], noise_amplitude
            )
        else:
            return {'error': f"deterministic intervals not implemented for these parameters: {attractor_params}"}
    
    def _henon_deterministic_bounds(self, a: float, b: float, alpha: float) -> Dict[str, Any]:
        """deterministic intervals for noisy henon map"""
        R = self.config.attractor_bound
        
        # parameter ranges under noise
        a_min, a_max = a - alpha, a + alpha
        b_min, b_max = b - alpha, b + alpha
        
        # jacobian: J = [[-2*a_n*x, 1], [b_n, 0]]
        # compute bounds over all possible (a_n, b_n, x) combinations
        corner_norms = []
        min_norms = []
        
        for a_val in [a_min, a_max]:
            for b_val in [b_min, b_max]:
                for x_val in [-R, R]:
                    J = np.array([[-2*a_val*x_val, 1], [b_val, 0]])
                    # largest singular value = spectral norm
                    svd_vals = np.linalg.svd(J, compute_uv=False)
                    corner_norms.append(svd_vals[0])  # largest
                    min_norms.append(svd_vals[-1])   # smallest
        
        max_norm = max(corner_norms)
        min_norm = min(min_norms)
        
        # lyapunov bounds: λ ∈ [ln(min_norm), ln(max_norm)]
        lambda_min = np.log(min_norm)
        lambda_max = np.log(max_norm)
        
        return {
            'lambda1_bounds': (lambda_min, lambda_max),
            'jacobian_norm_bounds': (min_norm, max_norm),
            'interval_width': lambda_max - lambda_min,
            'noise_amplitude': alpha,
            'attractor_bound': R
        }
    
    def _classify_regime(self, 
                        confidence_analysis: Dict[str, Any],
                        deterministic_analysis: Dict[str, Any]) -> RegimeClassification:
        """regime classification using tightest actionable bounds"""
        
        # extract bounds from each method
        conf_intervals = confidence_analysis.get('confidence_intervals', {})
        det_bounds = deterministic_analysis.get('lambda1_bounds')
        
        # confidence interval bounds (if available and valid)
        if conf_intervals and 'final_interval' in conf_intervals:
            final_interval = conf_intervals['final_interval']
            conf_bounds = (final_interval['lower_bound'], final_interval['upper_bound'])
            conf_center = final_interval['lambda_infinity']
            
            # extract lambda2 bounds if available
            lambda2_bounds = None
            if 'lambda2_infinity' in final_interval:
                # reconstruct lambda2 bounds from Intervals data
                Intervals = conf_intervals.get('Intervals', [])
                if Intervals:
                    latest_interval = Intervals[-1]
                    lambda2_bounds = (latest_interval['lambda2_lower'], latest_interval['lambda2_upper'])
            
            # use confidence bounds if they're tight enough for classification
            if conf_bounds[1] < 0:
                return RegimeClassification(
                    regime_type="synchronized", 
                    confidence_source="confidence_interval",
                    lambda1_bounds=conf_bounds,
                    lambda2_bounds=lambda2_bounds,
                    confidence_level=self.config.confidence_level
                )
            elif conf_bounds[0] > 0:
                return RegimeClassification(
                    regime_type="chaotic", 
                    confidence_source="confidence_interval",
                    lambda1_bounds=conf_bounds,
                    lambda2_bounds=lambda2_bounds,
                    confidence_level=self.config.confidence_level
                )
            else:
                # confidence interval straddles zero - check if center is decisive
                if abs(conf_center) > (conf_bounds[1] - conf_bounds[0]) / 4:
                    regime = "chaotic" if conf_center > 0 else "synchronized"
                    return RegimeClassification(
                        regime_type=f"{regime}_likely",
                        confidence_source="confidence_interval_center",
                        lambda1_bounds=conf_bounds,
                        lambda2_bounds=lambda2_bounds,
                        confidence_level=self.config.confidence_level * 0.8  # reduced confidence
                    )
        
        # fallback to deterministic intervals
        if det_bounds:
            if det_bounds[1] < 0:
                return RegimeClassification(
                    regime_type="synchronized",
                    confidence_source="deterministic", 
                    lambda1_bounds=det_bounds,
                    confidence_level=1.0
                )
            elif det_bounds[0] > 0:
                return RegimeClassification(
                    regime_type="chaotic",
                    confidence_source="deterministic",
                    lambda1_bounds=det_bounds, 
                    confidence_level=1.0
                )
            else:
                return RegimeClassification(
                    regime_type="uncertain",
                    confidence_source="deterministic",
                    lambda1_bounds=det_bounds,
                    confidence_level=1.0
                )
        
        # no valid bounds available
        return RegimeClassification(
            regime_type="undefined",
            confidence_source="insufficient_data",
            lambda1_bounds=(np.nan, np.nan),
            confidence_level=0.0
        )
    
    def _analyze_kaplan_yorke(self, confidence_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """kaplan-yorke dimension analysis"""
        if not self.config.enable_ky_tracking:
            return {'enabled': False}
        
        conf_intervals = confidence_analysis.get('confidence_intervals', {})
        if not conf_intervals or 'final_interval' not in conf_intervals:
            return {'error': 'no confidence interval data for ky analysis'}
        
        final_interval = conf_intervals['final_interval']
        
        # extract λ₁, λ₂ estimates (use center values)
        lambda1 = final_interval.get('lambda_infinity')
        lambda2 = final_interval.get('lambda2_infinity')  # if available
        
        if lambda1 is None or lambda2 is None:
            return {'error': 'need both λ₁ and λ₂ for kaplan-yorke dimension'}
        
        # compute kaplan-yorke dimension: D_KY = j + Σᵢ₌₁ʲ λᵢ / |λⱼ₊₁|
        # for 2d system: D_KY = 1 + λ₁/|λ₂| if λ₁ > 0, λ₂ < 0
        ky_dim = None
        interpretation = "undefined"
        
        if lambda1 > 0 and lambda2 < 0:
            ky_dim = 1 + lambda1 / abs(lambda2)
            
            if np.isclose(ky_dim, 1.0, atol=self.config.ky_convergence_threshold):
                interpretation = "approaching_limit_cycle"
            elif np.isclose(ky_dim, 2.0, atol=self.config.ky_convergence_threshold):
                interpretation = "approaching_torus"
            elif 1 < ky_dim < 2:
                interpretation = "strange_attractor"
            else:
                interpretation = "unusual_geometry"
        
        elif lambda1 < 0 and lambda2 < 0:
            ky_dim = 0.0  # both negative → fixed point
            interpretation = "fixed_point"
        
        elif lambda1 > 0 and lambda2 > 0:
            interpretation = "hyperchaotic"  # can't compute standard ky
        
        return {
            'enabled': True,
            'lambda1': lambda1,
            'lambda2': lambda2,
            'ky_dimension': ky_dim,
            'interpretation': interpretation,
            'is_strange': ky_dim is not None and not np.isclose(ky_dim, round(ky_dim), atol=self.config.ky_convergence_threshold)
        }
    
    def _compare_bound_scaling(self, 
                              confidence_analysis: Dict[str, Any],
                              deterministic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """compare scaling behavior of confidence vs deterministic intervals"""
        
        conf_intervals = confidence_analysis.get('confidence_intervals', {})
        det_bounds = deterministic_analysis.get('lambda1_bounds')
        
        if not conf_intervals or not det_bounds or 'final_interval' not in conf_intervals:
            return {'error': 'insufficient data for bound comparison'}
        
        final_interval = conf_intervals['final_interval']
        conf_width = final_interval['upper_bound'] - final_interval['lower_bound']
        det_width = det_bounds[1] - det_bounds[0]
        
        return {
            'confidence_interval_width': conf_width,
            'deterministic_width': det_width,
            'width_ratio': det_width / conf_width if conf_width > 0 else np.inf,
            'confidence_tighter_by_factor': det_width / conf_width if conf_width > 0 else np.inf,
            'bounds_consistent': (det_bounds[0] <= final_interval['lower_bound'] and 
                                final_interval['upper_bound'] <= det_bounds[1])
        }
    
    # helper methods from original confidence_intervals.py
    def _auto_determine_times(self) -> List[int]:
        """auto-determine analysis time points"""
        min_length = min(len(traj) for traj in self.trajectory_data)
        max_time = min(traj[-1]['time'] for traj in self.trajectory_data)
        min_time = max(traj[0]['time'] for traj in self.trajectory_data)
        
        n_points = min(self.config.bias_fit_points * 2, min_length // 10)
        if n_points < self.config.bias_fit_points:
            return list(range(min_time + self.config.min_time_points, 
                            max_time + 1, 
                            (max_time - min_time) // self.config.bias_fit_points))
        
        log_times = np.logspace(np.log10(min_time + self.config.min_time_points), 
                               np.log10(max_time), 
                               n_points)
        return [int(t) for t in log_times]
    
    def _extract_time_series(self, target_times: List[int]) -> List[Dict]:
        """extract lyapunov estimates at specific times across trajectories"""
        print(f"      DEBUG: extracting time series from {len(self.trajectory_data)} trajectories")
        
        # DEBUG: print keys from first trajectory point
        if self.trajectory_data and len(self.trajectory_data[0]) > 0:
            first_point = self.trajectory_data[0][0]
            print(f"      DEBUG: available keys in trajectory data: {list(first_point.keys())}")
        
        time_series = []
        
        for time_point in target_times:
            estimates_lyap1 = []
            estimates_lyap2 = []
            actual_times = []
            
            for trajectory in self.trajectory_data:
                times = [h['time'] for h in trajectory]
                closest_idx = np.argmin(np.abs(np.array(times) - time_point))
                
                if abs(times[closest_idx] - time_point) <= time_point * 0.1:
                    closest_point = trajectory[closest_idx]
                    estimates_lyap1.append(closest_point['mean_lyap1'])
                    # DEBUG: check if mean_lyap2 exists
                    if 'mean_lyap2' in closest_point:
                        estimates_lyap2.append(closest_point['mean_lyap2'])
                    else:
                        print(f"      DEBUG: mean_lyap2 missing at time {closest_point['time']}")
                        estimates_lyap2.append(0.0)  # fallback
                    actual_times.append(times[closest_idx])
            
            print(f"      DEBUG: time {time_point}: {len(estimates_lyap1)} estimates")
            print(f"      DEBUG: λ₁ estimates: {estimates_lyap1}")
            print(f"      DEBUG: λ₂ estimates: {estimates_lyap2}")
            print(f"      DEBUG: λ₁ variance: {np.var(estimates_lyap1):.8f}")
            
            if len(estimates_lyap1) >= max(3, self.config.min_trajectories // 2):
                time_series.append({
                    'target_time': time_point,
                    'actual_time': np.mean(actual_times),
                    'lyap1_estimates': np.array(estimates_lyap1),
                    'lyap2_estimates': np.array(estimates_lyap2),
                    'num_estimates': len(estimates_lyap1)
                })
        
        return time_series
    
    def _fit_bias_scaling(self, time_series_data: List[Dict]) -> Dict[str, Any]:
        """fit bias scaling: E[λ̂₁,T] ≈ λ∞ + B/T"""
        times = np.array([ts['actual_time'] for ts in time_series_data])
        mean_lyap1 = np.array([np.mean(ts['lyap1_estimates']) for ts in time_series_data])
        mean_lyap2 = np.array([np.mean(ts['lyap2_estimates']) for ts in time_series_data])
        
        inv_times = 1.0 / times
        
        # λ₁ bias fitting
        slope1, intercept1 = np.polyfit(inv_times, mean_lyap1, 1)
        pred1 = slope1 * inv_times + intercept1
        ss_res1 = np.sum((mean_lyap1 - pred1) ** 2)
        ss_tot1 = np.sum((mean_lyap1 - np.mean(mean_lyap1)) ** 2)
        r_squared1 = 1 - (ss_res1 / (ss_tot1 + 1e-10))
        
        # λ₂ bias fitting
        slope2, intercept2 = np.polyfit(inv_times, mean_lyap2, 1)
        pred2 = slope2 * inv_times + intercept2
        ss_res2 = np.sum((mean_lyap2 - pred2) ** 2)
        ss_tot2 = np.sum((mean_lyap2 - np.mean(mean_lyap2)) ** 2)
        r_squared2 = 1 - (ss_res2 / (ss_tot2 + 1e-10))
        
        return {
            'lyap1': {
                'lambda_infinity': intercept1,
                'bias_coefficient': slope1,
                'r_squared': r_squared1,
                'fit_quality': 'good' if r_squared1 > 0.8 else ('moderate' if r_squared1 > 0.5 else 'poor')
            },
            'lyap2': {
                'lambda_infinity': intercept2,
                'bias_coefficient': slope2,
                'r_squared': r_squared2,
                'fit_quality': 'good' if r_squared2 > 0.8 else ('moderate' if r_squared2 > 0.5 else 'poor')
            }
        }
    
    def _fit_fluctuation_scaling(self, time_series_data: List[Dict]) -> Dict[str, Any]:
        """fit fluctuation scaling: Var(λ̂₁,T) ≈ σ²/T"""
        print(f"      DEBUG: fluctuation scaling analysis")
        times = np.array([ts['actual_time'] for ts in time_series_data])
        var_lyap1 = np.array([np.var(ts['lyap1_estimates']) for ts in time_series_data])
        var_lyap2 = np.array([np.var(ts['lyap2_estimates']) for ts in time_series_data])
        
        print(f"      DEBUG: variances across time points: {var_lyap1}")
        
        inv_times = 1.0 / times
        
        # λ₁ fluctuation fitting
        slope1, intercept1 = np.polyfit(inv_times, var_lyap1, 1)
        pred1 = slope1 * inv_times + intercept1
        ss_res1 = np.sum((var_lyap1 - pred1) ** 2)
        ss_tot1 = np.sum((var_lyap1 - np.mean(var_lyap1)) ** 2)
        r_squared1 = 1 - (ss_res1 / (ss_tot1 + 1e-10))
        
        print(f"      DEBUG: sigma_squared (slope): {slope1:.8f}")
        print(f"      DEBUG: baseline_variance (intercept): {intercept1:.8f}")
        
        # λ₂ fluctuation fitting
        slope2, intercept2 = np.polyfit(inv_times, var_lyap2, 1)
        pred2 = slope2 * inv_times + intercept2
        ss_res2 = np.sum((var_lyap2 - pred2) ** 2)
        ss_tot2 = np.sum((var_lyap2 - np.mean(var_lyap2)) ** 2)
        r_squared2 = 1 - (ss_res2 / (ss_tot2 + 1e-10))
        
        return {
            'lyap1': {
                'sigma_squared': slope1,
                'baseline_variance': intercept1,
                'r_squared': r_squared1,
                'fit_quality': 'good' if r_squared1 > 0.8 else ('moderate' if r_squared1 > 0.5 else 'poor')
            },
            'lyap2': {
                'sigma_squared': slope2,
                'baseline_variance': intercept2,
                'r_squared': r_squared2,
                'fit_quality': 'good' if r_squared2 > 0.8 else ('moderate' if r_squared2 > 0.5 else 'poor')
            }
        }
    
    def _construct_confidence_intervals(self, 
                                      time_series_data: List[Dict],
                                      bias_analysis: Dict,
                                      fluctuation_analysis: Dict) -> Dict[str, Any]:
        """construct confidence intervals from scaling laws"""
        print(f"      DEBUG: constructing confidence intervals")
        z_score = self.config.get_z_score()
        
        print(f"      DEBUG: raw sigma_squared: {fluctuation_analysis['lyap1']['sigma_squared']:.8f}")
        
        intervals = []
        for ts_data in time_series_data:
            T = ts_data['actual_time']
            
            # bias-corrected estimates
            lambda1_corrected = (np.mean(ts_data['lyap1_estimates']) - 
                               bias_analysis['lyap1']['bias_coefficient'] / T)
            lambda2_corrected = (np.mean(ts_data['lyap2_estimates']) - 
                               bias_analysis['lyap2']['bias_coefficient'] / T)
            
            # confidence intervals with variance floor to prevent collapse
            sigma_sq1 = max(fluctuation_analysis['lyap1']['sigma_squared'], 1e-10)
            sigma_sq2 = max(fluctuation_analysis['lyap2']['sigma_squared'], 1e-10)
            
            print(f"      DEBUG: sigma_sq1 after floor: {sigma_sq1:.8f}")
            
            std_err1 = np.sqrt(sigma_sq1 / T)
            std_err2 = np.sqrt(sigma_sq2 / T)
            
            print(f"      DEBUG: std_err1: {std_err1:.8f}, width1: {2 * z_score * std_err1:.8f}")
            
            intervals.append({
                'time': T,
                'lambda1_center': lambda1_corrected,
                'lambda1_lower': lambda1_corrected - z_score * std_err1,
                'lambda1_upper': lambda1_corrected + z_score * std_err1,
                'lambda2_center': lambda2_corrected,
                'lambda2_lower': lambda2_corrected - z_score * std_err2,
                'lambda2_upper': lambda2_corrected + z_score * std_err2,
                'width1': 2 * z_score * std_err1,
                'width2': 2 * z_score * std_err2
            })
        
        # final estimates (latest time point)
        final_interval = intervals[-1] if intervals else None
        
        return {
            'intervals': intervals,
            'final_interval': {
                'lambda_infinity': final_interval['lambda1_center'],
                'lambda2_infinity': final_interval['lambda2_center'],
                'lower_bound': final_interval['lambda1_lower'],
                'upper_bound': final_interval['lambda1_upper'],
                'width': final_interval['width1'],
                'time': final_interval['time']
            } if final_interval else None,
            'convergence_trends': self._assess_convergence_trends(intervals)
        }
    
    def _assess_convergence_trends(self, intervals: List[Dict]) -> Dict[str, Any]:
        """assess convergence quality from interval sequence"""
        if len(intervals) < 3:
            return {'quality': 'insufficient_data'}
        
        times = np.array([interval['time'] for interval in intervals])
        widths1 = np.array([interval['width1'] for interval in intervals])
        centers1 = np.array([interval['lambda1_center'] for interval in intervals])
        
        # handle near-zero widths to prevent log(0) errors
        min_width = 1e-12
        widths1_safe = np.maximum(widths1, min_width)
        
        # width should decrease as ~1/√T
        log_times = np.log(times)
        log_widths = np.log(widths1_safe)
        width_slope, _ = np.polyfit(log_times, log_widths, 1)
        
        # center should stabilize (low trend)
        center_slope, _ = np.polyfit(times, centers1, 1)
        
        quality = "good"
        if abs(width_slope + 0.5) > 0.2:  # should be ~-0.5
            quality = "poor_width_scaling"
        elif abs(center_slope) > 1e-4:
            quality = "unstable_center"
        elif widths1[-1] > 0 and widths1[0] > 0 and widths1[-1] / widths1[0] > 0.5:  # insufficient narrowing
            quality = "slow_convergence"
        elif np.any(widths1 <= min_width):  # near-zero variance detected
            quality = "near_deterministic"
        
        # safe division for width reduction factor
        width_reduction = (widths1[0] / widths1[-1]) if widths1[-1] > 0 else np.inf
        
        return {
            'quality': quality,
            'width_scaling_exponent': width_slope,
            'center_trend': center_slope,
            'final_width': widths1[-1],
            'width_reduction_factor': width_reduction
        }
    
    def _compute_scaling_diagnostics(self, bias_analysis: Dict, fluctuation_analysis: Dict) -> Dict[str, Any]:
        """diagnostic metrics for scaling law quality"""
        return {
            'bias_fit_r2_lambda1': bias_analysis['lyap1']['r_squared'],
            'bias_fit_r2_lambda2': bias_analysis['lyap2']['r_squared'],
            'fluctuation_fit_r2_lambda1': fluctuation_analysis['lyap1']['r_squared'],
            'fluctuation_fit_r2_lambda2': fluctuation_analysis['lyap2']['r_squared'],
            'overall_quality': min(
                bias_analysis['lyap1']['r_squared'],
                fluctuation_analysis['lyap1']['r_squared']
            )
        }
    
    def get_latest_analysis(self) -> Optional[Dict[str, Any]]:
        """get most recent analysis result"""
        return self.analysis_history[-1] if self.analysis_history else None
    
    def format_regime_summary(self, analysis: Optional[Dict] = None) -> str:
        """format human-readable regime summary"""
        if analysis is None:
            analysis = self.get_latest_analysis()
        
        if not analysis:
            return "no regime analysis available"
        
        classification = analysis['regime_classification']
        ky_analysis = analysis['kaplan_yorke_analysis']
        
        summary = f"regime: {classification.regime_type}"
        summary += f" ({classification.confidence_level*100:.0f}% confidence via {classification.confidence_source})"
        
        bounds = classification.lambda1_bounds
        summary += f"\nλ₁ ∈ [{bounds[0]:.4f}, {bounds[1]:.4f}]"
        
        if ky_analysis.get('ky_dimension') is not None:
            ky_dim = ky_analysis['ky_dimension']
            interpretation = ky_analysis['interpretation']
            summary += f"\nkaplan-yorke dimension: {ky_dim:.3f} ({interpretation})"
        
        scaling = analysis['scaling_comparison']
        if not scaling.get('error'):
            ratio = scaling.get('confidence_tighter_by_factor', 0)
            summary += f"\nconfidence intervals {ratio:.0f}x tighter than deterministic intervals"
        
        return summary


# convenience functions for quick analysis
def analyze_regime_quick(attractor_params: Dict[str, float], 
                        noise_amplitude: float,
                        trajectory_data: List[List[Dict]],
                        config: Optional[RegimeConfig] = None) -> RegimeClassification:
    """quick regime classification from trajectory data"""
    if config is None:
        config = RegimeConfig()
    
    analyzer = RegimeAnalyzer(config)
    for traj in trajectory_data:
        analyzer.add_trajectory(traj)
    
    analysis = analyzer.analyze_regime(attractor_params, noise_amplitude)
    return analysis['regime_classification']


def find_critical_noise(attractor_params: Dict[str, float],
                       noise_range: Tuple[float, float],
                       trajectory_data_fn,  # function: noise_amp -> List[List[Dict]]
                       num_points: int = 20,
                       config: Optional[RegimeConfig] = None) -> Optional[float]:
    """find critical noise threshold where regime transitions"""
    if config is None:
        config = RegimeConfig()
    
    noise_amplitudes = np.linspace(noise_range[0], noise_range[1], num_points)
    regime_types = []
    
    for noise_amp in noise_amplitudes:
        trajectory_data = trajectory_data_fn(noise_amp)
        classification = analyze_regime_quick(attractor_params, noise_amp, trajectory_data, config)
        regime_types.append(classification.regime_type)
    
    # find transition points
    for i in range(len(regime_types) - 1):
        if ("chaos" in regime_types[i] and "sync" in regime_types[i+1]) or \
           ("sync" in regime_types[i] and "chaos" in regime_types[i+1]):
            # linear interpolation to find critical point
            return (noise_amplitudes[i] + noise_amplitudes[i+1]) / 2
    
    return None 
