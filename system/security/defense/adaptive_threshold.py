import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque, defaultdict


class HistoricalSigmaAdaptiveThreshold:
    """Adaptive threshold manager based on historical M' variance detection"""
    
    def __init__(self, 
                 initial_threshold: float = 2.0,
                 history_window: int = 10,
                 variance_threshold: float = 1.0,
                 top_k_for_threshold: int = 3):
        """
        Args:
            initial_threshold: Initial threshold
            history_window: Number of historical M' values per client
            variance_threshold: Variance anomaly threshold (default 1.0, update if exceeded)
            top_k_for_threshold: Number of top-K M' values for threshold calculation
        """
        self.current_threshold = initial_threshold
        self.history_window = history_window
        self.variance_threshold = variance_threshold
        self.top_k_for_threshold = top_k_for_threshold
        
        # Client history M' storage (independently maintained per client)
        self.client_histories = defaultdict(lambda: deque(maxlen=history_window))
        
        # Round M' storage: key=round_number, value=all M' values in that round
        self.round_m_primes = {}  # {round_number: [m_prime1, m_prime2, ...]}
        
        # Global historical M' storage (for statistics)
        self.all_m_primes = []
        
        # Statistics
        self.current_round = 0
        self.threshold_updates = 0
        self.total_detections = 0
        self.anomaly_detections = 0
        self.threshold_history = [initial_threshold]
        
        print(f"HistoricalSigmaAdaptiveThreshold initialized:")
        print(f"  - Initial threshold: {initial_threshold}")
        print(f"  - History window per client: {history_window}")
        print(f"  - Variance threshold: {variance_threshold}")
        print(f"  - Top-K for threshold update: {top_k_for_threshold}")
        print("  - Strategy: Use previous round M' values for variance detection and threshold update")
    
    def start_round(self, round_number: int) -> float:
        """
        Start new round, update threshold using previous round M' values
        
        Args:
            round_number: Current round number
            
        Returns:
            Threshold for current round
        """
        self.current_round = round_number
        
        # Round 0 uses initial threshold
        if round_number == 0:
            print(f"[Round {round_number}] First round (round 0), using initial threshold: {self.current_threshold}")
            return self.current_threshold
        
        # From round 1, use previous round M' values to evaluate threshold update
        previous_round = round_number - 1
        if previous_round in self.round_m_primes:
            previous_m_primes = self.round_m_primes[previous_round]
            print(f"[Round {round_number}] Evaluating threshold based on round {previous_round} M' values: {[f'{x:.3f}' for x in previous_m_primes]}")
            
            new_threshold = self._evaluate_threshold_update(previous_m_primes, previous_round)
            
            if new_threshold != self.current_threshold:
                print(f"[Round {round_number}] Threshold updated: {self.current_threshold:.3f} -> {new_threshold:.3f}")
                self.current_threshold = new_threshold
                self.threshold_updates += 1
                self.threshold_history.append(new_threshold)
            else:
                print(f"[Round {round_number}] No threshold update needed, using threshold: {self.current_threshold:.3f}")
        else:
            print(f"[Round {round_number}] No data from previous round, using current threshold: {self.current_threshold:.3f}")
        
        return self.current_threshold
    
    def add_detection(self, m_prime: float, client_id: Optional[int] = None) -> float:
        """
        Add new M' detection result to current round
        
        Args:
            m_prime: Current detected M' value
            client_id: Client ID (optional, for maintaining client history)
            
        Returns:
            Current threshold
        """
        self.total_detections += 1
        
        # Add M' value to current round
        if self.current_round not in self.round_m_primes:
            self.round_m_primes[self.current_round] = []
        self.round_m_primes[self.current_round].append(m_prime)
        
        # Add M' value to global history
        self.all_m_primes.append(m_prime)
        
        # If client ID provided, add to corresponding history
        if client_id is not None:
            self.client_histories[client_id].append(m_prime)
        
        return self.current_threshold
    

    
    def _evaluate_threshold_update(self, previous_m_primes: List[float], previous_round: int) -> float:
        """
        Evaluate if threshold update needed
        Strategy: If variance > 1, next round threshold = mean of top-3 M' values
        
        Args:
            previous_m_primes: All M' values from previous round
            previous_round: Previous round number
            
        Returns:
            New threshold (current threshold if no update needed)
        """
        if len(previous_m_primes) == 0:
            return self.current_threshold
        
        # Calculate statistics of previous round M' values
        m_values = np.array(previous_m_primes)
        mean_m = np.mean(m_values)
        variance_m = np.var(m_values)
        std_m = np.std(m_values)
        
        print(f"    Round {previous_round} statistics: mean={mean_m:.4f}, variance={variance_m:.4f}, std={std_m:.4f}")
        
        # Check if previous round variance > 1
        if variance_m > 1.0:
            # Use top-K M' values from previous round to calculate new threshold
            sorted_m_values = sorted(previous_m_primes, reverse=True)
            top_k_values = sorted_m_values[:self.top_k_for_threshold]
            new_threshold = np.mean(top_k_values)
            
            self.anomaly_detections += 1  # Record anomaly detection
            
            print(f"    High variance detected in round {previous_round}: {variance_m:.4f} > 1.0")
            print(f"    Round {previous_round} M' values: {[f'{x:.3f}' for x in previous_m_primes]}")
            print(f"    Top-{self.top_k_for_threshold} M' values from round {previous_round}: {[f'{x:.3f}' for x in top_k_values]}")
            print(f"    Calculated new threshold: {new_threshold:.4f}")
            
            return new_threshold
        else:
            # Previous round variance normal, threshold unchanged
            print(f"    Normal variance in round {previous_round}: {variance_m:.4f} <= 1.0, no threshold update")
            return self.current_threshold
    
    def get_current_threshold(self) -> float:
        """获取当前阈值"""
        return self.current_threshold
    
    def get_client_history(self, client_id: int) -> List[float]:
        """Get historical M' values for specified client"""
        if client_id in self.client_histories:
            return list(self.client_histories[client_id])
        return []
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'strategy': 'historical_3sigma_per_round',
            'current_threshold': self.current_threshold,
            'total_detections': self.total_detections,
            'anomaly_detections': self.anomaly_detections,
            'threshold_updates': self.threshold_updates,
            'rounds_processed': self.current_round,
            'clients_tracked': len(self.client_histories),
            'total_m_values': len(self.all_m_primes)
        }
        
        if len(self.all_m_primes) > 1:
            m_values = np.array(list(self.all_m_primes))
            stats.update({
                'current_m_mean': float(np.mean(m_values)),
                'current_m_std': float(np.std(m_values)),
                'current_m_min': float(np.min(m_values)),
                'current_m_max': float(np.max(m_values))
            })
        
        if len(self.threshold_history) > 1:
            stats.update({
                'threshold_range': (float(np.min(self.threshold_history)), float(np.max(self.threshold_history))),
                'threshold_stability': float(np.std(self.threshold_history))
            })
        
        return stats
    
    def reset(self):
        """Reset manager state"""
        self.client_histories.clear()
        self.all_m_primes.clear()
        self.round_m_primes.clear()
        self.threshold_history = [self.current_threshold]
        
        self.current_round = 0
        self.threshold_updates = 0
        self.total_detections = 0
        self.anomaly_detections = 0
        
        print("HistoricalSigmaAdaptiveThreshold reset completed")


class AdaptiveThresholdManager:
    """Unified interface for adaptive threshold manager"""
    
    def __init__(self, strategy: str = 'historical_3sigma', **kwargs):
        """
        Args:
            strategy: Adaptive strategy type
            **kwargs: Strategy-specific parameters
        """
        self.strategy = strategy
        
        if strategy == 'historical_3sigma':
            self.threshold_manager = HistoricalSigmaAdaptiveThreshold(**kwargs)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
    
    def start_round(self, round_number: int) -> float:
        """Start new round and return current threshold"""
        if hasattr(self.threshold_manager, 'start_round'):
            return self.threshold_manager.start_round(round_number)
        return self.threshold_manager.get_current_threshold()
    
    def add_detection(self, m_prime: float, client_id: Optional[int] = None) -> float:
        """Add detection result and return current threshold"""
        return self.threshold_manager.add_detection(m_prime, client_id)
    
    def end_round(self, round_number: int) -> float:
        """End current round and update threshold"""
        if hasattr(self.threshold_manager, 'end_round'):
            return self.threshold_manager.end_round(round_number)
        return self.threshold_manager.get_current_threshold()
    
    def get_current_threshold(self) -> float:
        return self.threshold_manager.get_current_threshold()
    
    def get_client_history(self, client_id: int) -> List[float]:
        """Get client history"""
        if hasattr(self.threshold_manager, 'get_client_history'):
            return self.threshold_manager.get_client_history(client_id)
        return []
    
    def get_statistics(self) -> Dict:
        return self.threshold_manager.get_statistics()
    
    def reset(self):
        """Reset manager"""
        self.threshold_manager.reset()