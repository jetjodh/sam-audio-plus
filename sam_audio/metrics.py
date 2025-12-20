# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Performance metrics collection and reporting for SAM-Audio.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import threading
import torch

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

from sam_audio.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MetricEntry:
    """A single metric measurement."""
    
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Centralized collector for performance metrics.
    
    Tracks timing, memory usage, and other custom metrics.
    Designed to be a singleton-like global instance.
    """
    
    _instance: Optional[MetricsCollector] = None
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics: Dict[str, List[MetricEntry]] = defaultdict(list)
        self.timers: Dict[str, float] = {}
        self.timers: Dict[str, float] = {}
        self.start_time = time.time()
        
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
            except Exception:
                pass
        
    @classmethod
    def get_instance(cls) -> MetricsCollector:
        """Get the global MetricsCollector instance."""
        if cls._instance is None:
            # Check env var for default enabled state
            enabled = os.environ.get("SAM_AUDIO_METRICS", "1") != "0"
            cls._instance = cls(enabled=enabled)
        return cls._instance
        
    def reset(self):
        """Reset all collected metrics."""
        self.metrics.clear()
        self.timers.clear()
        self.start_time = time.time()
        
    def start_timer(self, name: str):
        """Start a named timer."""
        if not self.enabled:
            return
        self.timers[name] = time.perf_counter()
        
    def stop_timer(self, name: str, unit: str = "s"):
        """Stop a named timer and record the duration."""
        if not self.enabled or name not in self.timers:
            return
        
        elapsed = time.perf_counter() - self.timers.pop(name)
        self.metrics[name].append(MetricEntry(value=elapsed, unit=unit))
        
    def log_metric(self, name: str, value: float, unit: str = ""):
        """Log a specific metric value."""
        if not self.enabled:
            return
        self.metrics[name].append(MetricEntry(value=value, unit=unit))
        
    def log_memory(self, name: str, device: Optional[torch.device] = None):
        """Log current VRAM usage."""
        if not self.enabled:
            return
            
        if device is None and torch.cuda.is_available():
            device = torch.device("cuda")
            
        if device and device.type == "cuda":
            # Record allocated and reserved memory in GB
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            
            self.metrics[f"{name}_vram_allocated"].append(MetricEntry(value=allocated, unit="GB"))
            self.metrics[f"{name}_vram_reserved"].append(MetricEntry(value=reserved, unit="GB"))
            self.metrics[f"{name}_vram_allocated"].append(MetricEntry(value=allocated, unit="GB"))
            self.metrics[f"{name}_vram_reserved"].append(MetricEntry(value=reserved, unit="GB"))

    def start_monitoring(self, interval: float = 0.5):
        """Start background monitoring of system metrics (GPU utilization)."""
        if not self.enabled or not HAS_PYNVML or self._monitoring_thread is not None:
            return

        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self._monitoring_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring."""
        if self._monitoring_thread is None:
            return
            
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=1.0)
        self._monitoring_thread = None

    def _monitor_loop(self, interval: float):
        """Loop to poll system metrics."""
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return
                
            # Monitor device 0 by default (or all?)
            # Simplified: just monitor device 0 for now as SAM-Audio is single-gpu mostly
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            while not self._stop_monitoring.is_set():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.metrics["gpu_utilization"].append(
                        MetricEntry(value=util.gpu, unit="%")
                    )
                    # memory utilization also available: util.memory
                except Exception:
                    pass
                time.sleep(interval)
        except Exception:
            pass

    @contextlib.contextmanager
    def measure_time(self, name: str):
        """Context manager to measure execution time."""
        if not self.enabled:
            yield
            return
            
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.metrics[name].append(MetricEntry(value=elapsed, unit="s"))

    def track_time(self, name: str):
        """Decorator to measure execution time of a function."""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                with self.measure_time(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def get_summary(self) -> Dict[str, Any]:
        """Compute summary statistics for all metrics."""
        summary = {}
        for name, entries in self.metrics.items():
            if not entries:
                continue
            
            values = [e.value for e in entries]
            avg = sum(values) / len(values)
            unit = entries[0].unit
            
            summary[name] = {
                "mean": avg,
                "min": min(values),
                "max": max(values),
                "count": len(values),
                "last": values[-1],
                "unit": unit,
                "total": sum(values) if unit == "s" else None 
            }
        return summary

    def get_report_table(self) -> str:
        """Generate a formatted table report of metrics."""
        if not self.metrics:
            return "No metrics collected."
            
        summary = self.get_summary()
        
        # Categorize metrics
        timing_metrics = {k: v for k, v in summary.items() if v["unit"] == "s"}
        memory_metrics = {k: v for k, v in summary.items() if "GB" in v["unit"]}
        system_metrics = {k: v for k, v in summary.items() if "%" in v["unit"]}
        other_metrics = {k: v for k, v in summary.items() 
                        if k not in timing_metrics 
                        and k not in memory_metrics
                        and k not in system_metrics}
        
        lines = []
        lines.append("\n" + "="*60)
        lines.append(f"{'PERFORMANCE METRICS REPORT':^60}")
        lines.append("="*60)
        
        if timing_metrics:
            lines.append(f"\n{' TIMING ':~^60}")
            lines.append(f"{'Metric':<30} | {'Last':>10} | {'Mean':>10} | {'Total':>10}")
            lines.append("-" * 66)
            for name, stats in sorted(timing_metrics.items()):
                lines.append(f"{name:<30} | {stats['last']:>9.3f}s | {stats['mean']:>9.3f}s | {stats['total'] or 0:>9.3f}s")
                
        if memory_metrics:
            lines.append(f"\n{' MEMORY (VRAM) ':~^60}")
            lines.append(f"{'Metric':<30} | {'Peak':>10} | {'Ave':>10} | {'Last':>10}")
            lines.append("-" * 66)
            for name, stats in sorted(memory_metrics.items()):
                lines.append(f"{name:<30} | {stats['max']:>9.2f}GB | {stats['mean']:>9.2f}GB | {stats['last']:>9.2f}GB")

        if system_metrics:
            lines.append(f"\n{' SYSTEM (GPU) ':~^60}")
            lines.append(f"{'Metric':<30} | {'Peak':>10} | {'Mean':>10} | {'Last':>10}")
            lines.append("-" * 66)
            for name, stats in sorted(system_metrics.items()):
                lines.append(f"{name:<30} | {stats['max']:>9.1f}%  | {stats['mean']:>9.1f}%  | {stats['last']:>9.1f}% ")
                
        if other_metrics:
            lines.append(f"\n{' OTHER ':~^60}")
            lines.append(f"{'Metric':<30} | {'Value':>10} | {'Unit':>10}")
            lines.append("-" * 66)
            for name, stats in sorted(other_metrics.items()):
                lines.append(f"{name:<30} | {stats['last']:>10.3f} | {stats['unit']:>10}")
                
        lines.append("="*60 + "\n")
        return "\n".join(lines)

    def export_json(self, path: str):
        """Export metrics to a JSON file."""
        data = {
            "timestamp": self.start_time,
            "metrics": {
                name: [
                    {"value": e.value, "unit": e.unit, "timestamp": e.timestamp} 
                    for e in entries
                ] 
                for name, entries in self.metrics.items()
            },
            "summary": self.get_summary()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Metrics exported to {path}")


# Global helper functions for easy access
def get_metrics_collector() -> MetricsCollector:
    return MetricsCollector.get_instance()

def track_time(name: str):
    """Decorator to track execution time."""
    return MetricsCollector.get_instance().track_time(name)

def measure_time(name: str):
    """Context manager to measure execution time."""
    return MetricsCollector.get_instance().measure_time(name)

def log_metric(name: str, value: float, unit: str = ""):
    """Log a metric value."""
    MetricsCollector.get_instance().log_metric(name, value, unit)

def log_memory(name: str, device: Optional[torch.device] = None):
    """Log memory usage."""
    MetricsCollector.get_instance().log_memory(name, device)
