"""Optional: Hardware-aware batch size estimation."""

import torch
from typing import Optional


class HardwareAnalyzer:
    """Estimate batch size based on GPU memory."""
    
    @staticmethod
    def get_available_vram() -> float:
        """
        Get available GPU VRAM in GB.
        
        Returns:
            VRAM in GB, or 0 if no GPU available
        """
        if not torch.cuda.is_available():
            return 0.0
        
        # Get total memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Get allocated memory
        allocated = torch.cuda.memory_allocated(0) / 1e9
        
        return total_memory - allocated
    
    @staticmethod
    def estimate_batch_size(
        vram_available: float,
        model_memory_gb: float = 2.0,
        per_sample_memory_gb: float = 0.5
    ) -> int:
        """
        Estimate max batch size.
        
        Args:
            vram_available: Available VRAM in GB
            model_memory_gb: Approximate model size in GB
            per_sample_memory_gb: Approximate per-sample overhead in GB
            
        Returns:
            Recommended batch size
        """
        available_for_batch = vram_available - model_memory_gb
        if available_for_batch <= 0:
            return 1
        
        batch_size = int(available_for_batch / per_sample_memory_gb)
        
        # Round to power of 2
        for bs in [4, 8, 16, 32, 64, 128, 256]:
            if bs > batch_size:
                return max(4, bs // 2)
        
        return 256
    
    @staticmethod
    def suggest_batch_size() -> Optional[int]:
        """
        Suggest batch size based on current GPU state.
        
        Returns:
            Suggested batch size, or None if no GPU
        """
        if not torch.cuda.is_available():
            return None
        
        vram = HardwareAnalyzer.get_available_vram()
        return HardwareAnalyzer.estimate_batch_size(vram)
