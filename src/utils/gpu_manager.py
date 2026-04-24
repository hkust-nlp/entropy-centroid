"""
GPU device management utilities.
"""

import os
from typing import List, Optional


class GPUManager:
    """
    Manage GPU device selection and configuration.
    """

    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        Initialize GPU manager.

        Args:
            device_ids: List of GPU IDs to use (e.g., [0, 1, 2, 3])
        """
        self.device_ids = device_ids
        self.available_gpus = self._get_available_gpus()

    @staticmethod
    def _get_available_gpus() -> int:
        """
        Get the number of available GPUs.

        Returns:
            Number of available GPUs
        """
        try:
            import torch
            return torch.cuda.device_count()
        except Exception:
            return 0

    def configure(self) -> int:
        """
        Configure GPU devices based on device_ids.

        Returns:
            Number of GPUs configured for use
        """
        if self.device_ids is None or len(self.device_ids) == 0:
            print("No GPU devices specified, using all available GPUs")
            return self.available_gpus

        # Validate device IDs
        for device_id in self.device_ids:
            if device_id >= self.available_gpus:
                raise ValueError(
                    f"GPU ID {device_id} not available. "
                    f"Only {self.available_gpus} GPUs detected."
                )

        # Set CUDA_VISIBLE_DEVICES
        device_str = ",".join(map(str, self.device_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        print(f"Configured GPUs: {device_str}")

        return len(self.device_ids)

    def get_tensor_parallel_size(self, requested_size: Optional[int] = None) -> int:
        """
        Get the tensor parallel size based on available GPUs.

        Args:
            requested_size: Requested tensor parallel size

        Returns:
            Actual tensor parallel size to use
        """
        num_gpus = len(self.device_ids) if self.device_ids else self.available_gpus

        if requested_size is None:
            return num_gpus

        if requested_size > num_gpus:
            print(
                f"Warning: Requested tensor_parallel_size={requested_size} "
                f"exceeds available GPUs ({num_gpus}). Using {num_gpus} instead."
            )
            return num_gpus

        return requested_size

    def print_gpu_info(self):
        """Print GPU information."""
        print(f"Total available GPUs: {self.available_gpus}")
        if self.device_ids:
            print(f"Using GPU IDs: {self.device_ids}")
        else:
            print("Using all available GPUs")

        try:
            import torch
            for i in range(self.available_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.2f} GB)")
        except Exception:
            pass


def create_gpu_manager(config: dict) -> GPUManager:
    """
    Create a GPU manager from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        GPUManager instance
    """
    gpu_config = config.get("gpu", {})
    device_ids = gpu_config.get("device_ids")

    return GPUManager(device_ids=device_ids)
