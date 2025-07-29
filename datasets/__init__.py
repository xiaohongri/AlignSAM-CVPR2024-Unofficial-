from datasets.coco_dataset import CocoDataset

def get_dataset(dataset_config):
    """
    Factory function to get the dataset based on the configuration.
    
    Args:
        dataset_config (dict): Configuration for the dataset.
        
    Returns:
        Dataset: An instance of the specified dataset.
    """
    if dataset_config['type'] == 'coco':
        return CocoDataset(**dataset_config)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_config['type']}")
