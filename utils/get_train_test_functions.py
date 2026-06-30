from utils.training import train_one_epoch
from utils.testing import test_one_epoch 
def get_train_test_functions(config):
    """
    Get the appropriate training and testing functions based on model configuration.
    
    Args:
        config: Model configuration dictionary containing model name
        
    Returns:
        tuple: (train_function, test_function) appropriate for the model type
    """
    model_name = config.get('name', '')

    
    if model_name:
        return train_one_epoch, test_one_epoch
    
    raise ValueError(f"Unsupported model name: {model_name}. Please provide a valid model name in the configuration.")