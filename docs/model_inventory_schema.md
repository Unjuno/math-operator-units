# Model Inventory Schema

Each completed seed writes `model_inventory.json` containing the random initialization, trained base, five specialist checkpoints, joint reference checkpoints, tokenizer ABI identifiers, checkpoint schedule, and runtime subset count. Fusion code should consume this file rather than reconstructing paths by convention.
