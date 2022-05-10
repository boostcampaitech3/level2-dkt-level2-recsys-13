def average_ensemble(all_preds):
    return [ sum(preds) / len(all_preds) for preds in zip(*all_preds) ]