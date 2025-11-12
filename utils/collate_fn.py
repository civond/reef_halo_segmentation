# Concats all dicts into one larger dict

def collate_fn(batch):
    return tuple(zip(*batch))