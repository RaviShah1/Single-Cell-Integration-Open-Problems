class CFG:
    tr_batch_size = 16 # 16
    va_batch_size = 128 # 32
    
    optimizer = "AdamW"
    lr = 1e-5
    weight_decay = 0.1
    betas = (0.9, 0.999)
    epochs = 50