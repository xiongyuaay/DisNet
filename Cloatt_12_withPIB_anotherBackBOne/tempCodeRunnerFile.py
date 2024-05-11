 ## train-prior
    opt_lr = scheduler.get_last_lr()
    train_prior(model,train_dataloader,
                optimizer,
                loss_func,
                scheduler,
                args.scale,
                batch_size=args.batch_size)
    