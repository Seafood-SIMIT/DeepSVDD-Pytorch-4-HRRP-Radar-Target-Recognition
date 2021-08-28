
import torch

def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, deep_SVDD):
    #device
    if torch.cuda.is_available():
        logger.info("Device GPU")
        device='cuda'
        
    else:
        logger.info("Device CPU")
        device='cpu'

    if hp.train.pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % hp.ae.optimizer)
        logger.info('Pretraining learning rate: %g' % hp.ae.lr)
        logger.info('Pretraining epochs: %d' % hp.ae.epochs)
        logger.info('Pretraining learning rate scheduler milestones: %s' % hp.ae.lr_milestone)
        logger.info('Pretraining batch size: %d' % hp.ae.batch_size)
        logger.info('Pretraining weight decay: %g' % hp.ae.weight_decay)

        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(trainloader,
                           testloader,
                           optimizer_name=hp.ae.optimizer,
                           lr=hp.ae.lr,
                           n_epochs=hp.ae.epochs,
                           lr_milestones=(hp.ae.lr_milestone, ),
                           batch_size=hp.ae.batch_size,
                           weight_decay=hp.ae.weight_decay,
                           device=device,
                           n_jobs_dataloader=hp.ae.n_jobs_dataloader)
    # Log training details
    logger.info('Training optimizer: %s' % hp.train.optimizer)
    logger.info('Training learning rate: %g' % hp.train.lr)
    logger.info('Training epochs: %d' % hp.train.epochs)
    logger.info('Training learning rate scheduler milestones: %s' % hp.train.lr_milestone)
    logger.info('Training batch size: %d' % hp.train.batch_size)
    logger.info('Training weight decay: %g' % hp.train.weight_decay)
    

    # Train model on dataset
    deep_SVDD.train(trainloader,
                           optimizer_name=hp.train.optimizer,
                           lr=hp.train.lr,
                           n_epochs=hp.train.epochs,
                           lr_milestones=(hp.train.lr_milestone, ),
                           batch_size=hp.train.batch_size,
                           weight_decay=hp.train.weight_decay,
                           device=device,
                           n_jobs_dataloader=hp.train.n_jobs_dataloader)

    # Test model
    deep_SVDD.test(testloader, device=device, n_jobs_dataloader=0)