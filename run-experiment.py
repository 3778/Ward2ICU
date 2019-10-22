import mlflow
import tempfile
import click
import torch
import torch.nn as nn
import numpy as np
from ward2icu.data import TimeSeriesVitalSigns
from ward2icu.logs import log_avg_loss, log_avg_grad, log_model, log_df
from ward2icu.models import CNNCGANGenerator, CNNCGANDiscriminator
from ward2icu.utils import synthesis_df, train_test_split_tensor, numpy_to_cuda, tile
from ward2icu.metrics import mean_feature_error, classify, tstr
from ward2icu.samplers import BinaryBalancedSampler, IdentitySampler
from ward2icu.trainers import BinaryClassificationTrainer, MinMaxBinaryCGANTrainer, SequenceTrainer
from torch import optim
from torch.utils.data import DataLoader
from torchgan import losses
from torchgan.metrics import ClassifierScore
from torchgan.trainer import Trainer
from slugify import slugify
from dmpy.datascience import DataPond
from ward2icu import make_logger

logger = make_logger(__file__)

@click.command()
@click.option("--lr", type=float)
@click.option("--epochs", type=int)
@click.option("--ncritic", type=int)
@click.option("--batch_size", type=int)
@click.option("--dataset_transform", type=str)
@click.option("--signals", type=int)
@click.option("--gen_dropout", type=float)
@click.option("--noise_size", type=int)
@click.option("--hidden_size", type=int)
@click.option("--flag", type=str)
def cli(**opt):
    main(**opt)

def main(**opt):
    logger.info(opt)
    batch_size = opt['batch_size'] if opt['batch_size'] != -1 else None

    dataset = TimeSeriesVitalSigns(transform=opt['dataset_transform'],
                                   vital_signs=opt['signals'])
    X = torch.from_numpy(dataset.X).cuda()
    y = torch.from_numpy(dataset.y).long().cuda()
    sampler = BinaryBalancedSampler(X, y, batch_size=batch_size)

    network = {
        'generator': {
            'name': CNNCGANGenerator,
            'args': {
                'output_size': opt['signals'],
                'dropout': opt['gen_dropout'],
                'noise_size': opt['noise_size'],
                'hidden_size': opt['hidden_size']
            },
            'optimizer': {
                'name': optim.RMSprop,
                'args': {
                    'lr': opt['lr']
                }
            }
        },
        'discriminator': {
            'name': CNNCGANDiscriminator,
            'args': {
                'input_size': opt['signals'],
                'hidden_size': opt['hidden_size']
            },
            'optimizer': {
                'name': optim.RMSprop,
                'args': {
                    'lr': opt['lr']
                }
            }
        }
    }

    wasserstein_losses = [losses.WassersteinGeneratorLoss(),
                          losses.WassersteinDiscriminatorLoss(),
                          losses.WassersteinGradientPenalty()]

    logger.info(network)

    trainer = SequenceTrainer(models=network,
                      recon=None,
                      ncritic=opt['ncritic'],
                      losses_list=wasserstein_losses,
                      epochs=opt['epochs'],
                      retain_checkpoints=1,
                      checkpoints=f"{MODEL_DIR}/",
                      mlflow_interval=50,
                      device=DEVICE)

    trainer(sampler=sampler)
    trainer.log_to_mlflow()
    logger.info(trainer.generator)
    logger.info(trainer.discriminator)

    df_synth, X_synth, y_synth = synthesis_df(trainer.generator, dataset)

    logger.info(df_synth.sample(10))
    logger.info(df_synth.groupby('cat_vital_sign')['value'].nunique()
                .div(df_synth.groupby('cat_vital_sign').size()))
    X_real = X.detach().cpu().numpy()
    mfe = np.abs(mean_feature_error(X_real, X_synth))
    logger.info(f'Mean feature error: {mfe}')

    mlflow.set_tag('flag', opt['flag'])
    log_df(df_synth, 'synthetic/vital_signs')
    mlflow.log_metric('mean_feature_error', mfe)

    trainer_class = classify(X_synth, y_synth, epochs=2_000, batch_size=batch_size)
    trainer_tstr = tstr(X_synth, y_synth, X, y, epochs=3_000, batch_size=batch_size)
    log_model(trainer_class.model, 'models/classifier')
    log_model(trainer_tstr.model, 'models/tstr')

if __name__ == '__main__':
    with mlflow.start_run():
        with tempfile.TemporaryDirectory() as MODEL_DIR:
            if torch.cuda.is_available():
                DEVICE = torch.device("cuda")
                torch.backends.cudnn.deterministic = True
            else:
                DEVICE = torch.device("cpu")
            logger.info(f'Running on device {DEVICE}')
            cli()
