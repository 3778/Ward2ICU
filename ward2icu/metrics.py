from ward2icu.samplers import BinaryBalancedSampler, IdentitySampler
from ward2icu.models import BinaryRNNClassifier
from torch import optim
from ward2icu.utils import (train_test_split_tensor,
                            numpy_to_cuda, tile)
from ward2icu.trainers import BinaryClassificationTrainer
from ward2icu import make_logger

logger = make_logger(__file__)


def mean_feature_error(X_real, X_synth):
    return (1 - X_synth.mean(axis=0)/X_real.mean(axis=0)).mean(axis=0).mean()


# TODO(dsevero) in theory, we should seperate the
# training and testing datasets.
def tstr(X_synth, y_synth, X_real, y_real, epochs=3_000, batch_size=None):

    logger.info('Running TSTR')
    logger.info(f'Synthetic size: {len(y_synth)}')
    logger.info(f'Real size: {len(y_real)}')
    logger.info(f'Class distribution: '
                f'[real {y_real.float().mean()}]'
                f'[synthetic {y_synth.mean()}]')

    sequence_length = X_real.shape[1]
    sequence_size = X_real.shape[2]
    X_train, y_train = X_synth, y_synth
    X_test, y_test = X_real, y_real

    X_train, y_train = numpy_to_cuda(X_train, y_train)
    y_test, y_train = y_test.float(), y_train.float()

    sampler_train = BinaryBalancedSampler(X_train, y_train, 
                                          tile=True, batch_size=batch_size)
    sampler_test = IdentitySampler(X_test, y_test, tile=True)

    model = BinaryRNNClassifier(sequence_length=sequence_length,
                                input_size=sequence_size,
                                dropout=0.5,
                                hidden_size=100).cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=0.001)
    trainer = BinaryClassificationTrainer(model,
                                          optimizer,
                                          sampler_train,
                                          sampler_test,
                                          metrics_prepend='tstr_')
    trainer.train(epochs)
    return trainer


def classify(X, y, epochs=3_000, batch_size=None):
    sequence_length = X.shape[1]
    sequence_size = X.shape[2]

    (X_train, y_train, 
     X_test, y_test) = train_test_split_tensor(X, y, 0.3)
    (X_train, y_train, 
     X_test, y_test) = numpy_to_cuda(X_train, y_train,
                                     X_test, y_test)
    y_test, y_train = y_test.float(), y_train.float()

    sampler_train = BinaryBalancedSampler(X_train, y_train, 
                                          tile=True, batch_size=batch_size)
    sampler_test = IdentitySampler(X_test, y_test, tile=True)

    model = BinaryRNNClassifier(sequence_length=sequence_length,
                                input_size=sequence_size,
                                dropout=0.8,
                                hidden_size=100).cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=0.001)
    trainer = BinaryClassificationTrainer(model,
                                          optimizer,
                                          sampler_train,
                                          sampler_test)
    logger.info(model)
    logger.info(f'Test size: {len(y_test)}')
    logger.info(f'Train size: {len(y_train)}')

    trainer.train(epochs)
    return trainer
