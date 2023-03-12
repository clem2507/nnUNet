# import torch
# from datetime import datetime

from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


# default parameters
use_compressed_data = False # if True, the training cases will not be decompressed. Reading compressed data is much more CPU and RAM intensive and should only be used if you know what you are doing
decompress_data = not use_compressed_data 
deterministic = False # makes training deterministic, but reduces training speed substantially. Deterministic training will make you overfit to some random seed. Don't use that.
fp32 = False # disable mixed precision training and run old school fp32
run_mixed_precision = not fp32
val_folder = "validation_raw" # name of the validation folder, no need to use this for most people
npz = False # makes the models save the softmax outputs during the final validation. It should only be used for trainings where you plan to run nnUNet_find_best_configuration afterwards (this is nnU-Nets automated selection of the best performing (ensemble of) configuration(s), see below). If you are developing new trainer classes you may not need the softmax predictions and should therefore omit the --npz flag.

def load_model_trainer(network, network_trainer, task, fold, plans_identifier=default_plans_identifier):
    ''' Method to load the model according to the given parameters

    Args:
        network::str
            Network name that identifies the requested U-Net configuration
        network_trainer::str 
            Name of the model trainer
            If you implement custom trainers (nnU-Net as a framework) you can specify your custom trainer here
        task::int 
            Task id specifies what dataset should be trained
        fold::str
            Fold id specifies which fold of the 5-fold-cross-validaton is trained
        plans_identifier::str
            Identifier for the experiment planner
            Only change this if you created a custom experiment planner

    Returns:
        trainer::NetworkTrainer
            Trainer module of the requested model
    '''

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)), \
            "If running 3d_cascade_fullres then your " \
            "trainer class must be derived from " \
            "nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class,
                          nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision)
    return trainer

def train(trainer, 
          max_num_epochs=1000, 
          patience=50,
          num_batches_per_epoch=250,
          num_val_batches_per_epoch=50,
          lr_threshold=1e-6,
          val_eval_criterion_alpha=0.9,
          wandb_log=False):
    ''' Method to train the model

    Args:
        trainer::NetworkTrainer
            Trainer module of the model to be trained
        max_num_epochs::int
            Maximum number of epochs to train the model
        patience::int
            Number of epochs to wait for the validation loss to improve before terminating the training
        num_batches_per_epoch::int
            Number of batches per epoch
        num_val_batches_per_epoch::int
            Number of validation batches per epoch
        lr_threshold::float
            Threshold for the learning rate. If the learning rate is above this threshold the training will not terminate
        val_eval_criterion_alpha::float
            Alpha value for the moving average of the validation loss
            If this is too low then the moving average will be too noisy and the training may terminate early
            If it is too high the training will take forever
        wandb_log::bool
            If True, wandb will log the training process
    '''

    trainer.max_num_epochs = max_num_epochs
    trainer.patience = patience
    trainer.num_batches_per_epoch = num_batches_per_epoch
    trainer.num_val_batches_per_epoch = num_val_batches_per_epoch
    trainer.lr_threshold = lr_threshold
    trainer.val_eval_criterion_alpha = val_eval_criterion_alpha
    trainer.wandb_log = wandb_log

    trainer.initialize(True)
    trainer.run_training()
    trainer.network.eval()

    # predict validation
    trainer.validate(save_softmax=npz, 
                     validation_folder_name=val_folder,
                     run_postprocessing_on_folds=False,
                     overwrite=False)
