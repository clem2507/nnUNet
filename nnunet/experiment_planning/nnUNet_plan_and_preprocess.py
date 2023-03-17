#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import nnunet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.utils import crop
from nnunet.paths import *
import shutil
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity
from nnunet.training.model_restore import recursive_find_python_class


def main(input_folder, 
         planner3d="ExperimentPlanner3D_v21", 
         planner2d="ExperimentPlanner2D_v21", 
         no_preprocessing=False, 
         num_processes_lowres=8, 
         num_processes_fullres=8, 
         verify_dataset=True, 
         overwrite_plans=None, 
         overwrite_plans_identifier=None):
    
    '''
    This is the main function for planning and preprocessing nnU-Net.

    args:
        input_folder: str
            Path to the folder containing the raw data
        planner3d: str
            Name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade
            Default is ExperimentPlanner3D_v21
            Can be 'None', in which case these U-Nets will not be configured
        planner2d: str
            Name of the ExperimentPlanner class for the 2D U-Net
            Default is ExperimentPlanner2D_v21
            Can be 'None', in which case this U-Net will not be configured
        no_preprocessing: bool
            Set this True if you dont want to run the preprocessing
            If this is set then this script will only run the experiment planning and create the plans file
        num_processes_lowres: int
            Number of processes used for preprocessing the low resolution data for the 3D low resolution U-Net
            This can be larger than num_processes_fullres
            Don't overdo it or you will run out of RAM
        num_processes_fullres: int
            Number of processes used for preprocessing the full resolution data of the 2D U-Net and 3D U-Net
            Don't overdo it or you will run out of RAM
        verify_dataset: bool
            Set this True to check the dataset integrity
            This is useful and should be done once for each dataset!
        overwrite_plans: str
            Use this to specify a plans file that should be used instead of whatever nnUNet plans to use
            This is useful if you want to use a different set of parameters than nnUNet would use
            If this is set then the planner3d and planner2d arguments are ignored
        overwrite_plans_identifier: str
            Use this to specify a plans identifier that should be used instead of whatever nnUNet plans to use
            This is useful if you want to use a different set of parameters than nnUNet would use
            If this is set then the planner3d and planner2d arguments are ignored
    '''
    dont_run_preprocessing = no_preprocessing
    tl = num_processes_lowres
    tf = num_processes_fullres
    planner_name3d = planner3d
    planner_name2d = planner2d

    if planner_name3d == "None":
        planner_name3d = None
    if planner_name2d == "None":
        planner_name2d = None

    if overwrite_plans is not None:
        if planner_name2d is not None:
            print("Overwriting plans only works for the 3d planner. I am setting '--planner2d' to None. This will "
                  "skip 2d planning and preprocessing.")
        assert planner_name3d == 'ExperimentPlanner3D_v21_Pretrained', "When using --overwrite_plans you need to use " \
                                                                       "'-pl3d ExperimentPlanner3D_v21_Pretrained'"

    if verify_dataset:
        verify_dataset_integrity(input_folder)

    crop(input_folder, True, tf)

    search_in = join(nnunet.__path__[0], "experiment_planning")

    if planner_name3d is not None:
        planner_3d = recursive_find_python_class([search_in], planner_name3d, current_module="nnunet.experiment_planning")
        if planner_3d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name3d)
    else:
        planner_3d = None

    if planner_name2d is not None:
        planner_2d = recursive_find_python_class([search_in], planner_name2d, current_module="nnunet.experiment_planning")
        if planner_2d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name2d)
    else:
        planner_2d = None

    cropped_out_dir = os.path.join(input_folder, "preprocessed_data", "cropped")
    preprocessing_output_dir_this_task = os.path.join(input_folder, "preprocessed_data", "final")
    #splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, t)
    #lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
    dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
    modalities = list(dataset_json["modality"].values())
    collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
    dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=tf)  # this class creates the fingerprint
    _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner


    maybe_mkdir_p(preprocessing_output_dir_this_task)
    shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
    shutil.copy(join(input_folder, "dataset.json"), preprocessing_output_dir_this_task)

    threads = (tl, tf)

    print("number of threads: ", threads, "\n")

    if planner_3d is not None:
        if overwrite_plans is not None:
            assert overwrite_plans_identifier is not None, "You need to specify -overwrite_plans_identifier"
            exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task, overwrite_plans,
                                        overwrite_plans_identifier)
        else:
            exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
        exp_planner.plan_experiment()
        if not dont_run_preprocessing:  # double negative, yooo
            exp_planner.run_preprocessing(threads)
    if planner_2d is not None:
        exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir_this_task)
        exp_planner.plan_experiment()
        if not dont_run_preprocessing:  # double negative, yooo
            exp_planner.run_preprocessing(threads)


if __name__ == "__main__":
    main()

