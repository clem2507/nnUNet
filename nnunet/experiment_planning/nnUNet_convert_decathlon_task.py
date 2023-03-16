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
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.configuration import default_num_threads
from nnunet.experiment_planning.utils import split_4d
from nnunet.utilities.file_endings import remove_trailing_slash


def crawl_and_remove_hidden_from_decathlon(folder):
    folder = remove_trailing_slash(folder)
    assert folder.split('/')[-1].startswith("Task"), "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    subf = subfolders(folder, join=False)
    assert 'imagesTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    assert 'imagesTs' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    assert 'labelsTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"


def main(input_folder, num_threads=default_num_threads, output_task_id=None):
    '''
    This is a utility to convert the 4D niftis that the MSD provides into the format nnU-Net expects.

    args:
        input_folder: str
            Path to the TaskXX folder that you downloaded from the MSD website
        num_threads: int
            How many processes to use for the conversion
        output_task_id: int
            If specified, this will overwrite the task id in the output folder
            If unspecified, the task id of the input folder will be used
    '''

    crawl_and_remove_hidden_from_decathlon(input_folder)
    split_4d(input_folder, num_threads, output_task_id)


if __name__ == "__main__":
    main()
