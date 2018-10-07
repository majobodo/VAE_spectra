import os
import warnings
import pickle

import numpy as np
from copy import deepcopy
from abc import ABCMeta, abstractmethod

from VAE_spectra.Dataset import SpectralData


class TaskCreator(object):

    def __init__(self,
                 evaluation_dataset,
                 result_location):
        """
        :type evaluation_dataset SpectralData
        """

        if not os.path.isdir(result_location):
            raise ValueError("Result location not found")

        # a simple check of the data
        if not evaluation_dataset.m_spectra.shape[0] == evaluation_dataset.m_label.shape[0]:
            raise ValueError("Number of stars in spectra and labels does not match")

        self.m_evaluation_dataset = evaluation_dataset
        self.m_splits = []

        # create Evaluator sub-folder
        self.m_result_location = result_location + "/Evaluation/"

        if not os.path.isdir(result_location + "/Evaluation/"):
            os.makedirs(result_location + "/Evaluation/")
        else:
            print("Found existing Evaluation workspace. Restore splits ...")
            self.restore_splits()

    def create_cross_validation_splits(self,
                                       fold):
        number_of_stars = self.m_evaluation_dataset.m_spectra.shape[0]

        random_list = np.arange(0, number_of_stars)
        np.random.shuffle(random_list)

        batches = np.array_split(random_list, fold)

        if os.path.exists(self.m_result_location + "Splits"):
            raise RuntimeError("Split folder already exists")

        os.makedirs(self.m_result_location + "Splits")

        # create the Splits
        for i, _ in enumerate(batches):
            tmp_test = batches[i]
            tmp_val = batches[i-1]
            tmp_train = np.setdiff1d(np.concatenate(batches[:]),
                                     np.concatenate([tmp_test, tmp_val]))

            fold_folder = self.m_result_location + "/Splits/Fold_" + str(i).zfill(2)
            os.makedirs(fold_folder)

            np.savetxt(fold_folder + "/Test_splits_fold_" + str(i).zfill(2) + ".txt",
                       tmp_test,
                       header='Test - Fold ' + str(i).zfill(2),
                       delimiter=',',
                       fmt='%i')

            np.savetxt(fold_folder + "/Validation_splits_fold_" + str(i).zfill(2) + ".txt",
                       tmp_val,
                       header='Validation - Fold ' + str(i).zfill(2),
                       delimiter=',',
                       fmt='%i')

            np.savetxt(fold_folder + "/Train_splits_fold_" + str(i).zfill(2) + ".txt",
                       tmp_train,
                       header='Train - Fold ' + str(i).zfill(2),
                       delimiter=',',
                       fmt='%i')

            self.m_splits.append((tmp_train, tmp_val, tmp_test))

    def restore_splits(self):

        self.m_splits.clear()

        if not os.path.exists(self.m_result_location + "Splits"):
            raise RuntimeError("Split folder not found. "
                               "Run create_cross_validation_splits in order to create one.")

        for subdir, dirs, files in os.walk(self.m_result_location + "Splits"):
            tmp_test, tmp_val, tmp_train = None, None, None
            miss = False
            for fname in os.listdir(subdir):
                if fname.startswith('Test'):
                    tmp_test = np.loadtxt(subdir + "/" + fname, dtype=np.int)
                elif fname.startswith('Validation'):
                    tmp_val = np.loadtxt(subdir + "/" + fname, dtype=np.int)
                elif fname.startswith('Train'):
                    tmp_train = np.loadtxt(subdir + "/" + fname, dtype=np.int)
                else:
                    miss = True

            if not miss:
                # check if the split belongs to the dataset we are working on
                if not np.concatenate([tmp_train, tmp_val, tmp_test]).shape[0] \
                       == self.m_evaluation_dataset.m_spectra.shape[0]:
                    raise ValueError("Dataset does not match Splits")

                self.m_splits.append((tmp_train, tmp_val, tmp_test))

    def create_task_from_splits(self,
                                id,
                                Task_sub_folder="Tasks/"):

        if not os.path.exists(self.m_result_location + Task_sub_folder):
            os.makedirs(self.m_result_location + Task_sub_folder)

        return EvaluationTask(self.m_evaluation_dataset[self.m_splits[id][0]],
                              self.m_evaluation_dataset[self.m_splits[id][1]],
                              self.m_evaluation_dataset[self.m_splits[id][2]],
                              self.m_splits[id][0],
                              self.m_splits[id][1],
                              self.m_splits[id][2],
                              id,
                              self.m_result_location + Task_sub_folder)


class EvaluationTask(object):

    def __init__(self,
                 training_data,
                 validation_data,
                 test_data,
                 training_splint_ids=None,
                 validation_split_ids=None,
                 test_split_ids=None,
                 task_id=None,
                 default_save_dir=None):
        """
        :type training_data SpectralData
        :type validation_data SpectralData
        :type test_data SpectralData
        """

        # Save the data
        self.m_training_data = training_data
        self.m_validation_data = validation_data
        self.m_test_data = test_data

        self.m_default_save_dir = default_save_dir
        self.m_task_id = task_id

        # Save splits
        self.m_training_splint_ids = training_splint_ids
        self.m_validation_split_ids = validation_split_ids
        self.m_test_split_ids = test_split_ids

        # Create empty datasets for test and validation results with same
        # size as the validation and test data

        def create_empty_copy(dataset):
            return SpectralData(deepcopy(dataset.m_wavelength),
                                np.zeros_like(dataset.m_spectra),
                                np.zeros_like(dataset.m_spectra_ivar),
                                np.zeros_like(dataset.m_label),
                                np.zeros_like(dataset.m_label_ivar),
                                label_names=dataset.m_label_names)

        self.m_validation_result = create_empty_copy(self.m_validation_data)
        self.m_test_results = create_empty_copy(self.m_test_data)
        self.m_run = False

    def run_task(self,
                 model):
        """
        :type model SpectralModel
        """
        # Train the model on the training data
        model.train(self.m_training_data)

        # infer spectra for the validation labels
        print("Inferring spectra for the validation labels")
        self.m_validation_result.m_spectra = \
            model.infer_spectra_from_labels(self.m_validation_data.m_label,
                                            1.0 / self.m_validation_data.m_label_ivar)

        # infer spectra for the test labels
        print("Inferring spectra for the test labels")
        self.m_test_results.m_spectra = \
            model.infer_spectra_from_labels(self.m_test_data.m_label,
                                            1.0 / self.m_test_data.m_label_ivar)

        # infer labels for the validation spectra
        print("Inferring labels for the validation spectra")
        self.m_validation_result.m_label = \
            model.infer_labels_from_spectra(self.m_validation_data.m_spectra,
                                            1.0 / self.m_validation_data.m_spectra_ivar)

        # infer labels for the test spectra
        print("Inferring labels for the test spectra")
        self.m_test_results.m_label = \
            model.infer_labels_from_spectra(self.m_test_data.m_spectra,
                                            1.0 / self.m_test_data.m_spectra_ivar)

        self.m_run = True

    def save_task(self,
                  comment=None):
        if not self.m_run:
            warnings.warn("Saving a Task without results.")

        if comment is None:
            pickle.dump(self, open(self.m_default_save_dir + "Task_" + str(self.m_task_id).zfill(2)
                                   + ".pkl", "wb"))
        else:
            pickle.dump(self, open(self.m_default_save_dir + "Task_" + str(self.m_task_id).zfill(2)
                                   + comment + ".pkl", "wb"))

    @classmethod
    def restore_from_file(cls,
                          task_location):
        return pickle.load( open(task_location, "rb"))


class SpectralModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self,
              train_dataset):
        pass

    @abstractmethod
    def infer_labels_from_spectra(self,
                                  fluxes_in,
                                  flux_vars_in):

        # returns labels (can be extended with their uncertainties)
        return 0

    @abstractmethod
    def infer_spectra_from_labels(self,
                                  labels_in,
                                  label_vars_in):
        # returns spectra (can be extended with their uncertainties)
        return 0


class Evaluator(object):

    def __init__(self,
                 task_dir):
        """

        :param task_dir: ends with /
        """

        self.m_task_dir = task_dir

        assert os.path.isdir(self.m_task_dir)

        self.m_tasks_files = [self.m_task_dir + f for f in os.listdir(self.m_task_dir)
                    if os.path.isfile(os.path.join(self.m_task_dir, f))]

        # LOAD AND MERGE ALL TASKS

        # get the size of the Task space by using the first file
        test_task = EvaluationTask.restore_from_file(self.m_tasks_files[0])

        all_ids = np.concatenate([test_task.m_training_splint_ids,
                                  test_task.m_validation_split_ids,
                                  test_task.m_test_split_ids])

        number_of_samples = np.sort(all_ids)[-1] + 1
        assert number_of_samples == all_ids.shape[0], "Splits not consitend"

        # INIT THE RESULT DATASETS
        def create_empty_dataset(number_of_samples_in):
            return SpectralData(test_task.m_training_data.m_wavelength,
                                np.zeros((number_of_samples,
                                          test_task.m_training_data.m_spectra.shape[1])),
                                np.zeros((number_of_samples,
                                          test_task.m_training_data.m_spectra_ivar.shape[1])),
                                np.zeros((number_of_samples,
                                          test_task.m_training_data.m_label.shape[1])),
                                np.zeros((number_of_samples,
                                          test_task.m_training_data.m_label_ivar.shape[1])),
                                label_names=test_task.m_training_data.m_label_names)

        self.m_ground_truth = create_empty_dataset(number_of_samples)
        self.m_validation_result = create_empty_dataset(number_of_samples)
        self.m_test_result = create_empty_dataset(number_of_samples)

        # FILL WITH THE RESULTS
        for tmp_task in self.m_tasks_files:
            tmp_task = EvaluationTask.restore_from_file(tmp_task)

            # set GT
            self.m_ground_truth.m_label[tmp_task.m_test_split_ids] =\
                tmp_task.m_test_data.m_label
            self.m_ground_truth.m_label_ivar[tmp_task.m_test_split_ids] = \
                tmp_task.m_test_data.m_label_ivar
            self.m_ground_truth.m_spectra[tmp_task.m_test_split_ids] = \
                tmp_task.m_test_data.m_spectra
            self.m_ground_truth.m_spectra_ivar[tmp_task.m_test_split_ids] = \
                tmp_task.m_test_data.m_spectra_ivar

            # set validation
            self.m_test_result.m_label[tmp_task.m_test_split_ids] = \
                tmp_task.m_test_results.m_label
            self.m_test_result.m_spectra[tmp_task.m_test_split_ids] = \
                tmp_task.m_test_results.m_spectra

            # set test
            self.m_validation_result.m_label[tmp_task.m_validation_split_ids] = \
                test_task.m_validation_result.m_label
            self.m_validation_result.m_spectra[tmp_task.m_validation_split_ids] = \
                test_task.m_validation_result.m_spectra

        # Placeholder for the results
        self.m_chi2_error_labels_test = None
        self.m_chi2_error_labels_validation = None
        self.m_chi2_error_spectra_test = None
        self.m_chi2_error_spectra_validation = None

    def compute_chi2_error(self):
        self.m_chi2_error_labels_test = \
            (self.m_test_result.m_label - self.m_ground_truth.m_label)**2 \
            * self.m_ground_truth.m_label_ivar

        self.m_chi2_error_labels_validation = \
            (self.m_validation_result.m_label - self.m_ground_truth.m_label)**2 \
            * self.m_ground_truth.m_label_ivar

        self.m_chi2_error_spectra_test = \
            (self.m_test_result.m_spectra - self.m_ground_truth.m_spectra) ** 2 \
            * self.m_ground_truth.m_spectra_ivar

        self.m_chi2_error_spectra_validation = \
            (self.m_validation_result.m_spectra - self.m_ground_truth.m_spectra) ** 2 \
            * self.m_ground_truth.m_spectra_ivar


