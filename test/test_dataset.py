import unittest
import numpy as np
from src.utils import load_yaml, inverse_stft_transform
from src.preprocess.feature_extractor import FeatureExtractor

save_path = "./test/result/test_model"


class DatasetSanityCheck(unittest.TestCase):
    def test_load(self):
        """python -m unittest -v test.test_dataset.DatasetSanityCheck.test_load"""
        from src.distrib import load_dataset

        path_conf = "./conf/config.yaml"
        args = load_yaml(path_conf)

        train_dataset, test_dataset = load_dataset(args)

        for train_data, test_data in zip(train_dataset, test_dataset):
            print(len(train_data), len(test_data))
            print(train_data[0].shape, train_data[1].shape)

    def test_fit_model(self):
        """
        python -m unittest -v test.test_dataset.DatasetSanityCheck.test_fit_model
        """
        from src.distrib import load_dataset, load_model

        path_conf = "./conf/config.yaml"
        args = load_yaml(path_conf)

        train_dataset, test_dataset = load_dataset(args)

        model = load_model(args)
        baseline_val_loss = model.evaluate(test_dataset)[0]

        print(f"Baseline accuracy {baseline_val_loss}")

        model.fit(
            train_dataset,
            steps_per_epoch=args.steps,
            validation_data=test_dataset,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    unittest.main()
