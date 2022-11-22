import unittest
from src.utils import load_yaml

save_path = "./test/result/test_model"

class DatasetSanityCheck(unittest.TestCase):
    def test_load(self):
        """python -m unittest -v test.test_dataset.DatasetSanityCheck.test_load
        """
        from src.distrib import load_dataset
        path_conf = "./conf/config.yaml"
        args = load_yaml(path_conf)

        train_dataset, test_dataset = load_dataset(args)

        for train_data, test_data in zip(train_dataset, test_dataset):
            print(test_data[-1][-1])
            break

    def test_fit_model(self):
        """python -m unittest -v test.test_dataset.DatasetSanityCheck.test_load
            When doing regenerating speech, this needs mean and std metadata, but in model.fit function in tensorflow, 
            they cannot pass the file name when compute loss function and metric.

            If we need to recover the original signal, it should not use fit function

            fit -> epoch, iter_dataset -> model.train_step -> compute_loss -> 
            [V not pass string object] y_true, y_pred shape check -> loss -> optimizer -> computer metrics
        """
        from src.distrib import load_dataset, load_model
        path_conf = "./conf/config.yaml"
        args = load_yaml(path_conf)

        train_dataset, test_dataset = load_dataset(args)

        model = load_model(args)
        baseline_val_loss = model.evaluate(test_dataset)[0]

        print(f"Baseline accuracy {baseline_val_loss}")

        model.fit(train_dataset,
                steps_per_epoch=args.steps,
                validation_data=test_dataset,
                epochs=args.epochs,
                )


if __name__=="__main__":
    unittest.main() 
