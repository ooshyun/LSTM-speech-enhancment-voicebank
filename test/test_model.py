import unittest
import logging
from src.utils import load_yaml

save_path = "./test/result/test_model"

class ModelSanityCheck(unittest.TestCase):
    def test_save(self, args=None, debug=False):
        """python -m unittest -v test.test_model.ModelSanityCheck.test_save"""
        from src.distrib import load_model, load_dataset
        if args is None:
            path_conf = "./test/conf/config.yaml"
            args = load_yaml(path_conf)
        train_dataset, test_dataset = load_dataset(args)
        model = load_model(args)

        model.fit(
            train_dataset,  # model.fit([pair_1, pair_2], labels, epochs=50)
            steps_per_epoch=1,  # you might need to change this
            validation_data=test_dataset,
            epochs=1,
        )

        from src.distrib import save_model_all

        save_model_all(save_path, model)

        if debug:
            return model, save_path

    def test_load(self):
        """python -m unittest -v test.test_model.ModelSanityCheck.test_load"""
        import os
        import shutil
        from src.distrib import load_model

        if os.path.isdir(save_path):
            shutil.rmtree(save_path)

        path_conf = "./test/conf/config.yaml"
        args = load_yaml(path_conf)
        saved_model, model_saved_path = self.test_save(args=args, debug=True)
        args.model.path = model_saved_path
        
        model = load_model(args)

        print(
            "Model parameter length: ",
            len(model.get_weights()),
            len(saved_model.get_weights()),
        )
        print(
            "Optimizer parameter length: ",
            len(model.optimizer.get_weights()),
            len(saved_model.optimizer.get_weights()),
        )

        print(" Model Test")
        for param, save_param in zip(model.get_weights(), saved_model.get_weights()):
            diff = save_param - param
            assert (diff < 1e-6).all()

        print(" Optimizer Test")
        for param, save_param in zip(
            model.optimizer.get_weights(), saved_model.optimizer.get_weights()
        ):
            diff = save_param - param
            assert (diff < 1e-6).all()

    def test_optim_multiple_load(self):
        """
        python -m unittest -v test.test_model.ModelSanityCheck.test_optim_multiple_load
        """
        import os
        import shutil
        from src.distrib import load_model
        
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        
        path_conf = "./test/conf/config.yaml"
        args = load_yaml(path_conf)

        for i in range(2):
            logging.info(f" {i+1} Model Training...")
            saved_model, saved_model_path = self.test_save(args=args, debug=True)
            args.model.path = saved_model_path

            model = load_model(args)

            print(
                "Model parameter length: ",
                # len(model.get_weights()),
                len(saved_model.get_weights()),
            )
            print(
                "Optimizer parameter length: ",
                # len(model.optimizer.get_weights()),
                len(saved_model.optimizer.get_weights()),
            )

            print(" Model Test")
            for param, save_param in zip(model.get_weights(), saved_model.get_weights()):
                diff = save_param - param
                assert (diff < 1e-6).all()

            print(" Optimizer Test")
            for param, save_param in zip(
                model.optimizer.get_weights(), saved_model.optimizer.get_weights()
            ):
                diff = save_param - param
                assert (diff < 1e-6).all()

            print(f" {i+1} Pass")

if __name__ == "__main__":
    unittest.main()
