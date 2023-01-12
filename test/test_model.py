import unittest
import logging
import numpy as np
from src.utils import load_yaml
from keras.backend import random_normal
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

    def test_lstm(self):
        """
        python -m unittest -v test.test_model.ModelSanityCheck.test_lstm
        """
        from src.utils import load_yaml
        from src.model.rnn import build_model_rnn

        config = load_yaml("./test/conf/config.yaml")
        batch = config.batch_size
        channel = config.dset.channels
        segment = config.dset.segment
        nfeature = config.dset.n_fft//2+1
        
        inputs = [random_normal(shape=(batch, channel, config.model.n_segment, config.model.n_feature)) for _ in range(3)]
        inputs_complex = np.array(inputs, dtype=np.complex64)
        inputs_complex.imag = inputs

        model = build_model_rnn(config)

        model.build(input_shape=(channel, config.model.n_segment, config.model.n_feature))
        model.summary()
        
        print("Real Training...")
        for step, input in enumerate(inputs):
            print(f"input shape: {input.shape}")
            output = model(input)
            print(f"Step {step}: Input shape={input.shape}, Output shape: {output.shape}")      
            break

        print("Complex Training...")
        for step, input in enumerate(inputs_complex):
            print(f"input shape: {input.shape}")
            output = model(input)
            print(f"Step {step}: Input shape={input.shape}, Output shape: {output.shape}")      
            break

    def test_crn(self):
        """
        python -m unittest -v test.test_model.ModelSanityCheck.test_crn
        """
        from src.utils import load_yaml
        from src.model.crn import build_crn_model_tf

        config = load_yaml("./test/conf/config.yaml")
        batch = config.batch_size
        channel = config.dset.channels
        segment = config.dset.segment
        nfeature = config.dset.n_fft//2+1
        
        inputs = [random_normal(shape=(batch, channel, config.model.n_segment, config.model.n_feature)) for _ in range(3)]
        inputs_complex = np.array(inputs, dtype=np.complex64)
        inputs_complex.imag = inputs

        model = build_crn_model_tf(config)

        model.build(input_shape=(channel, config.model.n_segment, config.model.n_feature))
        model.summary()
        
        print("Real Training...")
        for step, input in enumerate(inputs):
            print(f"input shape: {input.shape}")
            output = model(input)
            print(f"Step {step}: Input shape={input.shape}, Output shape: {output.shape}")      
            break

        print("Complex Training...")
        for step, input in enumerate(inputs_complex):
            print(f"input shape: {input.shape}")
            output = model(input)
            print(f"Step {step}: Input shape={input.shape}, Output shape: {output.shape}")      
            break

    def test_unet(self):
        """
        python -m unittest -v test.test_model.ModelSanityCheck.test_unet
        """
        from src.utils import load_yaml
        from src.model.unet import build_unet_model_tf

        config = load_yaml("./test/conf/config.yaml")
        batch = config.batch_size
        channel = config.dset.channels
        sample_rate = config.dset.sample_rate
        segment = config.dset.segment

        inputs = [random_normal(shape=(batch, channel, int(sample_rate*segment))) for _ in range(3)]
        
        model = build_unet_model_tf(config)

        model.build(input_shape=(batch, channel, int(sample_rate*segment)))
        model.summary()
        
        for step, input in enumerate(inputs):
            print(f"input shape: {input.shape}")
            output = model(input)
            print(f"Step {step}: Input shape={input.shape}, Output shape: {output.shape}")      
            break

    def test_conv_tasnet(self):
        """
        python -m unittest -v test.test_model.ModelSanityCheck.test_conv_tasnet
        """
        from src.utils import load_yaml
        from src.model.conv_tasnet import build_conv_tasnet_model_tf

        config = load_yaml("./test/conf/config.yaml")
        batch = config.batch_size
        channel = config.dset.channels
        sample_rate = config.dset.sample_rate
        segment = config.dset.segment

        inputs = [random_normal(shape=(batch, channel, int(sample_rate*segment))) for _ in range(3)]
        
        model = build_conv_tasnet_model_tf(config)

        model.build(input_shape=(batch, channel, int(sample_rate*segment)))
        model.summary()
        
        for step, input in enumerate(inputs):
            print(f"input shape: {input.shape}")
            output = model(input)
            print(f"Step {step}: Input shape={input.shape}, Output shape: {output.shape}")      
            break

if __name__ == "__main__":
    unittest.main()
