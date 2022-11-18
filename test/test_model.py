import unittest
from src.utils import load_yaml

save_path = "./test/result/test_model"

class ModelSanityCheck(unittest.TestCase):
    def test_save(self, debug=False):
        """python -m unittest -v test.test_model.ModelSanityCheck.test_save
        """
        from src.distrib import load_model, load_dataset
        
        path_conf = "./test/conf/config_init.yaml"
        args = load_yaml(path_conf)
        train_dataset, test_dataset = load_dataset(args)
        model = load_model(args)

    
        model.fit(train_dataset, # model.fit([pair_1, pair_2], labels, epochs=50)
            steps_per_epoch=1, # you might need to change this
            validation_data=test_dataset,
            epochs=1,
            )
        
        from src.distrib import save_model_all
        save_model_all(save_path, model) 

        if debug:
            return model

    def test_load(self):
        """python -m unittest -v test.test_model.ModelSanityCheck.test_load
        """
        import shutil
        from src.distrib import load_model
        
        shutil.rmtree(save_path)
        saved_model = self.test_save(debug=True)
        path_conf = "./test/conf/config.yaml"

        args = load_yaml(path_conf)
        model = load_model(args)

        print("Model parameter length: ", len(model.get_weights()), len(saved_model.get_weights()))
        print("Optimizer parameter length: ", len(model.optimizer.get_weights()), len(saved_model.optimizer.get_weights()))

        print(" Model Test")
        for param, save_param in zip(model.get_weights(), saved_model.get_weights()):
            diff  = save_param-param
            assert (diff < 1e-6).all()
        
        print(" Optimizer Test")
        for param, save_param in zip(model.optimizer.get_weights(), saved_model.optimizer.get_weights()):
            diff  = save_param-param
            assert (diff < 1e-6).all()
            
if __name__=="__main__":
    unittest.main() 
