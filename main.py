import argparse

def _main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
      '--mode',
      help='Mode for procedure of ML model',
      required=False,
      default='train')

    parser.add_argument(
        '--config',
      help='Configuration for ML model',
      required=False,
      default="./conf/config.yaml")  

    parser.add_argument(
        '--gpusize',
      help='Mode for procedure of ML model',
      required=False,
      default=768)  

    args = parser.parse_args()
    args.gpusize = int(args.gpusize)
    
    if args.mode == "preprocess":
        from src.create_dataset import main
    elif args.mode == "train":
        from src.train import main
    elif args.mode == "inference":
        from src.inference import main
    elif args.mode == "tflite":
        from src.convert_tflite import main
    else:
        raise ValueError(f"Mode is validable (preprocess, train, inference, tflite)")

    main(args.gpusize, args.config)

if __name__ == "__main__":
    _main()