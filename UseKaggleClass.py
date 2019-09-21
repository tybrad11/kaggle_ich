from KaggleModelClass import PneumothoraxModel

KaggleObject = PneumothoraxModel(test_path = '/data/Kaggle/test-png',
                                 dims = (1024,1024),
                                 batch_size = 8,
                                 val_split = .15,
                                 optimizer='Adam',
                                 loss='dice',
                                 multi_process=False)
KaggleObject.init_model(weights='Best_Pneumothorax_Model_Weights_512.h5')
KaggleObject.train(epochs=10)
KaggleObject.generate_submission()

