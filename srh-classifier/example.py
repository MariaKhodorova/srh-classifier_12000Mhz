from EffnetClassifier import EffnetClassifier
version = 'b0'
model = EffnetClassifier(model_name=f'efficientnet_{version}')
mode = "predict"
data = "12000Mhz" #6000Mhz, 12000Mhz


if mode == "train":
    model.get_efficientnet_dataloaders(data_dir=f"data/{data}", batch_size=32)
    model.train(
        num_epochs_stage1=50,
        num_epochs_stage2=100,
        learning_rate_stage1=1e-3,
        learning_rate_stage2=1e-4,
        unfreeze_depth=3
    )
    model.save_model(f"models/effnet_{version}_{data}.pth")

if mode=="eval":
    model.load_model(f"models/effnet_{version}_{data}.pth")
    model.get_efficientnet_dataloaders(data_dir=f"data/{data}", batch_size=32)
    model.confusion_matrix()

if mode=="predict":
    model.load_model(f"models/effnet_{version}_{data}.pth")
    model.predict("https://ftp.rao.istp.ac.ru/SRH/SRH0306/cleanMaps/20210712/srh_I_2021-07-12T01:59:42_3100.fit")