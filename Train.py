# create folder and download images
folder ="black"
file = 'urls_black.csv'

folder = 'teddys'
file = 'urls_teddys.csv'

folder = 'grizzly'
file = 'urls_grizzly.csv'

path = Path('data/bears')
dest = path/folder
dest.mkdir(parents =True, exist_ok=True)

classes = ['teddys','grizzly','black']

#repeat this for all categories
download_images("/content/data/bears/grizzly/grizzly.csv", "/content/data/bears/grizzly/", max_pics=200)

# remove images which cannot be opened
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)

np.random.seed(3)
data = ImageDataBunch.from_folder(path,train=".", valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats) 
data.show_batch(3, figsize= (7,10))
learn = cnn_learner(data, models.resnet50, metrics= error_rate)
learn.fit_one_cycle(4)
# epoch	train_loss	valid_loss	error_rate	time
# 0	0.915136	0.380371	0.138889	00:13
# 1	0.481794	0.274809	0.055556	00:07
# 2	0.324674	0.116859	0.013889	00:08
# 3	0.242618	0.079834	0.013889	00:07

learn.save("teddy_resnet50_v1")
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-5, 1e-4))

# epoch	train_loss	valid_loss	error_rate	time
# 0	0.097840	0.061098	0.013889	00:07
# 1	0.051848	0.051780	0.013889	00:07
# 2	0.048494	0.052547	0.013889	00:07
# 3	0.049926	0.054957	0.013889	00:07

learn.save("model_teddy_resnet50_v2")
img_visual = ClassificationInterpretation.from_learner(learn)
img_visual.plot_confusion_matrix()

# we can clean images and retrain 
# ds, idxs = DatasetFormatter().from_similars(learn_cln)
# ImageCleaner(ds, idxs, path, duplicates=True)
# ds, idxs = DatasetFormatter().from_toplosses(learn_cln)
# ImageCleaner(ds, idxs, path)
# new cleaned.csv is created need to create databunch from this csv

# db = (ImageList.from_csv(path, 'cleaned.csv', folder='.')
#                    .split_none()
#                    .label_from_df()
#                    .transform(get_transforms(), size=224)
#                    .databunch()
#      )

# learn_cln = cnn_learner(db, models.resnet50, metrics=error_rate)

# save model file to put in production
learn.export() # this will create pickel file in same path as export.pkl.

#Now to check model perfromance
learn = load_learner(path)
pred_class,pred_idx,outputs = learn.predict(img) # for any dummy image we can check its class
