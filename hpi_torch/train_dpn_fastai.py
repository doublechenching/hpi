from utils import init_env
init_env('7')
from hpi import *
warnings.filterwarnings('ignore')
train_val_names = list({f[:36] for f in os.listdir(cfg.train_dir)})
test_names = list({f[:36] for f in os.listdir(cfg.test_dir)})
train_names, val_names = train_test_split(train_val_names, test_size=0.1, random_state=42)
batch_size = cfg.batch_size
target_size = 512
img_ds = get_data(train_names, val_names, test_names, target_size, batch_size, n_workers=5)
learner = ConvLearner.pretrained(dpn92, img_ds, ps=[0.5])  # use dropout 50%
learner.opt_fn = optim.Adam
learner.clip = 1.0
learner.crit = FocalLoss()
learner.metrics = [acc]
print(learner.summary)
lr = 5e-4
learner.fit(lr, 1)
learner.unfreeze()
lrs = np.array([lr/10, lr/3, lr])
learner.fit(lrs/4, 4, cycle_len=2, use_clr=(10, 20), best_save_name='best_dpn_s1')
learner.fit(lrs/4, 2, cycle_len=4, use_clr=(10, 20), best_save_name='best_dpn_s2')
learner.fit(lrs/8, 1, cycle_len=8, use_clr=(5, 20), best_save_name='best_dpn_s3')
val_th = get_val_threshold(learner)
# TTA
preds_t, y_t = learner.TTA(n_aug=4, is_test=True)
preds_t = np.stack(preds_t, axis=-1)
preds_t = sigmoid_np(preds_t)
pred_t = preds_t.max(axis=-1)  # max works better for F1 macro score
test_names = learner.data.test_ds.fnames
save_pred(pred_t, test_names, val_th, 'protein_classification_v.csv')

man_th = np.array([0.565, 0.39, 0.55, 0.345, 0.33, 0.39, 0.33, 0.45, 0.38, 0.39,
                   0.34, 0.42, 0.31, 0.38, 0.49, 0.50, 0.38, 0.43, 0.46, 0.40,
                   0.39, 0.505, 0.37, 0.47, 0.41, 0.545, 0.32, 0.1])
print('Fractions: ', (pred_t > man_th).mean(axis=0))
save_pred(pred_t, test_names, man_th, 'protein_classification.csv')
lb_prob = [0.362397820, 0.043841336, 0.075268817, 0.059322034, 0.075268817,
           0.075268817, 0.043841336, 0.075268817, 0.010000000, 0.010000000,
           0.010000000, 0.043841336, 0.043841336, 0.014198783, 0.043841336,
           0.010000000, 0.028806584, 0.014198783, 0.028806584, 0.059322034,
           0.010000000, 0.126126126, 0.028806584, 0.075268817, 0.010000000,
           0.222493880, 0.028806584, 0.010000000]
test_th = get_test_threshold(pred_t, lb_prob, min_th=0.1)
save_pred(pred_t, test_names, test_th, 'protein_classification_f.csv')

save_pred(pred_t, test_names, 0.5, 'protein_classification_05.csv')

label_count, label_fraction = get_dataset_fraction(pd.read_csv(cfg.train_csv).set_index('Id'))
train_th = get_test_threshold(pred_t, label_fraction, min_th=0.05)
save_pred(pred_t, test_names, train_th, 'protein_classification_t.csv')

brute_th = get_brute_threshold(pred_t)
save_pred(pred_t, test_names, brute_th, 'protein_classification_b.csv')