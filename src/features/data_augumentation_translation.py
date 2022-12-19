import nlpaug.augmenter.word as naw
from tqdm import tqdm

def get_augumented_data(data):
    aug = naw.BackTranslationAug(from_model_name='Helsinki-NLP/opus-mt-en-ru',
                                 to_model_name='Helsinki-NLP/opus-mt-ru-en',
                                 name='BackTranslationAug',
                                 batch_size=32,
                                 device='cuda',
                                 force_reload=False,
                                 verbose=0)

    train_formal_aug = []
    train_informal_aug = []
    for i in tqdm(data.shape[0]):
        tmp = aug.augment(data['Formal text'][i])
        train_formal_aug.append(tmp)

        tmp_in = aug.augment(data['Informal text'][i])
        train_informal_aug.append(tmp_in)
    return train_formal_aug, train_informal_aug