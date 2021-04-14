import os
import tqdm
import shutil
import pickle
import hashlib


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf8'))
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


if __name__ == '__main__':
    for part in ['all_train.txt', 'all_val.txt', 'all_test.txt']:
        url_file = 'C:/ProjectData/cnn-dailymail-master/url_lists/%s' % part
        url_list = read_text_file(url_file)
        url_hashes = get_url_hashes(url_list)

        pickle.dump(obj=url_hashes, file=open(part.replace('txt', 'pkl'), 'wb'))

    load_path = 'C:/ProjectData/Pretreatment/Step0_OriginData'
    save_path = 'C:/ProjectData/Pretreatment/Step1_SeparatePart'
    if not os.path.exists(save_path):
        os.makedirs(save_path + r'/train')
        os.makedirs(save_path + r'/val')
        os.makedirs(save_path + r'/test')

    train_sha = pickle.load(open('all_train.pkl', 'rb'))
    val_sha = pickle.load(open('all_val.pkl', 'rb'))
    test_sha = pickle.load(open('all_test.pkl', 'rb'))

    for part_name in ['cnn', 'dailymail']:
        for file_name in tqdm.tqdm(os.listdir(os.path.join(load_path, part_name, 'stories'))):
            if file_name.replace('.story', '') in train_sha:
                shutil.copy(os.path.join(load_path, part_name, 'stories', file_name),
                            os.path.join(save_path, 'train', file_name))
            if file_name.replace('.story', '') in val_sha:
                shutil.copy(os.path.join(load_path, part_name, 'stories', file_name),
                            os.path.join(save_path, 'val', file_name))
            if file_name.replace('.story', '') in test_sha:
                shutil.copy(os.path.join(load_path, part_name, 'stories', file_name),
                            os.path.join(save_path, 'test', file_name))
