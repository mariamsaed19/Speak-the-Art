from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from image_synthesis.utils.misc import instantiate_from_config
from tqdm import tqdm
import pickle

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class FlickerDataset(Dataset):
    def __init__(self, data_root,input_file, phase = 'train', im_preprocessor_config=None, drop_caption_rate=0.0):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.image_folder = os.path.join(data_root, 'Images/Flicker8k_Dataset')
        self.root = os.path.join(data_root, phase)
        pickle_path = os.path.join(self.root, "filenames.pickle")
        self.name_list = pickle.load(open(pickle_path, 'rb'), encoding="bytes")

        self.name_list += self.name_list
        #self.name_list = self.name_list[:50]
        self.num = len(self.name_list)
        if phase == "train":
          embedding_filename_ar = os.path.join(data_root, 'speech/speech_embeddings_train_ar.pickle')
          embedding_filename_en = os.path.join(data_root, 'speech/speech_embeddings_train_en.pickle')
        else:
          embedding_filename_ar = os.path.join(data_root, 'speech/speech_embeddings_test_ar.pickle')
          embedding_filename_en = os.path.join(data_root, 'speech/speech_embeddings_test_en.pickle')
        with open(embedding_filename_ar, 'rb') as f:
            e1 = pickle.load(f, encoding="bytes")
        with open(embedding_filename_en, 'rb') as f:
            e2 = pickle.load(f, encoding="bytes")

        self.embeddings = np.vstack((np.array(e1),np.array(e2)))
        print("Total: ",self.embeddings.shape)
            
        self.drop_rate = drop_caption_rate
        self.phase = phase
        '''
        # load all caption file to dict in memory
        self.caption_dict = {}
        
        for index in tqdm(range(self.num)):
            name = self.name_list[index]
            this_text_path = os.path.join(data_root, 'text', 'text', name+'.txt')
            with open(this_text_path, 'r') as f:
                caption = f.readlines()
            self.caption_dict[name] = caption

        print("load caption file done")
        '''

    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        name = self.name_list[index]
        image_path = os.path.join(self.image_folder, name)
        image = load_img(image_path)
        image = np.array(image).astype(np.uint8)
        image = self.transform(image = image)['image']
        embeddings = self.embeddings[index, :, :]
        #print(embeddings.shape)
        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[embedding_ix, :]
        embedding = embedding.reshape((1,1024))
        '''
        caption_list = self.caption_dict[name]
        caption = random.choice(caption_list).replace('\n', '').lower()
        '''
        data = {
                'image': np.transpose(image.astype(np.float32), (2, 0, 1)),
                'speech': embedding,
        }
        
        return data
