import json
import logging

logger = logging.getLogger(__name__)
import os
import re
from collections import defaultdict
from typing import List
import pickle
import clip

from .base_dataset import BaseDataset 


class CUBDatasetEN(BaseDataset):
    print("Hello from **data/flickr_stat/filckr_dataset** class FlickrDataSet ")
    def __init__(
        self,
        dataset_root: str,
        text_file: str,
        modalities: List,
        split: str = "train",
        image_transform=None,
        audio_transform=None,
        target_sr: int = 16_000,
        load_audio: bool = True,
        load_image: bool = False,
        tokenizeText: bool = False,
        wav_rm_silence: bool = False,
        clip_image_transform: str = None,
        **kwargs,
    ):
        if clip_image_transform is not None:
            logger.info(
                "Load clip ({}) for image transform".format(clip_image_transform)
            )
            _, image_transform = clip.load(clip_image_transform, "cpu")

        super().__init__(
            dataset_root=dataset_root,
            split=split,
            image_transform=image_transform,
            audio_transform=audio_transform,
            target_sr=target_sr,
            load_audio=load_audio,
            load_image=load_image,
            tokenizeText=tokenizeText,
            **kwargs,
        )

        assert len(modalities) > 0, "Dataset's modalities cannot be none"
        self.modalities = modalities

        image_list_txt = os.path.join(
            self.dataset_root, f"CUB.{self.split}Images.txt"
        )

        if wav_rm_silence:
            print("Using wav w/o silence data")
            wav_file = "wavs_with_no_silence"
        else:
            wav_file = "wavs"

        wav_file_en ="org_audio" 
        #wav_file_ar ="audio_arabic"
        wav_base_path_en = os.path.join(self.dataset_root, wav_file_en)
        #wav_base_path_ar = os.path.join(self.dataset_root, wav_file_ar)

        #wav_list_en = os.listdir(wav_base_path_en)
        #wav_list_ar = os.listdir(wav_base_path_ar)
        with open (os.path.join(self.dataset_root,"english_audio_names.pickle"),'rb') as pick:
          wav_list_en = pickle.load(pick)

        #with open (os.path.join(self.dataset_root,"arabic_audio_names.pickle"),'rb') as pick:
         # wav_list_ar = pickle.load(pick)

        
        wav_names_list = []
        for p in wav_list_en :
          if p.split(".")[-1] in ["wav","mp3"]  :
            tobeRemoved = p.split("_")[-1] 
            name = p.replace("_"+tobeRemoved,"")
            wav_names_list.append(name)

        wav_names = {e for e in wav_names_list}

        wav_names_to_paths = defaultdict(list)
        for p in wav_list_en:
            tobeRemoved = p.split("_")[-1] 
            name = p.replace("_"+tobeRemoved,"")
            if name in wav_names:
                wav_names_to_paths[name].append(os.path.join(wav_base_path_en, p))
        '''
        for p in wav_list_ar:
          tobeRemoved = p.split("_")[-1] 
          name = p.replace("_"+tobeRemoved,"")
          if name in wav_names:
              wav_names_to_paths[name].append(os.path.join(wav_base_path_ar, p))
        '''
        


        assert text_file in [
            "captions.txt",
            "CUB.lemma.token.txt",
            "CUB.token.txt",
        ], "CUB text file must be one of them {}".format(
            ["captions.txt", "CUB.lemma.token.txt", "CUB.token.txt"]
        )
        caption_txt_path = os.path.join(self.dataset_root, text_file)
        imageName2captions = {}
        '''
        if text_file == "captions.txt":
            with open(caption_txt_path, "r") as f:
                for _l in f.readlines():
                    # skip first line
                    if _l.strip() == "image,caption":
                        continue

                    _imgName, _caption = _l.split(".jpg,")
                    assert isinstance(_imgName, str)
                    assert isinstance(_caption, str)
                    _caption = _caption.lower().strip()
                    if _caption[-1] == ".":
                        _caption = _caption[:-1]
                        _caption = _caption.strip()
                    if _imgName not in imageName2captions:
                        imageName2captions[_imgName] = []
                    imageName2captions[_imgName].append(_caption)
        else:
            print("125 :",caption_txt_path)
            with open(caption_txt_path, "r") as f:
                for i, _line in enumerate(f.readlines()):
                    _line = _line.strip()
                    _out = re.split("#[0-9]+", _line)
                    assert len(_out) == 2, _line
                    _imgName, _caption = re.split("#[0-9]+", _line)
                    _imgName = _imgName.replace(".jpg", "")
                    _caption = _caption.strip()
                    if _caption[-1] == ".":
                        _caption = _caption[:-1].strip()

                    if _imgName not in imageName2captions:
                        imageName2captions[_imgName] = []
                    imageName2captions[_imgName].append(_caption)
        '''
        id_pairs_path = os.path.join(self.dataset_root, "CUB_idPairs.json")
        with open(id_pairs_path, "r") as f:
            _data = json.load(f)
            id2Filename = _data["id2Filename"]
            filename2Id = _data["filename2Id"]
        #print("146: ",image_list_txt)
        #print("147: ",wav_names)
        with open(image_list_txt, "r") as fp:
            for line in fp:
                line = line.strip()
                if line == "":
                    continue

                image_name = line.split(".jpg")[0]  # removed ".jpg"
            
                audio_name = image_name+'/'+image_name.split('/')[-1]
                image_path = os.path.join(dataset_root, "images", line)
                #print('line ',line,' img ',image_name)
                if audio_name in wav_names:
                  #print("158: ",image_name)
                  # notFoundImages = ['3242919570_39a05aa2ee','3286193613_fc046e8016','3286111436_891ae7dab9','3286045254_696c6b15bd']
                  # print("caaaaaaase",filename2Id.get('3242919570_39a05aa2ee')== None)

                  if filename2Id.get(image_name)is not None :
                   # print("168")
                    if "audio" in self.modalities or "text" in self.modalities:
                        for p in wav_names_to_paths[audio_name]:
                            _entry = {"id": filename2Id[image_name]}

                            if "txt" in os.path.basename(p).split("_")[-1].replace(
                                ".wav", ""
                            ).replace(".mp3",""):
                                continue

                            _subID = int(
                                os.path.basename(p).split("_")[-1].replace(".wav", "").replace(".mp3","")
                            )

                            if "audio" in self.modalities:
                                _entry["wav"] = p
                            if "image" in self.modalities:
                                _entry["image"] = image_path
                            if "text" in self.modalities:
                                _entry["text"] = imageName2captions[image_name][_subID]
                            self.data.append(_entry)
                    else:
                        self.data.append(
                            {
                                "image": image_path,
                                "id": filename2Id[image_name],
                            }
                        )

        logger.info(f"CUB ({self.split}): {len(self.data)} samples")
