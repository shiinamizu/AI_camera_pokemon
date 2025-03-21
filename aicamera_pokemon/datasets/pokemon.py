from torch.utils.data import Datasets
import os
import cv2

class Pokemon(Datasets):



    def __init__(self,path,train):
        self.images = []
        self.labels = []
        self.index_map =[]
        self.label_names=[]

        label_names = os.listdir(path)
        self.get_labels_name(label_names)

        for i,label in enumerate(self.label_names):

            label_path = path + '/' + label
            for img_name in os.listdir(label_path):
                img_path = label_path+'/'+img_name
                img = cv2.imread(img_path)

                self.images.append(img)
                self.labels.append(i)




    
    def get_labels_name(self,names):
        for n in names:
            self.label_names.append(n)

    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self,idx):
        index,t = self.index_map[idx]
        img = self.images[index]
        label = self.labels[index]


        return img, label, index




