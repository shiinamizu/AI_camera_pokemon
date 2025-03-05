import os
import cv2


class DataSizes:

    def __init__(self):
        self.xsize=0
        self.ysize=1
        self.colorsize=2

        self.img_imfo=0
        self.count=1
        self.data_size_list=[]
    
    def compare_size(self,base_size,target_size):
     
        if(base_size[self.img_imfo][self.xsize]==target_size[self.xsize]
            and base_size[self.img_imfo][self.ysize]==target_size[self.ysize] ):

            return True
        else:
            return False
    
    def addition(self,data):

        input =[data, 0]
        self.data_size_list.append(input)
    
    def update(self,data_size):
        if len(self.data_size_list)!=0:
            
            for i,size in enumerate(self.data_size_list):
                if(self.compare_size(size,data_size)):
                    self.data_size_list[i][self.count]+=1
                    return
            
        
        self.addition(data_size)


            
        

def showIMG(image):
    cv2.imshow('window',image)
    cv2.waitKey()


def all_images_serch(dir):
    dirList = os.listdir(dir)
    ds =DataSizes()
    for file in dirList:
        path = dir + file

        img = cv2.imread(path)
        shape = img.shape
        # print(shape)
        ds.update(shape)

    print(ds.data_size_list)







def main():

    imgDirPath ="datasets/images/"

    

    # img = cv2.imread(imgDirPath+dirList[0])

    # showIMG(img)
    all_images_serch(imgDirPath)
   


if __name__ =='__main__':
    main()

