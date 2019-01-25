import numpy as np
import csv
import glob
import os
import pickle
import multiprocessing
from multiprocessing.pool import ThreadPool
from PIL import Image
import pandas as pd
import shutil

def process_images():
    
    #all_images = glob.glob(base_path + 'images/*')

    def resize(arg):
        dir, image_file = arg
        im = Image.open(image_file)
        im = im.resize((84, 84), resample=Image.LANCZOS)
        image_name = image_file.split('/')[-1]
        print("saving ", "miniImageNet/"+ dir+'/'+image_name)
        im.save("miniImageNet/"+ dir+'/'+image_name)

    print("Resizing images to 84 x 84 pixels.")
    results = []
    pool = ThreadPool(multiprocessing.cpu_count())
    all_dirs = [x for x in os.listdir('tars/') if os.path.isdir('tars/'+x)]

    for i, dir in enumerate(all_dirs):
        images = glob.glob('tars/'+dir+'/*')

        if not os.path.join('miniImageNet', dir):
            os.makedirs(os.path.join('miniImageNet', dir))
        for image_file in images:
            arg = (dir, image_file)
            results.append(pool.apply_async(resize, (arg,)))    
    pool.close()
    pool.join()
    print("Resizing complete.")

def moveImages():
    base_path = 'images/'
    
    print("Moving images to train, test and val")
    if not os.path.isdir('train'):
        os.makedirs('train')
    if not os.path.isdir('test'):
        os.makedirs('test')
    if not os.path.isdir('val'):
        os.makedirs('val')

    for datatype in ['train', 'val', 'test']:
        print("Moving {0:} images.".format(datatype))
        
        with open(datatype + '.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                label = row[1]
                image_name = row[0]
                if not os.path.isdir(os.path.join(datatype,label)):
                    os.makedirs(os.path.join(datatype, label))

                image_num = image_name.split('.')[0]
                image_num = image_num.split('0')[-1]
                real_image_name = label + '_' + image_num + '.JPEG'
                src = os.path.join(base_path, real_image_name)
                dest = os.path.join(datatype, label, real_image_name)
                os.rename(src, dest)

    print("Finished moving images")

def resample():
    base_path = 'miniImageNet/'
        
    df1 = pd.read_csv("train.csv")
    df2 = pd.read_csv("test.csv")
    df3 = pd.read_csv("val.csv")
    df = pd.concat([df1, df2, df3])
    
    classes = df.label.unique()
    for cls in classes:        
        imgs = np.array(os.listdir(os.path.join(base_path, cls)))
        total_im = len(imgs)
        idx = np.random.randint(0, total_im, 600)
        imgs = imgs[idx]
        if not os.path.isdir("pool/"+cls):
            os.makedirs("pool/"+cls)

        for img in imgs:
            src = os.path.join(base_path, cls, img)
            print(img)
            dst = os.path.join("pool", cls, img)
            shutil.copyfile(src, dst)

    print("Finished resampling images.")

def moving():
    print("Moving images to train, test and val")
    if not os.path.isdir('train'):
        os.makedirs('train')
    if not os.path.isdir('test'):
        os.makedirs('test')
    if not os.path.isdir('val'):
        os.makedirs('val')

    for datatype in ['train','test', 'val']:
        print("Forming "+datatype)
        df = pd.read_csv(datatype +'.csv')
        classes = df.label.unique()
        for cls in classes:
            dst = os.path.join(datatype, cls)
            if not os.path.isdir(dst):
                os.makedirs(dst)
            imgs = glob.glob("pool/"+cls+"/*")
            for img in imgs:
                print(img)
                src = os.path.join(img)
                name = img.split('/')[-1]
                print(name)
                dst = os.path.join(datatype, cls, name)
                print(dst)
                shutil.copyfile(src, dst)

    print("Finished moving")

def makeCSV():
    datatypes = ['train', 'test', 'val']
    
    for datatype in datatypes:
        print("Fixing ", datatype)
        samples = []
        labels = []
        classes = os.listdir(datatype)
        for cls in classes:
            path = os.path.join(datatype, cls)
            imgs = [x for x in os.listdir(path) if '.JPEG' in x]
            label = [cls for i in range(len(imgs))]
            labels = labels + label
            samples = samples + imgs
        df = pd.DataFrame()
        df['filename'] = samples
        df['label'] = labels
        print(df.head())
        df1 = df.sample(frac=1.0)
        df1.to_csv('fixed_'+datatype+'.csv', index=False)
        
    print("Finished generating fixed csv")
        
def save_file(data_name):
    dir_list = [os.path.join(data_name, x) for x in os.listdir(data_name)]

    output = np.zeros((len(dir_list), 600, 84, 84, 3), dtype=np.uint8)
    for i, dir in enumerate(dir_list):
        out = np.zeros((600, 84, 84, 3), dtype=np.uint8)
        im_files = glob.glob(os.path.join(dir, "*.JPEG"))
        if len(im_files) != 600:
            print("Folder: {0:} should have 600 images in it and it has {1:d}"
                  .format(os.path.join(data_name, dir), len(im_files)))

        for j, im_file in enumerate(im_files):
            print(im_file)
            im = Image.open(im_file).convert('RGB')
            out[j] = im

        output[i] = out

    pickle.dump(output, open("mini_imagenet_"+data_name+".pkl", "wb"), protocol=2)


if __name__ == "__main__":
    #process_images()
    #moveImages()
    #resample()
    #moving()
    #makeCSV()
    save_file('train')
    save_file('test')
    save_file('val')
    print("Finished pickling images.")
