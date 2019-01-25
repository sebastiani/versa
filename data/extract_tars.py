import os, sys, tarfile, pickle, multiprocessing
from multiprocessing.pool import ThreadPool


def untar(filename):
    print('extracting {}'.format(filename))
    dirname = filename.split('.')[0]
    print("making dir {}".format(dirname))
    os.makedirs(dirname)
    tar = tarfile.open(filename)
    tar.extractall(path=dirname)
    tar.close()


if __name__ == '__main__':
    pool = ThreadPool(multiprocessing.cpu_count())

    #directories = [d for d in os.listdir("/media/akasha/My Passport/imagenet/") if os.path.isdir(d)]
    tars = [tar for tar in os.listdir('.') if '.tar' in tar]
    #tars2extract = [tar for tar in tars if not tar in directories]
    results = []
    
    for tar in tars:
        results.append(pool.apply_async(untar, (tar,)))
        
    pool.close()
    pool.join()
                       
