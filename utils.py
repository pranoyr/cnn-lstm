import os 

# loading partition and labels
def load_data(dir_path):
    labels={}
    partition={}
    
    train_list=[]
    data=open(os.path.join(dir_path,'train.txt'),'r').read()
    c=data.split('\n')
    for i in c[:len(c)-1]:
        labels.update({i.split(' ')[0]:int(i.split(' ')[1])})
        train_list.append(i.split(' ')[0])

    val_list=[]
    data=open(os.path.join(dir_path,'val.txt'),'r').read()
    c=data.split('\n')
    for i in c[:len(c)-1]:
        labels.update({i.split(' ')[0]:int(i.split(' ')[1])})
        val_list.append(i.split(' ')[0])

    partition['train']=train_list
    partition['val']=val_list
    return partition,labels


# converting labels to integers
def convert_to_integer(path):
    dict_labels={}
    a=open(path,'r').read()
    c=a.split('\n')
    for i in c[:len(c)-1]:
        dict_labels.update({i.split(' ')[1]:i.split(' ')[0]})
    return dict_labels

