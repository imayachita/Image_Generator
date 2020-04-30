from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from augment import ImageAugment

class Generator():
    def __init__(self, feat, labels, width, height):
        self.feat = feat
        self.labels = labels
        self.width = width
        self.height = height

    def gen(self):
        '''
        Yields generator object for training or evaluation without batching
        Yields:
            im: np.array of (1,width,height,1) of images
            label: np.array of one-hot vector of label (1,num_labels)
        '''
        feat = self.feat
        labels = self.labels
        width = self.width
        height = self.height
        i=0
        while (True):
            im = cv2.imread(feat[i],0)
            im = im.reshape(width,height,1)
            im = np.expand_dims(im,axis=0)
            label = np.expand_dims(labels[i],axis=0)
            yield im,label
            i+=1

            if i>=len(feat):
                i=0


    def gen_test(self):
        '''
        Yields generator object to do prediction
        Yields:
            im: np.array of (1,width,height,1) of images
        '''
        feat = self.feat
        width = self.width
        height = self.height
        i=0
        while (True):
            im = cv2.imread(feat[i],0)
            im = im.reshape(width,height,1)
            im = np.expand_dims(im,axis=0)
            yield im
            i+=1


    def gen_batching(self, batch_size):
        '''
        Yields generator object with batching of batch_size
        Args:
            batch_size (int): batch_size
        Yields:
            feat_batch: np.array of (batch_size,width,height,1) of images
            label_batch: np.array of (batch_size,num_labels)
        '''
        feat = self.feat
        labels = self.labels
        width = self.width
        height = self.height
        num_examples = len(feat)
        num_batch = num_examples/batch_size
        X = []
        for n in range(num_examples):
            im = cv2.imread(feat[n],0)
            try:
                im = im.reshape(width,height,1)
            except:
                print('Error on this image: ', feat[n])
            X.append(im)
        X = np.array(X)

        feat_batch = np.zeros((batch_size,width,height,1))
        label_batch = np.zeros((batch_size,labels.shape[1]))
        while(True):
            for i in range(batch_size):
                index = np.random.randint(X.shape[0],size=1)[0] #shuffle the data
                feat_batch[i] = X[index]
                label_batch[i] = labels[index]
            yield feat_batch,label_batch

    # def on_next(self):
    #     '''
    #     Advance to the next generator object
    #     '''
    #     gen_obj = self.gen_test()
    #     return next(gen_obj)
    #
    # def gen_show(self, pred):
    #     '''
    #     Show the image generator object
    #     '''
    #     i=0
    #     while(True):
    #         image = self.on_next()
    #         image = np.squeeze(image,axis=0)
    #         cv2.imshow('image', image)
    #         cv2.waitKey(0)
    #         i+=1

    def gen_augment(self,batch_size,augment):
        '''
        Yields generator object with batching of batch_size and augmentation.
        The number of examples for 1 batch will be multiplied based on the number of augmentation

        augment represents [speckle, gaussian, poisson]. It means, the augmentation will be done on the augment list element that is 1
        for example, augment = [1,1,0] corresponds to adding speckle noise and gaussian noise
        if batch_size = 100, the number of examples in each batch will become 300

        Args:
            batch_size (int): batch_size
            augment (list): list that defines what kind of augmentation we want to do
        Yields:
            feat_batch: np.array of (batch_size*n_augment,width,height,1) of images
            label_batch: np.array of (batch_size*n_augment,num_labels)
        '''
        feat = self.feat
        labels = self.labels
        width = self.width
        height = self.height

        num_examples = len(feat)
        num_batch = num_examples/batch_size
        X = []
        for n in range(num_examples):
            im = cv2.imread(feat[n],0)
            try:
                im = im.reshape(width,height,1)
            except:
                print('Error on this image: ', feat[n])
            X.append(im)
        X = np.array(X)

        n_augment = augment.count(1)
        print('Number of augmentations: ', n_augment)
        feat_batch = np.zeros(((n_augment+1)*batch_size,width,height,1))
        label_batch = np.zeros(((n_augment+1)*batch_size,labels.shape[1]))

        while(True):
            i=0
            while (i<=batch_size):
                index = np.random.randint(X.shape[0],size=1)[0] #shuffle the data
                aug = ImageAugment(X[index])
                feat_batch[i] = X[index]
                label_batch[i] = labels[index]

                j=0
                if augment[0] == 1:
                    feat_batch[(j*n_augment)+i+batch_size] = aug.add_speckle_noise()
                    label_batch[(j*n_augment)+i+batch_size] = labels[index]
                    j+=1

                if augment[1] == 1:
                    feat_batch[(j*n_augment)+i+batch_size] = aug.add_gaussian_noise()
                    label_batch[(j*n_augment)+i+batch_size] = labels[index]
                    j+=1

                if augment[2] == 1:
                    feat_batch[(j*n_augment)+i+batch_size] = aug.add_poisson_noise()
                    label_batch[(j*n_augment)+i+batch_size] = labels[index]
                    j+=1

                i+=1


            yield feat_batch,label_batch

def CNN_model(width,height):
    # #create model
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(width,height,1)))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(labels.shape[1], activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    input_dir = './mnist'
    output_file = 'dataset.csv'

    filename = []
    label = []
    for root,dirs,files in os.walk(input_dir):
        for file in files:
            full_path = os.path.join(root,file)
            filename.append(full_path)
            label.append(os.path.basename(os.path.dirname(full_path)))

    data = pd.DataFrame(data={'filename': filename, 'label':label})
    data.to_csv(output_file,index=False)

    labels = pd.get_dummies(data.iloc[:,1]).values

    X, X_val, y, y_val = train_test_split(
                                            filename, labels,
                                            test_size=0.2,
                                            random_state=1234,
                                            shuffle=True,
                                            stratify=labels
                                            )

    X_train, X_test, y_train, y_test = train_test_split(
                                                        X, y,
                                                        test_size=0.025,
                                                        random_state=1234,
                                                        shuffle=True,
                                                        stratify=y
                                                        )

    width = 28
    height = 28

    test_data = pd.DataFrame(data={'filename': X_test})


    image_gen_train = Generator(X_train,y_train,width,height)
    image_gen_val = Generator(X_val,y_val,width,height)
    image_gen_test = Generator(X_test,None,width,height)


    batch_size = 900
    print('len data: ', len(X_train))
    print('len test data: ', len(X_test))

    #augment represents [speckle, gaussian, poisson]. It means, the augmentation will be done on the augment list element that is 1
    #for example, augment = [1,1,0] corresponds to adding speckle noise and gaussian noise
    augment = [1,1,1]
    model = CNN_model(width,height)

    model.fit_generator(
                        generator=image_gen_train.gen_augment(batch_size=batch_size,augment=augment),
                        steps_per_epoch=np.ceil(len(X_train)/batch_size),
                        epochs=20,
                        verbose=1,
                        validation_data=image_gen_val.gen(),
                        validation_steps=len(X_val)
                        )
    model.save('model_aug_3.h5')
    model = tf.keras.models.load_model('model_aug_3.h5')

    #Try evaluate_generator
    image_gen_test = Generator(X_test,y_test,width,height)
    print(model.evaluate_generator(
                            generator=image_gen_test.gen(),
                            steps=len(X_test)
                            ))

    #Try predict_generator
    image_gen_test = Generator(X_test,None,width,height)
    pred = model.predict_generator(
                            generator=image_gen_test.gen_test(),
                            steps=len(X_test)
                            )
    pred = np.argmax(pred,axis=1)
    # image_gen_test = Generator(X_test,pred,width*3,height*3)
    # image_gen_test.gen_show(pred)
    wrong_pred = []
    for i,ex in enumerate(zip(pred,y_test)):
        if ex[0] != np.argmax(ex[1]):
            wrong_pred.append(i)
    print(wrong_pred)

    # for i in range(len(X_test)):
    #     im = cv2.imread(X_test[i],0)
    #     im = cv2.putText(im, str(pred[i]), (10,15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    #     print(i)
    #     cv2.imshow('image',im)
    #     cv2.waitKey(0)
