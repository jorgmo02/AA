from scipy.io import loadmat
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from get_vocab_dict import getVocabDict
import process_email as mail
import codecs
import glob


def visualize_boundary(X, y, svm, file_name):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)
    plt.savefig(file_name)
    plt.close()


def load_example_data(filename):
    data = loadmat(filename)
    X = data['X']
    y = data['y'][:, 0]
    return X, y


def kernel_lineal():
    C = 100
    svm = SVC(kernel='linear', C=C)
    X, y = load_example_data('ex6data1.mat')
    svm.fit(X, y)
    visualize_boundary(X, y, svm, 'ex1-C{}.png'.format(C))


def kernel_gaussiano():
    X, y = load_example_data("ex6data2.mat")
    C = 1
    sigma = 0.1
    svm = SVC(kernel='rbf', C=C, gamma=1 / (2 * sigma ** 2))
    svm.fit(X, y)
    visualize_boundary(X, y, svm, 'ex2-C{}.png'.format(C))


def porcentaje_aciertos(X, y, svm):
    h = svm.predict(X)
    aciertos = np.sum(h == y)
    return (aciertos / X.shape[0]) * 100


def busca_mejores_parametros():
    data = loadmat('ex6data3.mat')
    X = data['X']
    y = data['y'][:, 0]
    print(y)
    X_val = data['Xval']
    Y_val = data['yval'][:, 0]

    C_samples = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    Sigma_samples = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    Result = np.empty((C_samples.shape[0], Sigma_samples.shape[0]))

    for i, C in enumerate(C_samples):
        for j, sigma in enumerate(Sigma_samples):
            svm = SVC(kernel='rbf', C=C, gamma=1 / (2 * sigma ** 2))
            svm.fit(X, y)
            Result[i, j] = porcentaje_aciertos(X_val, Y_val, svm)

    mejor = np.unravel_index(np.argmax(Result), Result.shape)
    print("Mejor C {} y mejor Sigma {}".format(C_samples[mejor[0]], Sigma_samples[mejor[1]]))


def load_mail(filename, vocab):
    email_contents = codecs.open(filename, 'r', encoding='utf-8', errors='ignore').read()
    email = mail.email2TokenList(email_contents)
    vec = np.zeros(len(vocab))
    for word in email:
        if word in vocab.keys():
            vec[vocab[word] - 1] = 1

    return vec


def LoadSet(setFiles, yVal, vocab):
    spamFiles = glob.glob(setFiles)
    X = np.zeros((len(spamFiles), len(vocab)))
    Y = np.full(len(spamFiles), fill_value=yVal)
    for i, file in enumerate(spamFiles):
        np.copyto(dst=X[i], src=load_mail(file, vocab))

    return X, Y


def filtra_spam():
    vocab = getVocabDict()
    X_Spam, Y_Spam = LoadSet("spam/*.txt", 1, vocab)
    X_EHam, Y_EHam = LoadSet("easy_ham/*.txt", 0, vocab)
    X_HHam, Y_HHam = LoadSet("hard_ham/*.txt", 0, vocab)

    print(X_HHam.shape)
    print(Y_HHam.shape)
    print(Y_HHam)

    percTrain = 0.6
    percVal = 0.2
    percTest = 0.2

    X = np.vstack((X_Spam[:(int)(percTrain * X_Spam.shape[0])],
                   X_EHam[:(int)(percTrain * X_EHam.shape[0])],
                   X_HHam[:(int)(percTrain * X_HHam.shape[0])]))

    Y = np.hstack((Y_Spam[:(int)(percTrain * Y_Spam.shape[0])],
                   Y_EHam[:(int)(percTrain * Y_EHam.shape[0])],
                   Y_HHam[:(int)(percTrain * Y_HHam.shape[0])]))

    X_test = np.vstack((X_Spam[(int)(percTrain * X_Spam.shape[0]):],
                        X_EHam[(int)(percTrain * X_EHam.shape[0]):],
                        X_HHam[(int)(percTrain * X_HHam.shape[0]):]))

    Y_test = np.hstack((Y_Spam[(int)(percTrain * Y_Spam.shape[0]):],
                        Y_EHam[(int)(percTrain * Y_EHam.shape[0]):],
                        Y_HHam[(int)(percTrain * Y_HHam.shape[0]):]))

    svm = SVC(kernel='rbf', C=1, gamma=1 / (2 * 0.1 ** 2))
    svm.fit(X, Y)
    print(porcentaje_aciertos(X_test, Y_test, svm))


def main():
    busca_mejores_parametros()
    filtra_spam()


main()
