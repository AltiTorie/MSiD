# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np

from utils import polynomial

def mean_squared_error(x, y, w):
    """
    :param x: ciąg wejściowy Nx1
    :param y: ciąg wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: błąd średniokwadratowy pomiędzy wyjściami y oraz wyjściami
     uzyskanymi z wielowamiu o parametrach w dla wejść x
    """
    # y_h = design_matrix(x, len(w) - 1) @ w
    y_h = polynomial(x, w)
    return np.mean((y - y_h) ** 2)


def design_matrix(x_train, M):
    """
    :param x_train: ciąg treningowy Nx1
    :param M: stopień wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzędu M
    """

    output = np.ones(shape=(len(x_train), 1))
    for col in range(1, M + 1):
        output = np.concatenate([output, x_train ** col], axis=1)
    return output
    # x_train.shape = x_train.shape[0]
    # return np.array([x_train**m for m in range(M+1)]).T


def least_squares(x_train, y_train, M):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego 
    wielomianu, a err to błąd średniokwadratowy dopasowania
    """
    phi = design_matrix(x_train, M)
    w = np.linalg.inv(phi.T @ phi) @ phi.T @ y_train
    err = mean_squared_error(x_train, y_train, w)
    return w, err


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego
    wielomianu zgodnie z kryterium z regularyzacją l2, a err to błąd 
    średniokwadratowy dopasowania
    """
    phi = design_matrix(x_train, M)
    w = np.linalg.inv(phi.T @ phi + regularization_lambda * np.eye(M + 1)) @ phi.T @ y_train
    err = mean_squared_error(x_train, y_train, w)
    return w, err


def model_selection(x_train, y_train, x_val, y_val, M_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, które mają byc sprawdzone
    :return: funkcja zwraca krotkę (w,train_err,val_err), gdzie w są parametrami
    modelu, ktory najlepiej generalizuje dane, tj. daje najmniejszy błąd na 
    ciągu walidacyjnym, train_err i val_err to błędy na sredniokwadratowe na 
    ciągach treningowym i walidacyjnym
    """
    temp_err = None
    output = None
    for m in M_values:
        w = least_squares(x_train, y_train, m)[0]
        err_train = mean_squared_error(x_train, y_train, w)
        err_val = mean_squared_error(x_val, y_val, w)
        if temp_err is None:
            temp_err = err_val
            output = (w, err_train, err_val)
        else:
            if temp_err > err_val:
                temp_err = err_val
                output = (w, err_train, err_val)
    return output


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M: stopień wielomianu
    :param lambda_values: lista z wartościami różnych parametrów regularyzacji
    :return: funkcja zwraca krotkę (w,train_err,val_err,regularization_lambda),
    gdzie w są parametrami modelu, ktory najlepiej generalizuje dane, tj. daje
    najmniejszy błąd na ciągu walidacyjnym. Wielomian dopasowany jest wg
    kryterium z regularyzacją. train_err i val_err to błędy średniokwadratowe
    na ciągach treningowym i walidacyjnym. regularization_lambda to najlepsza
    wartość parametru regularyzacji
    """
    temp_err = None
    output = None
    for lam in lambda_values:
        w = regularized_least_squares(x_train, y_train, M, lam)[0]
        err_train = mean_squared_error(x_train, y_train, w)
        err_val = mean_squared_error(x_val, y_val, w)
        if temp_err is None:
            temp_err = err_val
            output = (w, err_train, err_val, lam)
        else:
            if temp_err > err_val:
                temp_err = err_val
                output = (w, err_train, err_val, lam)
    return output
