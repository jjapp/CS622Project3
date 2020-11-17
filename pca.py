import helpers
import numpy as np


class PCA:

    def __init__(self, X, centering=True, scaling=True):
        self.X = X
        self.centering = centering
        self.scaling = scaling
        self.mean_vector = np.mean(X, axis=0)
        self.std_vector = np.std(self.X, axis=0)
        self.X_centered = self.X - self.mean_vector
        self.X_scaled = self.X_centered / self.std_vector


    def get_z_matrix(self):
        if self.centering is False and self.scaling is False:
            return self.X
        elif self.centering is True and self.scaling is False:
            return self.X_centered
        else:
            return self.X_scaled


def compute_Z(X, centering=True, scaling=False):
    model = PCA(X, centering=centering, scaling=scaling)
    Z = model.get_z_matrix()
    return Z


def compute_covariance_matrix(Z):
    return np.dot(Z.T, Z)



def find_pcs(COV):
    L, PCS = np.linalg.eig(COV)
    idx = L.argsort()[::-1]
    L = L[idx]
    PCS = PCS[:, idx]
    return L, PCS


def project_data(Z, PCS, L, k, var):
    total_var = np.sum(L)
    if k > 0:
        slice = k
        new_vector = PCS[:slice]
    else:
        i = 0
        part_var = 0
        for row in L:
            part_var = part_var + row
            temp_var = part_var/total_var
            i=i+1
            if temp_var > var:
                slice = i
                new_vector = PCS[:slice]
                break

    Z_star=new_vector.dot(Z.T)

    return Z_star


if __name__ == "__main__":
    X, Y = helpers.load_data("data_2.txt")
    Z = compute_Z(X, centering=True, scaling=False)
    cov = compute_covariance_matrix(Z)
    L, PCS = find_pcs(cov)
    #print(PCS)
    #Z_star = project_data(Z, PCS, L, 1, 0)
    print(Z)
