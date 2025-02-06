###
# Source:
# R. Bravo, localECM: A Local Empirical Cubature Method Implementation, GitHub, 2024. [Online]. Available: https://github.com/Rbravo555/localECM. [Accessed: Feb. 3, 2025].
###


import numpy as np

try:
    from matplotlib import pyplot as plt
    missing_matplotlib = False
except ImportError as e:
    missing_matplotlib = True


class EmpiricalCubatureMethod():
    """
    This class selects a subset of elements and corresponding positive weights necessary for the construction of a hyper-reduced order model
    Reference: Local-ECM: An empirical cubature hyper-reduction method adapted to local reduced order models." arXiv preprint arXiv:2310.15769 (2023)"
    """



    def __init__(
        self,
        ECM_tolerance = 0,
        Filter_tolerance = 0,
        Plotting = False,
        MaximumNumberUnsuccesfulIterations = 100
    ):
        """
        Constructor setting up the parameters for the Element Selection Strategy
            ECM_tolerance: approximation tolerance for the element selection algorithm
            Filter_tolerance: parameter limiting the number of candidate points (elements) to those above this tolerance
            Plotting: whether to plot the error evolution of the element selection algorithm
        """
        self.ECM_tolerance = ECM_tolerance
        self.Filter_tolerance = Filter_tolerance
        self.Name = "EmpiricalCubature"
        self.Plotting = Plotting
        self.MaximumNumberUnsuccesfulIterations = MaximumNumberUnsuccesfulIterations

    def SetUp(
        self,
        G,
        Weights,
        constrain_sum_of_weights = False,
        InitialCandidatesSet = None,
        use_L2_weighting = False
    ):
        """
        Method for setting up the element selection
        input:  ResidualsBasis: numpy array containing a basis to the residuals projected
        """
        self.W = Weights
        self.G = G
        self.y = InitialCandidatesSet
        self.add_constrain_count = None
        self.use_L2_weighting = use_L2_weighting
        if constrain_sum_of_weights:
            #This avoids the trivial solution w = 0
            constant_function = np.ones(np.shape(self.G)[1])
            projection_of_constant_function_on_col_G = constant_function - self.G.T@( self.G @ constant_function)
            norm_projection = np.linalg.norm(projection_of_constant_function_on_col_G)
            if norm_projection>1e-10:
                projection_of_constant_function_on_col_G/= norm_projection
                self.G = np.vstack([ self.G , projection_of_constant_function_on_col_G] )
                self.add_constrain_count = -1
        if self.use_L2_weighting:
            self.b = self.G @ np.sqrt(self.W)
        else:
            self.b = self.G @ self.W
        self.UnsuccesfulIterations = 0



    def Initialize(self):
        """
        Method performing calculations required before launching the Calculate method
        """
        self.Gnorm = np.linalg.norm(self.G[:self.add_constrain_count,:], axis = 0)
        M = np.shape(self.G)[1]
        normB = np.linalg.norm(self.b)

        if self.y is None:
            self.y = np.arange(0,M,1) # Set of candidate points (those whose associated column has low norm are removed)

            if self.Filter_tolerance > 0:
                TOL_REMOVE = self.Filter_tolerance * normB
                rmvpin = np.where(self.Gnorm[self.y] < TOL_REMOVE)
                self.y = np.delete(self.y,rmvpin)
        else:
            self.y_complement = np.arange(0,M,1)
            self.y_complement = np.delete(self.y_complement, self.y)
            if self.Filter_tolerance > 0:
                TOL_REMOVE = self.Filter_tolerance * normB
                self.y_complement = np.delete(self.y_complement,np.where(self.Gnorm[self.y_complement] < TOL_REMOVE))
                self.y = np.delete(self.y,np.where(self.Gnorm[self.y] < TOL_REMOVE))
                if np.size(self.y)==0:
                    self.y=self.y_complement.copy()

        self.z = {}  # Set of intergration points
        self.mPOS = 0 # Number of nonzero weights
        self.r = self.b.copy() # residual vector
        self.m = len(self.b) # Default number of points
        self.nerror = np.linalg.norm(self.r)/normB
        self.nerrorACTUAL = self.nerror


    def Run(self):
        """
        Method launching the element selection algorithm to find a set of elements: self.z, and weights: self.w
        """
        self.Initialize()
        self.Calculate()

    def expand_candidates_with_complement(self):
        self.y = np.r_[self.y,self.y_complement]
        print('expanding set to include the complement...')
        ExpandedSetFlag = True
        return ExpandedSetFlag


    def Calculate(self):
        """
        Method calculating the elements and weights, after the Initialize method was performed
        """
        MaximumLengthZ = 0
        ExpandedSetFlag = False
        k = 1 # number of iterations
        self.success = True
        while self.nerrorACTUAL > self.ECM_tolerance and self.mPOS < self.m and np.size(self.y) != 0:

            if  self.UnsuccesfulIterations >  self.MaximumNumberUnsuccesfulIterations and not ExpandedSetFlag:
                ExpandedSetFlag = self.expand_candidates_with_complement()

            #Step 1. Compute new point
            if np.size(self.y)==1:#, np.int64) or isinstance(self.y, np.int32):
                indSORT = 0
                i = int(self.y)
            else:
                ObjFun = self.G[:,self.y].T @ self.r.T
                ObjFun = ObjFun.T # / self.Gnorm[self.y]
                indSORT = np.argmax(ObjFun)
                i = self.y[indSORT]
            if k==1:
                alpha = np.linalg.lstsq(self.G[:, [i]], self.b)[0]
                H = 1/(self.G[:,i] @ self.G[:,i].T)
            else:
                H, alpha = self._UpdateWeightsInverse(self.G[:,self.z],H,self.G[:,i],alpha)

            #Step 3. Move i from set y to set z
            if k == 1:
                self.z = i
            else:
                self.z = np.r_[self.z,i]

            if np.size(self.y)==1:#isinstance(self.y, np.int64) or isinstance(self.y, np.int32):
                self.expand_candidates_with_complement()
                self.y = np.delete(self.y,indSORT)
            else:
                self.y = np.delete(self.y,indSORT)

            # Step 4. Find possible negative weights
            if any(alpha < 0):
                print("WARNING: NEGATIVE weight found")
                indexes_neg_weight = np.where(alpha <= 0.)[0]
                self.y = np.append(self.y, (self.z[indexes_neg_weight]).T)
                self.z = np.delete(self.z, indexes_neg_weight)
                H = self._MultiUpdateInverseHermitian(H, indexes_neg_weight)
                alpha = H @ (self.G[:, self.z].T @ self.b)
                alpha = alpha.reshape(len(alpha),1)


            if np.size(self.z) > MaximumLengthZ :
                self.UnsuccesfulIterations = 0
            else:
                self.UnsuccesfulIterations += 1

            #Step 6 Update the residual
            if np.size(alpha)==1:
                self.r = self.b.reshape(-1,1) - (self.G[:,self.z] * alpha).reshape(-1,1)
                self.r = np.squeeze(self.r)
            else:
                Aux = self.G[:,self.z] @ alpha
                self.r = np.squeeze(self.b - Aux.T)
            self.nerror = np.linalg.norm(self.r) / np.linalg.norm(self.b)  # Relative error (using r and b)
            self.nerrorACTUAL = self.nerror


            # STEP 7
            self.mPOS = np.size(self.z)
            print(f'k = {k}, m = {np.size(self.z)}, error n(res)/n(b) (%) = {self.nerror*100},  Actual error % = {self.nerrorACTUAL*100} ')

            if k == 1:
                ERROR_GLO = np.array([self.nerrorACTUAL])
                NPOINTS = np.array([np.size(self.z)])
            else:
                ERROR_GLO = np.c_[ ERROR_GLO , self.nerrorACTUAL]
                NPOINTS = np.c_[ NPOINTS , np.size(self.z)]

            MaximumLengthZ = max(MaximumLengthZ, np.size(self.z))
            k = k+1

            if k-MaximumLengthZ>1000 and ExpandedSetFlag:
                """
                this means using the initial candidate set, it was impossible to obtain a set of positive weights.
                Try again without constraints!!!
                """
                self.success = False
                break

        if self.use_L2_weighting:
            self.w = (alpha.T * np.sqrt(self.W[self.z])).T
        else:
            self.w = alpha

        print(f'Total number of iterations = {k}')


        if missing_matplotlib == False and self.Plotting == True:
            plt.plot(NPOINTS[0], ERROR_GLO[0])
            plt.title('Element Selection Error Evolution')
            plt.xlabel('Number of elements')
            plt.ylabel('Error %')
            plt.show()


    def _UpdateWeightsInverse(self, A,Aast,a,xold):
        """
        Method for the cheap update of weights (self.w), whenever a negative weight is found
        """
        c = np.dot(A.T, a)
        d = np.dot(Aast, c).reshape(-1, 1)
        s = np.dot(a.T, a) - np.dot(c.T, d)
        aux1 = np.hstack([Aast + np.outer(d, d) / s, -d / s])
        if np.shape(-d.T / s)[1]==1:
            s = s.reshape(1,-1)
            aux2 = np.squeeze(np.hstack([-d.T / s, 1 / s]))
        else:
            aux2 = np.hstack([np.squeeze(-d.T / s), 1 / s])
        Bast = np.vstack([aux1, aux2])
        v = np.dot(a.T, self.r) / s
        x = np.vstack([(xold - d * v), v])
        return Bast, x


    def _MultiUpdateInverseHermitian(self, invH, neg_indexes):
        """
        Method for the cheap update of weights (self.w), whenever a negative weight is found
        """
        neg_indexes = np.sort(neg_indexes)
        for i in range(np.size(neg_indexes)):
            neg_index = neg_indexes[i] - i
            invH = self._UpdateInverseHermitian(invH, neg_index)
        return invH


    def _UpdateInverseHermitian(self, invH, neg_index):
        """
        Method for the cheap update of weights (self.w), whenever a negative weight is found
        """
        if neg_index == np.shape(invH)[1]:
            aux = (invH[0:-1, -1] * invH[-1, 0:-1]) / invH(-1, -1)
            invH_new = invH[:-1, :-1] - aux
        else:
            aux1 = np.hstack([invH[:, 0:neg_index], invH[:, neg_index + 1:], invH[:, neg_index].reshape(-1, 1)])
            aux2 = np.vstack([aux1[0:neg_index, :], aux1[neg_index + 1:, :], aux1[neg_index, :]])
            invH_new = aux2[0:-1, 0:-1] - np.outer(aux2[0:-1, -1], aux2[-1, 0:-1]) / aux2[-1, -1]
        return invH_new






