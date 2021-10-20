#This weeks code focuses on understanding basic functions of pandas and numpy 
#This will help you complete other lab experiments


# Do not change the function definations or the parameters
import numpy as np
import pandas as pd

#input: tuple (x,y)    x,y:int 
def create_numpy_ones_array(shape):
	#Returns a numpy array with one at all index
	array = np.ones(shape)
	return array

#input: tuple (x,y)    x,y:int 
def create_numpy_zeros_array(shape):
	#Returns a numpy array with zeros at all index
	array = np.zeros(shape)
	return array

#input: int  
def create_identity_numpy_array(order):
	#Returns a identity numpy array of the defined order
    array = np.identity(order)
    return array

#input: numpy array
def matrix_cofactor(array):
	#Returns the cofactor matrix of the given array
    if np.linalg.det(array) != 0: # If matrix is invertable
        array = np.linalg.inv(array).T * np.linalg.det(array) # Formula to calculate cofactor matrix  == det(A) * inverse(A)^T
        return array

#Input: (numpy array, int ,numpy array, int , int , int , int , tuple,tuple)
#tuple (x,y)    x,y:int 
def f1(X1,coef1,X2,coef2,seed1,seed2,seed3,shape1,shape2):

    # Generating Random Matrices W1 and W2 using seeds
    np.random.seed(seed1)
    W1 = np.random.rand(shape1[0], shape1[1])
    np.random.seed(seed2)
    W2 = np.random.rand(shape2[0], shape2[1])
    
    # X1 and X2 calculations
    X1 = np.linalg.matrix_power(X1, coef1)
    X2 = np.linalg.matrix_power(X2, coef2)
    
    # Checking if matrix multiplication is possible
    if W1.shape[1] != X1.shape[0] or W2.shape[1] != X2.shape[0]:
        return -1
    
    # Calculating Products
    Y1 = np.matmul(W1, X1)
    Y2 = np.matmul(W2, X2)
    
    # Checking if the shapes of the matrices will allow for addition
    if Y1.shape != Y2.shape:
        return -1
    # Addition
    Y3 = np.add(Y1, Y2)
    
    # Generating B
    np.random.seed(seed3)
    B = np.random.rand(Y3.shape[0], Y3.shape[1])
    
    # Final Answer
    ans = np.add(Y3, B)
    return ans



def fill_with_mode(filename, column):
    """
    Fill the missing values(NaN) in a column with the mode of that column
    Args:
        filename: Name of the CSV file.
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
    """
    df = pd.read_csv(filename)
    df[column] = df[column].fillna(df[column].mode()[0])
    return df

def fill_with_group_average(df, group, column):
    """
    Fill the missing values(NaN) in column with the mean value of the 
    group the row belongs to.
    The rows are grouped based on the values of another column

    Args:
        df: A pandas DataFrame object representing the data.
        group: The column to group the rows with
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
    """
    df[column] = df[column].fillna(df.groupby(group)[column].transform('mean'))
    return df


def get_rows_greater_than_avg(df, column):
    """
    Return all the rows(with all columns) where the value in a certain 'column'
    is greater than the average value of that column.

    row where row.column > mean(data.column)

    Args:
        df: A pandas DataFrame object representing the data.
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
	"""

    average = df[column].mean() # Gets the mean of the column
    return df.loc[df[column] > average]
    

