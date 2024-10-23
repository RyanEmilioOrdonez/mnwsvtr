# mnwsvtr.jl  Readme File

This Julia code file is meant to be a transcribing of the Stata code
written by Matt Webb and used in the paper:

James G. MacKinnon, Morten Ø. Nielsen, and Matthew D. Webb, "Testing
  for the appropriate level of clustering in linear regression models"

  https://ideas.repec.org/p/qed/wpaper/1428.html

This file implements the score-variance test for the level of 
clustering as proposed in the paper.

The three main functions of the file are VECHTEST, MNWTEST, and IMTEST.

The file also contains 4 utility functions and EXAMPLE CODE.

The example code can be found at Line 632 and beyond. You will need to replace the 
path to your data files in all instances.

The Julia code also contains catches and throws that will alert the user if any 
package is not installed and will AUTOMATICALLY install any needed packages.

###############################################################################################
### VECHTEST Definition #######################################################################
###############################################################################################

The VECHTEST function is defined to take the following arguments:

VECHTEST(df::DataFrame, y::AbstractString, xv::AbstractString, 
cv::AbstractString, fc::AbstractString, cc::AbstractString)

df: the dataframe (of type = DataFrame)

y: the outcome variable (a string representing the column name of the 
outcome variable)

xv: the variable(s) of interest (a string representing the column name(s) 
of the variables interest, separated by a single `space`)

cv: the control variables (a string representing the column name(s) of all 
other columns, separated by a single `space`)

fc: the null variable, a.k.a the (fine) level of clustering under the null
(a string representing the column name of the null variable)

cc: the alt variable, a.k.a the (coarse) level of clustering under the alternative
(a string representing the column name of the alt variable)

This function will return a Dictionary{Symbol, Any} type object, which is essentially Julia's version of a Open Hash table, so it has a key-value pair. The dictionary will contain the 
following keys, with their associated values:

:H = the number of clusters under the null

:G = the number of clusters under alternative

:xn = the number of x-variable(s) of interest (length of `xv` from above)

:XV = the string representing the column names of the x-variable(s) of interest

:CV = the string representing the column names of control variables 
(the inputted `cv` from above)

:theta = the vector of contrasts as defined in the paper; the single column vectorization of the lower triangular of the difference between the empirical score matrices

:tau = the studentized test statistic

:var = the variance estimator

:chi_df = the degrees of freedom for the specific number of x-variables of interest

:data = a modified version of the inputted dataframe that includes the columns: "calt, cnul, cros, id, ccnt, tempres, tempfit"


###############################################################################################
### MNWTEST Definition ########################################################################
###############################################################################################

The MNWTEST function is defined as follows: 
MNWTEST(df::DataFrame, y::AbstractString, xv::AbstractString, cv::AbstractString, fc::AbstractString, cc::AbstractString, b::Int)

df: the dataframe (of type = DataFrame)

y: the outcome variable (a string representing the column name of the 
outcome variable)

xv: the variable(s) of interest (a string representing the column name(s) 
of the variables interest)

cv: the control variables (a string representing the column name(s) of all 
other columns)

fc: the null variable, a.k.a the (fine) level of clustering under the null
(a string representing the column name of the null variable)

cc: the alt variable, a.k.a the (coarse) level of clustering under the alternative
(a string representing the column name of the alt variable)

b: the desired number of bootstraps to occur, must be an integer

This function will also return a Dictionary type object. The outputs are as follows:

:H = the number of clusters under the null

:G = the number of clusters under alternative

:theta = the single column vectorization of the lower triangular of the difference between the empirical score matrices

:tau = the studentized test statistic

:chi_df = the degrees of freedom for the specific number of x-variables of interest

:MNW_P = the p-value calculated using the tau quantile from a VECHTEST call, the lower of either the right or left tails

:bp = the bootstrap p-value

###############################################################################################
### IMTEST Definition #########################################################################
###############################################################################################

The IMTEST function is implemented as follows:

IMTEST(df::DataFrame, y::AbstractString, xv::AbstractString, cv::AbstractString,  fr::AbstractString, fc::AbstractString, cc::AbstractString, tm::Int)

df, y, xv, cv, fc, cc - same definitions as above

fr: a string representing a `fixed effect` as defined in the context of fixed effect models. (this is also a variable that represents a (coarse) level clustering under the alternative.)

tm - a desired number of replication loops, must be an integer

The output of this function is a Dictionary{Symbol, :Any} type with the following key-value pairs:

:S2 = the IM (2012) standard error, it is just the variance of the j number of betas

:IM_p = the p-value of this test

###############################################################################################
### Utility Functions #########################################################################
###############################################################################################

1. vech(A::AbstractMatrix) - analogous to the R implementation of the function of the same name. Returns a vectorization of the lower triangular of a matrix.

2. extract_factor_column(s::AbstractString) - looks at the string of column names and determines if any column_name is of the form "factor(column_name)" and returns `nothing` if not or the column_name if so. Mainly, used for replication of the paper's dataset. 

3. replace_factor(s::AbstractString) - If extract_factor_column() was not empty, this will replace that element in the string to be of the form "column_name". These functions are only truly useful if R was used to manipulate the dataframe, and columns got converted to factor types, and their names got changed as a result

4. lcat(lsmnw::Dict) - instead of using the getindex(dict_name, :key) method in Julia to call individual values from the MNWTEST's output, one may simply input the dictionary name into this function to get all results at once. example: ls1 = MNWTEST(...) => lcat(ls1) will 


