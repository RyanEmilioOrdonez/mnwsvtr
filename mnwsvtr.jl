####### Import necessary packages and installing them if they arent already
####### and using catches to give proper error messages
try
    using DataFrames
catch e
    if isa(e, ArgumentError) && occursin("DataFrames", e.msg)
        println("DataFrames is not already installed. Installing now...")
        import Pkg
        Pkg.add("DataFrames")
        using DataFrames
    else
        rethrow(e)
    end
end

try
    using GLM
catch e
    if isa(e, ArgumentError) && occursin("GLM", e.msg)
        println("GLM is not already installed. Installing now...")
        import Pkg
        Pkg.add("GLM")
        using GLM
    else
        rethrow(e)
    end
end

try
    using CategoricalArrays
catch e
    if isa(e, ArgumentError) && occursin("CategoricalArrays", e.msg)
        println("CategoricalArrays is not already installed. Installing now...")
        import Pkg
        Pkg.add("CategoricalArrays")
        using CategoricalArrays
    else
        rethrow(e)
    end
end

try
    using StatsModels
catch e
    if isa(e, ArgumentError) && occursin("StatsModels", e.msg)
        println("StatsModels is not already installed. Installing now...")
        import Pkg
        Pkg.add("StatsModels")
        using StatsModels
    else
        rethrow(e)
    end
end

try
    using Distributions
catch e
    if isa(e, ArgumentError) && occursin("Distributions", e.msg)
        println("Distributions is not already installed. Installing now...")
        import Pkg
        Pkg.add("Distributions")
        using Distributions
    else
        rethrow(e)
    end
end

try
    using FixedEffectModels
catch e
    if isa(e, ArgumentError) && occursin("FixedEffectModels", e.msg)
        println("FixedEffectModels is not already installed. Installing now...")
        import Pkg
        Pkg.add("FixedEffectModels")
        using FixedEffectModels
    else
        rethrow(e)
    end
end

try
    using StatsBase
catch e
    if isa(e, ArgumentError) && occursin("StatsBase", e.msg)
        println("StatsBase is not already installed. Installing now...")
        import Pkg
        Pkg.add("StatsBase")
        using StatsBase
    else
        rethrow(e)
    end
end

try
    using StatFiles
catch e
    if isa(e, ArgumentError) && occursin("StatFiles", e.msg)
        println("StatFiles is not already installed. Installing now...")
        import Pkg
        Pkg.add("StatFiles")
        using StatFiles
    else
        rethrow(e)
    end
end

try
    using StatsFuns
catch e
    if isa(e, ArgumentError) && occursin("StatsFuns", e.msg)
        println("StatsFuns is not already installed. Installing now...")
        import Pkg
        Pkg.add("StatsFuns")
        using StatsFuns
    else
        rethrow(e)
    end
end

try
    using CSV
catch e
    if isa(e, ArgumentError) && occursin("CSV", e.msg)
        println("DataFrames is not already installed. Installing now...")
        import Pkg
        Pkg.add("CSV")
        using CSV
    else
        rethrow(e)
    end
end

using LinearAlgebra
using Random


# Defining custom vech function equivalent to the one in R
function vech(A::AbstractMatrix)
    # finds the nrows of the matrix A
    n = size(A, 1)
    # defines v to be a vector whose elements are the same type 
    # as those found in A, and give it the size of the lower
    # triangular
    v = Vector{eltype(A)}(undef, (n * (n + 1)) ÷ 2)
    # setting the start of the v vector's indices
    k = 1
    
    # looping over 
    @inbounds for j in 1:n, i in j:n
        v[k] = A[i, j]
        k += 1
    end
    return v
end

# Function to extract the column name from factor(column_name)
function extract_factor_column(s::AbstractString)
    # defining an expression to search for
    regex = r"factor\s*\(\s*([^)\s]+)\s*\)"
    # checking for a match
    m = match(regex, s)
    # returning nothing or the column_name
    return m === nothing ? nothing : m.captures[1]
end

# # For use with XVARSX, XVARAX, XVARBX to replace "function()" with "term()" so that glm understands it
function replace_factor(s::AbstractString)
    # defining the pattern to look for
    pattern = r"factor\s*\(\s*([^)]+)\s*\)"
    # replacing the pattern `factor(...)` 
    return replace(s, pattern => s"\1")
end

function VECHTEST(df::DataFrame, y::AbstractString, 
    xv::AbstractString, cv::AbstractString, 
    fc::AbstractString, cc::AbstractString)

    # defining temporary pointer nodes
    stard=df
    CTEMP=cv
    XTEMP=xv
    y=y
    null=fc
    alt1=cc


    # Check if the 'newid' column exists
    if :newid ∉ names(df)
    # Add the 'newid' column with sequential numbers
    stard.newid = 1:nrow(stard)
    #println("Column 'newid' added.")
    end

    # making the fixed effect x-variables into an equation-type string 
    rightC = join(split(CTEMP, r"\s+"), "+")
    #making the X-variables of interest into an equation-type string
    rightX = join(split(XTEMP, r"\s+"), "+")
    # putting the two strings together
    right = join([rightX, rightC], "+")

    # For use in XVARSX... replacing "factor(variable)"" with just the variable name
    # in the strings and converting the data values to type = string so that feols can
    # be used without issue.
    # extracting just the column name without the `factor(`column_name`)` part
    #factor_column = extract_factor_column(right)
    
    # Use a regular expression to extract variables inside the factor() function
    factor_vars = match(r"factor\(([^)]+)\)", right)

    if factor_vars != nothing
        # Find all matches of the pattern in the string
        matches = collect(eachmatch(r"factor\(([^)]+)\)", right))
        # Extract the captured groups (the variables inside factor(...))
        extracted_vars = [m.captures[1] for m in matches]
        # Combine the extracted variables into a single string
        factor_column = join(extracted_vars, " ")
        # Parse the string into an array of column names
        column_names = split(factor_column)

        # Convert specified columns to string
        for col in column_names
            stard[!, col] = string.(stard[!, col])
        end
        pattern = r"factor\s*\(\s*([^)]+)\s*\)"
        # replacing the pattern `factor(...)` 
        right = replace(right, pattern => s"\1")
        rightC = replace(rightC, pattern => s"\1")
        rightX = replace(rightX, pattern => s"\1")
    end

    # Converting the column to a categorical array and then to 
    # integer indices, storing in new column `calt`
    stard.calt = levelcode.(CategoricalArray(stard[:, alt1]))

    # /*determine number of clusters under alternative*/
    # Calculate the maximum value of `calt`
    G = maximum(stard.calt)

    # /*correct for possibility of non-unique naming of fine clusters*/
    # /*determine number of clusters under null*/

    # Converting the column `null` to a categorical array and then to 
    # integer indices, storing in new column `cnul`
    stard.cnul = levelcode.(CategoricalArray(stard[!, null]))

    # new temporary dataframe noder
    temp = stard

    # Group by 'calt' and 'cnul'
    # create 'cros' column with unique group IDs
    temp.cros = groupindices(groupby(temp, [:calt, :cnul]))

    # Add 'cros' column from 'temp' to 'stard'
    stard.cros = temp.cros

    #determine the number of clusters under the null
    # Calculate the maximum value of `cros`
    H = maximum(stard.cros)

    # sorting stard by `calt` and `cros`
    sort!(stard, [:calt, :cros])

    # making a new dataframe that includes a groupID on the stard
    # dataframe

    groups = groupby(stard, [:calt, :cros])

    # Create the `id` column for each group with row numbers
    dt = combine(groups, sdf -> DataFrame(id = 1:nrow(sdf)))

    # defining column `ccnt` that is that new dataframe's groupID column
    stard.ccnt = dt.id

    # defining linear regression formula as a string
    frm = y * " ~ " * right

    # converting string-formula to a formula type for use in lm()
    flm = @eval(@formula($(Meta.parse(frm))))

    # Fitting a linear model
    model = lm(flm, stard)

    # extracting values from the regression
    stard.tempres = residuals(model)
    stard.tempfit = fitted(model)
    NMK = dof_residual(model)
    N = nobs(model)
    K = N-NMK

    # defining variables of interest
    xterms = split(rightX, '+')
    # finding how many variables of interest there are
    xnum = length(xterms)

    # making a vector of placeholders numbered for each var of interest
    cns = ["tempsc$i" for i in 1:xnum]

    # defining new dataframe with columns: `calt`,`cnul`, and `cros`
    # from the stard dataframe
    VC = DataFrame(
        calt = stard.calt,
        cnul = stard.cnul,
        cros = stard.cros
    )

    # defining new temporary dataframe named orid
    orid = stard

    # fitting an lm without the vars of interest
    for x in xterms
        flmtemp = @eval(@formula($(Meta.parse("$x ~ $rightC"))))
        mdltemp = lm(flmtemp, orid)
        VC[!, Symbol("tempres_$x")] = stard.tempres .* residuals(mdltemp)
    end

    # setting values to rename the VC dataframe's columns and then renaming them
    l = 4
    r = ncol(VC)
    rename(VC, Dict(zip(names(VC)[l:r], cns)))


    M_A = (G/(G - 1)) * ((N-1)/NMK)
    M_F = (N-1)/(NMK)*H/(H-1)
    Gk = zeros(xnum^2, div(xnum * (xnum + 1), 2))
        

    for i in 1:xnum
        for j in i:xnum
            a = (j - 1) * xnum + i
            b = (i - 1) * xnum + j
            c = (j - 1) * (xnum - j ÷ 2) + i
            Gk[a, c] = 1
            Gk[b, c] = 1
        end
    end

    # Hk calculation
    Hk = inv(transpose(Gk) * Gk) * transpose(Gk)

    # Initialize matrices
    global temp_sumg = zeros(xnum, xnum)
    global temp_num_alt = zeros(xnum, xnum)
    global var_right = zeros(size(Hk, 1), size(Hk, 1))
    global var_left = zeros(size(Hk, 1), size(Hk, 1))   
    global ALT = zeros(xnum, xnum)
    global NLL = zeros(xnum, xnum)
    global theta = zeros(size(Hk, 1), 1)

    #aggregation
    cols_to_aggregate = names(VC)[l:r]

    # /*sum scores by either alt or null clusters*/
    sh_h = combine(groupby(VC, Symbol("cros")),
            [Symbol(col) => sum => Symbol(col) for col in cols_to_aggregate])
    sg = combine(groupby(VC, Symbol("calt")),
            [Symbol(col) => sum => Symbol(col) for col in cols_to_aggregate])

    # Extract specific columns
    lx = 2
    lr = lx + xnum - 1


    sh_h = Matrix(sh_h[:, lx:lr])
    sg = Matrix(sg[:, lx:lr])

    for g in 1:G

        # Calling the global variables so that they can be used in the loop
        global temp_sumg
        global temp_num_alt
        global var_right
        global var_left 
        global ALT
        global NLL
        global theta

        temp_sumh = zeros(xnum, xnum)
        temp_var_left =zeros(xnum, xnum)
        temp_var_right =zeros(xnum, xnum)

        # /*alt numerator*/
        temp_sg = sg[g,:]
        temp_num_alt = temp_sg * transpose(temp_sg)
        ALT = ALT + temp_num_alt

        # /*which obs are in cluster g*/
        idx = Vector(stard[findall((stard.calt .== g) .& (stard.ccnt .== 1)), :cros])

        for i in idx
            # /*extract relevant row, i,  from score matrix*/
            sh1 = sh_h[i, :]
            temp_cross = sh1 * sh1'

            # var left
            temp_sumh = temp_sumh + temp_cross

            # var right
            temp_var_right = Hk * kron(temp_cross, temp_cross) * Hk'
            var_right = temp_var_right + var_right
        end

        # var left
        temp_var_left = Hk * kron(temp_sumh, temp_sumh) * Hk'
        var_left = temp_var_left .+ var_left
        # /*null numerator*/
        temp_sumg = temp_sumg .+ temp_sumh
    end

    # /*theta*/
    NLL= temp_sumg
    NLL = M_F*NLL
    ALT = M_A*ALT

    theta = vech(ALT - NLL)

    # /*variance*/
    var_left = 2 * var_left
    var_right = 2 * var_right
    var = var_left - var_right

    # /*tau stat*/
    if xnum == 1
        tau = first(theta) / sqrt(first(var))
    else
        tau = dot(theta, inv(var), theta)
    end

    chi_df = xnum * (xnum + 1) / 2

    return (
        Dict(
            :H => H,
            :G => G,
            :xn => xnum,
            :XV => XTEMP,
            :CV => CTEMP,
            :theta => theta,
            :tau => tau,
            :var => var,
            :chi_df => chi_df,
            :data => stard
        )
    )
end

function MNWTEST(df::DataFrame, y::AbstractString, 
    xv::AbstractString, cv::AbstractString, 
    fc::AbstractString, cc::AbstractString, b::Int)



    # Defining temporary pointers
    df = df
    y = y
    XTEMP = xv
    CTEMP = cv
    null = fc
    alt1 = cc
    B = b

    # calling the VECHTEST function
    ls = VECHTEST(df,y,XTEMP, CTEMP, null, alt1)

    # extracting VECHTEST values
    xnum = getindex(ls,:xn)
    tauhat = getindex(ls, :tau)
    chi_df = getindex(ls, :chi_df)

    # P-vals
    if xnum == 1
        # One-sided P_value
        MNW_P_1s = 1-normcdf(tauhat)
        # Two-sided P_value
        MNW_P = 2 * min(normcdf(tauhat), 1-normcdf(tauhat))
    else
        if xnum >=2
            MNW_P = 1-chisqcdf(chi_df, tauhat)
        end
    end

    dt = getindex(ls, :data)

    # Sorting dt by `null`
    sort(dt,null)

    # extracting some values
    temper = dt[:,:tempres]
    tempft = dt[:,:tempfit]

    # /*intitialize variables and bootstrap matrix*/
    taus = fill(1.0, B)
    taus_1s = fill(1.0, B)
    Random.seed!(42)

    # /*calculate tau for bootstrap sample*/
    for i in 1:B
        
        # new temporary dataframe
        dg = dt
            
        
        # Group by `null`, then add a new column `uni` with 
        # random uniform values
        dg = transform(groupby(dt, null)) do sdf
            (uni = rand(nrow(sdf)),)
        end
        
        # /*create bootstrap y - using wild cluster bootstrap*/
        temp_uni = dg.uni
        temp_pos = temp_uni .<0.5
        # rademacher indicator*/
        temp_ernew = (2 .*temp_pos .-1).*temper
        # /*transformed residuals */
        temp_ywild = tempft + temp_ernew
        
        # /*calculate tau using bootstrap y*/
        dt.booty = temp_ywild

        lstemp= VECHTEST(dt,"booty",XTEMP,CTEMP,null,alt1)

        taus[i] = getindex(lstemp,:tau)
    end

    # /*calculate and display bootstrap P value*/
    if xnum == 1
        temp_rej = abs(tauhat) .<= abs.(taus)
        # One-sided
        temp_rej_1s = tauhat .<= taus
    else
        temp_rej =  tauhat .<= taus
    end

    temp_U = fill(1, length(temp_rej), 1)
    # One-sided
    temp_U_1s = fill(1, length(temp_rej_1s), 1)

    temp_sum = transpose(temp_U)*temp_rej
    # One-sided
    temp_sum_1s = transpose(temp_U_1s)*temp_rej_1s

    boot_p = temp_sum / length(temp_rej)
    # One-sided
    boot_p_1s = temp_sum_1s / length(temp_rej_1s)

    return (
        Dict(
            :H => getindex(ls,:H),
            :G => getindex(ls,:G),
            :theta => getindex(ls, :theta),
            :tau => getindex(ls, :tau),
            :chi_df => getindex(ls,:chi_df),
            :MNW_P => MNW_P,
            :MNW_P_1s => MNW_P_1s,
            :bp => boot_p,
            :bp1s => boot_p_1s
            )
        )
end

function IMTEST(df::DataFrame, y::AbstractString, 
    xv::AbstractString, cv::AbstractString,fr::AbstractString,
    fc::AbstractString, cc::AbstractString, tm::Int)

    df=df
    y=y 
    xv=xv # xs or xa
    cv=cv # XVARSI or XVARAI
    fr= fr # "schid1n"
    fc= fc # "newid" # null 'newid' or 'clsid'
    cc= cc #"schid1n" # alt1 `schid1n`


    # Specify the variable of interest
    variable_of_interest = Symbol(xv)  # Replace with your variable


    # Convert the `alternative's` column name to a Symbol
    alt = Symbol(cc)
    # Convert the `null's` column name to a Symbol
    null = Symbol(fc)
    # Group the DataFrame by the `alternative`
    grouped_df = groupby(df, alt)

    # Create a dictionary to map group values to group numbers
    group_values = unique(df[!,alt])
    group_numbers = Dict(value => idx for (idx, value) in enumerate(group_values))

    # Create a new column for group numbers and assign values
    df.group_number = [group_numbers[row[alt]] for row in eachrow(df)]


    # *store number of clusters as j
    j = maximum(df.group_number);

    # /*create empty matrices to store the j beta and s.e.*/
    beta = zeros(j,1);
    omega = zeros(j,1);

    # Split the fixed_effects string by spaces to get individual elements
    cvs = split(cv)
    # Mapping each element to the format "fe(element)"
    #formatted_Cs = ["fe($e)" for e in cvs]
    formatted_Cs = ["$e" for e in cvs]
    # making the fixed effect x-variables into an equation-type string 
    rightC = join(formatted_Cs, "+")
    #making the X-variables of interest into an equation-type string
    rightX = join(split(xv, r"\s+"), "+")
    # putting the two strings together to form the RHS of the regression EQN
    right = join([rightX, rightC], "+")   
    right2 = "$right + fe($fr)"

    # joining the LHS and RHS to make the full EQN
    frm = join([y, right2], "~")
    # making the formula readable in reg()
    flm = @eval(@formula($(Meta.parse(frm))))

    for g in 1:j
        
    # /*calculate the beta and s.e. per coarse cluster*/
    # /*cluster s.e. under the null*/

        temp_g = df[df.group_number .== g, :]
        # Create the regression model
        fs = reg(temp_g, flm, Vcov.cluster(null), save = true)

        # Extract coefficients
        coefs = coef(fs)
        # Extract standard errors
        se = stderror(fs)

        # extracting the coefficients' names
        vec = coefnames(fs)
        
        # making the variable of interest searchable in the coefficients' names
        search_str = "$xv"

        # Creating a regular expression pattern that searches for the variable of interest
        pattern = Regex("$(search_str).*")

        # Finding the index of the first occurrence that matches the pattern
        index = findfirst(x -> occursin(pattern, x), vec)

        # Get the coefficient and standard error for the specific variable given the index
        coefficient = coefs[index]
        standard_error = se[index]
        
        # setting the coef and se to the g-th spot in the beta and omega vectors'
        beta[g, 1] = coefficient
        omega[g, 1] = standard_error

    end


    for i in 1:length(beta)
        if beta[i] == NaN
            beta[i] = 0
        end
    end
    for i in 1:length(omega)
        if omega[i] == NaN
            omega[i] = 0
        end
    end

    beta[isnan.(beta)] .= 0
    omega[isnan.(omega)] .= 0

    # /*Calculate the IM (2012) standard error*/  
    # /*it is just the variance of the j beta */
    s2 = var(beta)

    time = tm

    # /*matrix for all the yj estimates*/ 
    ybar = fill(NaN, time, 1)

    # /*replication loop*/
    for k in 1:size(ybar,1)

        #/*multiply the standard errors by standard normals*/
        yj = omega .* quantile.(Normal(), rand(j))

        # /*calculate the average of the yj*/
        avey = mean(yj)

        #/*square the mean differences*/
        sy2 = (yj .- avey) .* (yj .- avey)
        
        # /*sum the squares, divide by (j-1) */
        ybk = (1/(j - 1))*sum(sy2)
        
        # /*store in the matrix*/
        ybar[k, 1] = ybk
    end
    
    # /*calculate the p-value*/
    temp_rej = s2 .< ybar
    temp_U = ones(size(temp_rej, 1), 1)
    temp_sum = transpose(temp_U) * temp_rej
    IM_p = temp_sum / size(temp_rej, 1)

    return (
    Dict(
        :S2 => s2,
        :IM_p => IM_p
        )
    )
end

function lcat(lsmnw::Dict)
    return Dict(
        "H" => getindex(lsmnw, :H),
        "G" => getindex(lsmnw, :G),
        "theta" => join(string.(getindex(lsmnw, :theta)), " "),
        "tau" => getindex(lsmnw, :tau),
        "chi_df" => getindex(lsmnw, :chi_df),
        "MNW_P" => getindex(lsmnw, :MNW_P),
        "bp" => getindex(lsmnw, :bp)
    )
end




##############################################################################
##### EXAMPLE CODE PART 1 ####################################################
##############################################################################


# # Specify the full file path
# file_path = raw"C:\Users\ryan-\OneDrive\Documents\vCarleton Summer RA Work\Julia Project\Tonghui Rcode\star_test.dta"

# # # Read the Stata file into a DataFrame
# df = DataFrame(load(file_path))

# # Testing the VECHTEST function

# B=399;

# XVARS="aide_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4";
# XVARA="small_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4";
# XVARB="treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4  hdg2 hdg3 hdg4";


# y="treadss1";
  
# xs="small_1";
# xa="aide_1"; 
  
# xb="small_1 aide_1";

# fc = "newid";

# cc = "clsid";

# XVARSX="aide_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4 factor(schid1n)";
# XVARAX="small_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4 factor(schid1n)";
# XVARBX="treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4  hdg2 hdg3 hdg4 factor(schid1n)";

# VECHTEST_time = @elapsed begin
# v1 = VECHTEST(df, y, xs, XVARS, "clsid", "schid1n");
# end;
# println("Elapsed time: $VECHTEST_time seconds")

# println(v1)
############################################################################################################################################################
############################################################################################################################################################
## MNWTEST TESTING

# Specify the full file path
# file_path = raw"C:\Users\ryan-\OneDrive\Documents\vCarleton Summer RA Work\Julia Project\Tonghui Rcode\star_test.dta"

# # # Read the Stata file into a DataFrame
# df = DataFrame(load(file_path))


# B=399;

# XVARS="aide_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4";
# XVARA="small_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4";
# XVARB="treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4  hdg2 hdg3 hdg4";


# y="treadss1"; 
  
# xs="small_1";
# xa="aide_1";
  
# xb="small_1 aide_1";

# fc = "newid";

# cc = "clsid";


# elapsed_time = @elapsed begin
# ls1= MNWTEST(df,y,xs,XVARS,"newid","clsid",B)
# ls2= MNWTEST(df,y,xs,XVARS,"newid","schid1n",B)
# ls3= MNWTEST(df,y,xs,XVARS,"clsid","schid1n",B)
# ls4= MNWTEST(df,y,xa,XVARA,"newid","clsid",B);
# ls5= MNWTEST(df,y,xa,XVARA,"newid","schid1n",B)
# ls6= MNWTEST(df,y,xa,XVARA,"clsid","schid1n",B)
# ls7=MNWTEST(df,y,xb,XVARB,"newid","clsid",B)
# ls8=MNWTEST(df,y,xb,XVARB,"newid","schid1n",B)
# ls9=MNWTEST(df,y,xb,XVARB,"clsid","schid1n",B)

# XVARSX="aide_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4 factor(schid1n)"
# XVARAX="small_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4 factor(schid1n)"
# XVARBX="treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4  hdg2 hdg3 hdg4 factor(schid1n)"

# ls10= MNWTEST(df,y,xs,XVARSX,"newid","clsid",B)
# ls11= MNWTEST(df,y,xs,XVARSX,"newid","schid1n",B)
# ls12= MNWTEST(df,y,xs,XVARSX,"clsid","schid1n",B)
# ls13= MNWTEST(df,y,xa,XVARAX,"newid","clsid",B)
# ls14= MNWTEST(df,y,xa,XVARAX,"newid","schid1n",B)
# ls15= MNWTEST(df,y,xa,XVARAX,"clsid","schid1n",B)

# ls16= MNWTEST(df,y,xb,XVARBX,"newid","clsid",B)
# ls17= MNWTEST(df,y,xb,XVARBX,"newid","schid1n",B)
# ls18= MNWTEST(df,y,xb,XVARBX,"clsid","schid1n",B)
    
# end;


# # Print the result
# println("Elapsed time: $elapsed_time seconds")
# println(getindex(ls4, :tau))
# println(getindex(ls6, :tau))

############################################################################################################################################################
############################################################################################################################################################
## IMTEST TESTING
# df = DataFrame(load(file_path));

# y="treadss1";

# xs="small_1";
# xa="aide_1";
# fr="schid1n";

# nulla="newid";
# nullb="clsid";
# alt1="schid1n";

# XVARSI="aide_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4";
# XVARAI="small_1 treadssk male nonwhite teach_nonwhite totexp1 freelunch brys2 brys3 brys5 sbq2 sbq3 sbq4 hdg2 hdg3 hdg4";
# fatr="schid1n";

# tm = 99;
 

# elapsed_time3 = @elapsed begin
    
# im1 = IMTEST(df, y, xs, XVARSI, fr, nulla, alt1, 99);

# df = DataFrame(load(file_path));
# im2 = IMTEST(df, y, xs, XVARSI, fr, "clsid", alt1, 99);

# df = DataFrame(load(file_path));
# im3 = IMTEST(df, y, xa, XVARAI, fr, nulla, alt1, 99);

# df = DataFrame(load(file_path));
# im4 = IMTEST(df, y, xa, XVARAI, fr, "clsid", alt1, 99);
# end;

# println("Elapsed time: $elapsed_time3 seconds")
# println("")
# print(im1)
# println("")
# print(im2)
# println("")
# print(im3)
# println("")
# print(im4)

##############################################################################
##### EXAMPLE CODE PART 2 ####################################################
##############################################################################


# # STUDENT AND EMPLOYED DATASET TESTING

# df2 =  CSV.read("C:/Users/ryan-/OneDrive/Documents/vCarleton Summer RA Work/Julia Project/Tonghui Rcode/min_wage_teen_hours2.csv", DataFrame)

# B=999

# null1 = "newid"
# null2 = "styear"
# null3= "statefip"

# alt1 = "styear"
# alt2 = "statefip"
# alt3 = "region"

# y = "hours2"
# xs = "mw"
# # XVARS = "black female year age statefip educ"
# XVARS = "black female factor(year) factor(age) factor(statefip) factor(educ)"

# elapsed_time = @elapsed begin
# mw1= MNWTEST(df2,y,xs,XVARS,null1,alt1,B)
# mw2 = MNWTEST(df2,y,xs,XVARS,null1,alt2,B)
# mw3= MNWTEST(df2,y,xs,XVARS,null1,alt3,B)
# mw4= MNWTEST(df2,y,xs,XVARS,null2,alt2,B)
# mw5= MNWTEST(df2,y,xs,XVARS,null2,alt3,B)
# mw6= MNWTEST(df2,y,xs,XVARS,null3,alt3,B)
# end;
# println("The elapsed time is: $elapsed_time seconds")
# println(getindex(mw1, :tau))
# println(getindex(mw2, :tau))
# println(getindex(mw3, :tau))
# println(getindex(mw1, :MNW_P_1s))
# println(getindex(mw2, :MNW_P_1s))
# println(getindex(mw3, :MNW_P_1s))
# println(getindex(mw1, :bp1s))
# println(getindex(mw2, :bp1s))
# println(getindex(mw3, :bp1s))

# println(getindex(mw4, :tau))
# println(getindex(mw5, :tau))
# println(getindex(mw6, :tau))
# println(getindex(mw4, :MNW_P_1s))
# println(getindex(mw5, :MNW_P_1s))
# println(getindex(mw6, :MNW_P_1s))
# println(getindex(mw4, :bp1s))
# println(getindex(mw5, :bp1s))
# println(getindex(mw6, :bp1s))