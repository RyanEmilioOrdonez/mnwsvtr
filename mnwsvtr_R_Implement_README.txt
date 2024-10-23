# A README file for the R-Implementation of mnwsvtr.jl

# PLEASE REFER TO THE README.md file or the mnwsvtr_README.txt  for detailed descriptions of the VECHTEST, MNWTEST, and IMTEST functions.

This code serve as guide to accessing the mnwsvtr.jl file and its functions through R.

Included in my repository are both an RMarkdown file and a basic R file.

Both files include comments that are aimed to help guide you to easily through the process
of accessing Julia through R with the JuliaCall package.

You may refer to the mnwsvtr.jl README file for info on all of the variables needed to run the 
VECHTEST, MNWTEST, and IMTEST functions.

Below I will include a detailed walkthrough of the code:

Firstly, you will need to install that R packages "haven" and " "JuliaCall", if not previously installed. "haven" is used for importing certain file types such as Stata files. "JuliaCall" is
the package that helps R studio speak to Julia.

Next, you will need to know the location of the julia.exe in your file directory. This will 
usually be a the path such as "/Users/username-/.julia/juliaup/julia-1.10.3+0.x64.w64.mingw32/bin".

Once you have the path, the function julia_setup(path_name, force = TRUE) will establish the connection to the Julia compiler. The "force = TRUE" argument will make it run again until the connection is established, this is something that fixed a problem for me once so I kept it in. 

Run the function "julia_command("pwd()")$out" to find the directory path to your current R file. Make sure that the mnwsvtr.jl file is in that same folder. 

The function "Julia_source("filename.jl")" will check if the desired .jl file can be ran. Make 
sure that any example code is commented out otherwise it will run it which could take time.

You can then test if any functions that were in the .jl file are callable with the command:
"julia_exists("function_name")"; In our case we test if the functions VECHTEST and MNWTEST are
found. The result is a binary TRUE/FALSE.

We then go on to defining variable in R normally, and assign those variables into Julia with the "julia_assign("desired_julia_variable_name", variable_in_R_name)" function.

Once all variables are assigned in Julia, we can then start obtaining the tau and p_values as described in the paper "Testing for the appropriate level of clustering in linear
regression models" by James G. MacKinnon, Morten Ørregaard Nielsen, Matthew D. Webb (2022).




 




