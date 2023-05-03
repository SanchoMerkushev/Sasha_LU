/* CLASS = A */
/*
   This file is generated automatically by the setparams utility.
   It sets the number of processors and the class of the NPB
   in this directory. Do not modify it by hand.   
*/

/* full problem size */
#define ISIZ1  64
#define ISIZ2  64
#define ISIZ3  64

/* number of iterations and how often to print the norm */
#define ITMAX_DEFAULT  250
#define INORM_DEFAULT  250
#define DT_DEFAULT     2.0

#define CONVERTDOUBLE  false
#define COMPILETIME "03 May 2023"
#define NPBVERSION "3.3.1"
#define CS1 "pgcc"
#define CS2 "$(CC)"
#define CS3 "-lm"
#define CS4 "-I../common"
#define CS5 "-g -Wall -O3 -acc -ta=multicore -mcmodel=me..."
#define CS6 "-O3 -acc -ta=multicore -mcmodel=medium"
#define CS7 "randdp"
