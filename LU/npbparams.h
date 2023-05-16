/* CLASS = B */
/*
   This file is generated automatically by the setparams utility.
   It sets the number of processors and the class of the NPB
   in this directory. Do not modify it by hand.   
*/

/* full problem size */
#define ISIZ1  102
#define ISIZ2  102
#define ISIZ3  102

/* number of iterations and how often to print the norm */
#define ITMAX_DEFAULT  250
#define INORM_DEFAULT  250
#define DT_DEFAULT     2.0

#define CONVERTDOUBLE  false
#define COMPILETIME "16 May 2023"
#define NPBVERSION "3.3.1"
#define CS1 "nvc"
#define CS2 "$(CC)"
#define CS3 "-lm"
#define CS4 "-I../common"
#define CS5 "-O3 -acc -ta=nvidia -Minfo=all  -mcmodel=me..."
#define CS6 "-O3 -acc -ta=nvidia -Minfo=all  -mcmodel=me..."
#define CS7 "randdp"
